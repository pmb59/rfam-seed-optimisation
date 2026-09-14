#!/usr/bin/env python3
"""Select an RNA seed subset using normalized CM probabilities and grouped CV.

Python >= 3.10; NumPy is the only Python dependency. `optimize` also requires
Infernal's cmbuild. Optional CMCompare and cmcalibrate calls use shell=False.

Read METHODOLOGY.md before interpreting results. This is a research reference
implementation: it implements GLOBAL, full-sequence generative probabilities,
not Infernal's local database-search score. A bounded subset search is a
heuristic, not a certificate that the best of 2**N subsets has been found.

Examples:
  python cm_subset_optimizer.py selftest
  python cm_subset_optimizer.py optimize --alignment candidates.sto --out run
  python cm_subset_optimizer.py compare --a A.cm --b B.cm --out comparison.json
  python cm_subset_optimizer.py score --cm A.cm --fasta homologs.fa --out scores.tsv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

VERSION = "0.1.0"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ALPHABET = "ACGU"
GAPS = ".-_"
NEG_INF = -float("inf")
STATE_TYPES = {"S", "D", "B", "E", "ML", "MR", "MP", "IL", "IR"}


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, file=sys.stderr, flush=True)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def json_safe(value):
    """JSON does not have IEEE infinities; retain them explicitly as strings."""
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return str(float(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical(sequence: str, *, aligned: bool = False) -> str:
    sequence = sequence.upper().replace("T", "U")
    allowed = set(ALPHABET + (GAPS if aligned else ""))
    bad = set(sequence) - allowed
    if bad:
        raise ValueError(f"Unsupported sequence symbols {sorted(bad)}. Use A/C/G/U (or T); "
                         "resolve ambiguous/missing residues before running this model.")
    return sequence


def read_fasta(path: Path) -> OrderedDict[str, str]:
    records: OrderedDict[str, str] = OrderedDict()
    current = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].split()[0] if line[1:].split() else ""
            if not current or current in records:
                raise ValueError("FASTA identifiers must be nonempty and unique.")
            records[current] = ""
        elif current is None:
            raise ValueError("FASTA sequence encountered before its identifier.")
        else:
            records[current] += canonical(line)
    if not records or any(not value for value in records.values()):
        raise ValueError("Provide a nonempty FASTA with nonempty sequences.")
    return records


@dataclass
class Alignment:
    sequences: OrderedDict[str, str]
    ss_cons: str
    rf: str

    @classmethod
    def read(cls, path: Path) -> "Alignment":
        """Read one Stockholm alignment, including interleaved sequence blocks."""
        sequences: OrderedDict[str, str] = OrderedDict()
        ss, rf, starts, ends = [], [], 0, 0
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if line.startswith("# STOCKHOLM"):
                starts += 1
            elif line == "//":
                ends += 1
            elif line.startswith("#=GC"):
                fields = line.split()
                if len(fields) != 3:
                    raise ValueError("Malformed #=GC annotation.")
                if fields[1] == "SS_cons":
                    ss.append(fields[2])
                elif fields[1] == "RF":
                    rf.append(fields[2])
            elif line and not line.startswith("#"):
                if ends:
                    raise ValueError("Content follows the Stockholm terminator.")
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError("Expected sequence identifier and aligned sequence.")
                identifier, segment = fields
                sequences[identifier] = sequences.get(identifier, "") + canonical(segment, aligned=True)
        if starts != 1 or ends != 1 or not sequences:
            raise ValueError("Expected exactly one complete Stockholm alignment.")
        result = cls(sequences, "".join(ss), "".join(rf))
        result.validate()
        return result

    def validate(self) -> None:
        lengths = {len(s) for s in self.sequences.values()}
        if len(lengths) != 1 or not self.ss_cons or not self.rf:
            raise ValueError("Sequences must be aligned; SS_cons and RF are required.")
        length = lengths.pop()
        if len(self.ss_cons) != length or len(self.rf) != length:
            raise ValueError("Sequence, SS_cons and RF lengths differ.")
        if not any(c not in ".-_~" for c in self.rf):
            raise ValueError("RF defines no consensus columns.")
        # Crossing/pseudoknot annotation is rejected rather than silently removed.
        stack = []
        closes = {">": "<", ")": "(", "]": "[", "}": "{"}
        for index, symbol in enumerate(self.ss_cons):
            if symbol in "<([{":
                stack.append((symbol, index))
            elif symbol in closes:
                if not stack or stack[-1][0] != closes[symbol]:
                    raise ValueError("SS_cons is unbalanced or contains crossing pairs.")
                _, left = stack.pop()
                if self.rf[left] in ".-_~" or self.rf[index] in ".-_~":
                    raise ValueError("A base pair involves an RF insert column; curate fixed RF/SS_cons first.")
            elif symbol not in ".,:_-~":
                raise ValueError("Unsupported SS_cons annotation (including letter-coded pseudoknots).")
        if stack:
            raise ValueError("Unbalanced SS_cons.")
        if any(not s.translate(str.maketrans("", "", GAPS)) for s in self.sequences.values()):
            raise ValueError("All-gap candidate sequence.")

    @property
    def raw(self) -> OrderedDict[str, str]:
        return OrderedDict((name, s.translate(str.maketrans("", "", GAPS)))
                           for name, s in self.sequences.items())

    def write_subset(self, path: Path, identifiers: Iterable[str]) -> None:
        keep = set(identifiers)
        if not keep or not keep <= self.sequences.keys():
            raise ValueError("Invalid or empty seed subset.")
        # Keep ALL original columns, including columns that become all-gap.
        # Existing sequence weights are intentionally not copied: cmbuild recomputes them.
        width = max(map(len, keep)) + 2
        lines = ["# STOCKHOLM 1.0", "#=GF ID selected_seed"]
        lines.extend(f"{name:<{width}}{seq}" for name, seq in self.sequences.items() if name in keep)
        lines.extend([f"#=GC SS_cons {self.ss_cons}", f"#=GC RF {self.rf}", "//"])
        path.write_text("\n".join(lines) + "\n")


@dataclass
class State:
    kind: str
    children: tuple[int, ...]
    transition: np.ndarray
    emission: np.ndarray


def probabilities(log_odds: list[str], background: np.ndarray) -> np.ndarray:
    scores = np.array([NEG_INF if x == "*" else float(x) for x in log_odds])
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("Invalid probability field in CM.")
    values = np.exp2(scores) * background
    total = float(values.sum())
    if total == 0:
        return values
    # ASCII CM probabilities are rounded. Renormalize EACH categorical distribution.
    if not math.isfinite(total) or abs(total - 1.0) > 0.02:
        raise ValueError(f"CM probability distribution sums to {total:g}; unsupported or corrupt format.")
    return values / total


class CM:
    """A normalized global CM grammar, reconstructed from Infernal 1/a ASCII.

    Only insert self-loops are allowed; all other children must have larger
    state indices. B has two deterministic children. These properties permit
    exact bottom-up Inside evaluation without iterative fixed-point solving.
    """

    def __init__(self, states: list[State], name: str = "CM", metadata: dict | None = None):
        self.states, self.name, self.metadata = states, name, metadata or {}
        self.validate()
        self.lt = [self.log_probabilities(s.transition) for s in states]
        self.le = [self.log_probabilities(s.emission) for s in states]

    @staticmethod
    def log_probabilities(values: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log2(values)

    @classmethod
    def read(cls, path: Path) -> "CM":
        lines = path.read_text().splitlines()
        if sum(line.startswith("INFERNAL") for line in lines) != 1:
            raise ValueError("Use one CM per file, not a multi-model library.")
        if not lines or not lines[0].startswith("INFERNAL1/a"):
            raise ValueError("Only INFERNAL1/a ASCII is supported; convert other formats with cmconvert.")
        header, states, section, terminated = {}, [], False, False
        null = None
        for line in lines[1:]:
            fields = line.split()
            if not fields:
                continue
            if not section:
                if fields == ["CM"]:
                    section = True
                    if header.get("ALPH") != "RNA" or null is None:
                        raise ValueError("Expected ALPH RNA and a NULL distribution.")
                else:
                    header[fields[0]] = " ".join(fields[1:])
                    if fields[0] == "NULL":
                        if len(fields) != 5:
                            raise ValueError("Expected four NULL entries.")
                        null = probabilities(fields[1:], np.full(4, 0.25))
                continue
            if fields[0] == "//":
                terminated = True
                break  # The following HMMER filter model is NOT part of the CM grammar.
            if fields[0] == "[":
                continue
            kind = fields[0]
            if kind not in STATE_TYPES or len(fields) < 10:
                raise ValueError(f"Unrecognized CM state line: {line[:100]}")
            index, first, count = int(fields[1]), int(fields[4]), int(fields[5])
            if index != len(states):
                raise ValueError("CM states are not consecutively indexed.")
            ne = 16 if kind == "MP" else 4 if kind in {"ML", "MR", "IL", "IR"} else 0
            nt = 0 if kind in {"B", "E"} else count
            if len(fields) != 10 + nt + ne:
                raise ValueError(f"Unexpected field count for CM state {index}.")
            if kind == "B":
                children, transition = (first, count), np.array([])
            elif kind == "E":
                children, transition = (), np.array([])
            else:
                children = tuple(range(first, first + count))
                transition = probabilities(fields[10:10 + nt], np.ones(nt))
            background = np.outer(null, null).ravel() if ne == 16 else null
            emission = probabilities(fields[10 + nt:], background) if ne else np.array([])
            states.append(State(kind, children, transition, emission))
        if not terminated or len(states) != int(header.get("STATES", -1)):
            raise ValueError("Incomplete CM state section.")
        return cls(states, header.get("NAME", path.stem), header)

    def validate(self) -> None:
        if not self.states or self.states[0].kind != "S":
            raise ValueError("CM must start at an S state.")
        reachable = {0}
        for v, state in enumerate(self.states):
            if state.kind not in STATE_TYPES:
                raise ValueError("Unknown CM state type.")
            for child in state.children:
                if not 0 <= child < len(self.states) or child < v:
                    raise ValueError("Unsupported non-forward CM transition.")
                if child == v and state.kind not in {"IL", "IR"}:
                    raise ValueError("Only emitting insert self-loops are supported.")
            if state.kind == "B":
                if len(state.children) != 2:
                    raise ValueError("B state must have two children.")
            elif state.kind == "E":
                if state.children:
                    raise ValueError("E state has children.")
            elif len(state.children) != len(state.transition):
                raise ValueError("Transition/child mismatch.")
            if v not in reachable:
                continue  # Detached unreachable states need not have nonzero distributions.
            if state.kind == "B":
                reachable.update(state.children)
            elif state.kind != "E":
                if not np.isclose(state.transition.sum(), 1.0):
                    raise ValueError("Reachable state has an improper transition distribution.")
                reachable.update(c for c, p in zip(state.children, state.transition) if p > 0)
                self_mass = sum(p for c, p in zip(state.children, state.transition) if c == v)
                if self_mass >= 1:
                    raise ValueError("Non-terminating insert state.")
            expected_ne = 16 if state.kind == "MP" else 4 if state.kind in {"ML", "MR", "IL", "IR"} else 0
            if len(state.emission) != expected_ne or (expected_ne and not np.isclose(state.emission.sum(), 1.0)):
                raise ValueError("Invalid reachable emission distribution.")

    def log_probability(self, sequence: str, memory_mb: float = 1024) -> float:
        """Return log2 P(sequence), summing ALL global parses without banding.

        A[v,i,d] is the log probability that state v generates x[i:i+d].
        The recurrence is vectorized across start positions; self-loops consume
        residues, so shorter spans have already been evaluated when required.
        """
        sequence = canonical(sequence)
        x = np.array([ALPHABET.index(c) for c in sequence], dtype=np.int64)
        length = len(x)
        required = (len(self.states) + 12) * (length + 1) ** 2 * 8 / 2**20
        if required > memory_mb:
            raise MemoryError(f"Inside array needs about {required:.1f} MiB plus overhead; increase --memory-mb.")
        a = np.full((len(self.states), length + 1, length + 1), NEG_INF)
        valid = np.arange(length + 1)[:, None] + np.arange(length + 1)[None, :] <= length
        for v in range(len(self.states) - 1, -1, -1):
            state = self.states[v]
            if state.kind == "E":
                a[v, :, 0] = 0.0
                continue
            if state.kind == "B":
                left, right = state.children
                for d in range(length + 1):
                    indices = np.arange(length - d + 1)
                    total = np.full(len(indices), NEG_INF)
                    for split in range(d + 1):
                        total = np.logaddexp2(total, a[left, indices, split] + a[right, indices + split, d - split])
                    a[v, indices, d] = total
                continue
            dl = int(state.kind in {"ML", "IL", "MP"})
            dr = int(state.kind in {"MR", "IR", "MP"})
            if v not in state.children:
                # Without a self-loop every child is already complete. Evaluate
                # all spans together instead of looping in Python over lengths.
                total = np.full((length + 1, length + 1), NEG_INF)
                for child, lp in zip(state.children, self.lt[v]):
                    if lp != NEG_INF:
                        total = np.logaddexp2(total, lp + a[child])
                if not dl and not dr:
                    a[v] = total
                elif state.kind == "MP" and length >= 2:
                    i = np.arange(length - 1)
                    right_indices = np.minimum(i[:, None] + i[None, :] + 1, length - 1)
                    a[v, :length - 1, 2:] = total[1:length, :length - 1] + self.le[v][4 * x[:length - 1, None] + x[right_indices]]
                elif dl and not dr and length:
                    a[v, :length, 1:] = total[1:, :length] + self.le[v][x][:, None]
                elif dr and not dl and length:
                    i = np.arange(length)
                    right_indices = np.minimum(i[:, None] + i[None, :], length - 1)
                    a[v, :length, 1:] = total[:length, :length] + self.le[v][x[right_indices]]
                a[v][~valid] = NEG_INF
                continue
            for d in range(dl + dr, length + 1):
                indices = np.arange(length - d + 1)
                total = np.full(len(indices), NEG_INF)
                for child, lp in zip(state.children, self.lt[v]):
                    if lp != NEG_INF:
                        total = np.logaddexp2(total, lp + a[child, indices + dl, d - dl - dr])
                if state.kind == "MP":
                    total += self.le[v][4 * x[indices] + x[indices + d - 1]]
                elif dl:
                    total += self.le[v][x[indices]]
                elif dr:
                    total += self.le[v][x[indices + d - 1]]
                a[v, indices, d] = total
        return float(a[0, 0, length])

    def sample(self, rng: np.random.Generator, max_length: int = 500, max_steps: int = 1000000) -> str:
        """Sample from exactly the grammar scored above, with no rejection/truncation.

        If a limit is exceeded, abort instead of discarding and redrawing: redraws
        would condition the distribution and invalidate the reported JS estimate.
        """
        stack: list[int | str] = [0]
        output, emitted, steps = [], 0, 0
        while stack:
            item = stack.pop()
            if isinstance(item, str):
                output.append(item)
                continue
            steps += 1
            if steps > max_steps:
                raise RuntimeError("Sampling exceeded the step limit; no truncated sample was used.")
            state = self.states[item]
            if state.kind == "E":
                continue
            if state.kind == "B":
                stack.extend(reversed(state.children))
                continue
            child = state.children[int(rng.choice(len(state.children), p=state.transition))]
            if not len(state.emission):
                stack.append(child)
                continue
            symbol = int(rng.choice(len(state.emission), p=state.emission))
            if state.kind == "MP":
                stack.extend([ALPHABET[symbol % 4], child, ALPHABET[symbol // 4]])
                emitted += 2
            elif state.kind in {"ML", "IL"}:
                stack.extend([child, ALPHABET[symbol]])
                emitted += 1
            else:
                stack.extend([ALPHABET[symbol], child])
                emitted += 1
            if emitted > max_length:
                raise RuntimeError("Sample exceeds --max-length. Increase the limit; samples are never censored.")
        return "".join(output)


def js_divergence(a: CM, b: CM, samples: int, seed: int, memory_mb: float,
                  max_length: int) -> dict:
    """Independent samples from A and B estimate JS and its Monte Carlo SE."""
    rng = np.random.default_rng(seed)
    terms, kl_terms = [[], []], [[], []]
    cache: dict[str, tuple[float, float]] = {}
    for side, model in enumerate((a, b)):
        for _ in range(samples):
            x = model.sample(rng, max_length)
            if x not in cache:
                cache[x] = (a.log_probability(x, memory_mb), b.log_probability(x, memory_mb))
            la, lb = cache[x]
            own, other = (la, lb) if side == 0 else (lb, la)
            if not math.isfinite(own):
                raise ArithmeticError("A model assigned zero probability to its own sample.")
            mixture = float(np.logaddexp2(la, lb)) - 1.0
            terms[side].append(own - mixture)
            kl_terms[side].append(own - other)
    value = 0.5 * (float(np.mean(terms[0])) + float(np.mean(terms[1])))
    se = 0.5 * math.sqrt(float(np.var(terms[0], ddof=1) + np.var(terms[1], ddof=1)) / samples)
    # Do not clip negative Monte Carlo estimates to zero or hide uncertainty.
    return {"js_bits": value, "mc_se_bits": se,
            "mc_normal_95_low": value - 1.96 * se, "mc_normal_95_high": value + 1.96 * se,
            "kl_a_to_b_bits": float(np.mean(kl_terms[0])),
            "kl_b_to_a_bits": float(np.mean(kl_terms[1])),
            "samples_per_model": samples, "seed": seed,
            "configuration": "normalized_global_full_sequence", "distinct_sampled_sequences": len(cache)}


def sequence_groups(alignment: Alignment, supplied: Path | None, threshold: float) -> dict[str, str]:
    names = list(alignment.sequences)
    if supplied:
        result = {}
        with supplied.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["sequence_id", "group"]:
                raise ValueError("Group TSV header must be: sequence_id<TAB>group")
            for row in reader:
                identifier, group = row["sequence_id"], row["group"]
                if identifier in result or not group:
                    raise ValueError("Duplicate sequence or empty group in group TSV.")
                result[identifier] = group
        if set(result) != set(names):
            raise ValueError("Group TSV must contain every candidate exactly once and no other IDs.")
    else:
        # Single-linkage components ensure any pair >= threshold stays together.
        parent = list(range(len(names)))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in itertools.combinations(range(len(names)), 2):
            x, y = alignment.sequences[names[i]], alignment.sequences[names[j]]
            pairs = [(u, v) for u, v in zip(x, y) if u not in GAPS or v not in GAPS]
            identity = sum(u == v and u not in GAPS for u, v in pairs) / len(pairs)
            if identity >= threshold:
                parent[root(j)] = root(i)
        labels = {r: f"cluster_{k + 1:03d}" for k, r in enumerate(sorted({root(i) for i in range(len(names))}))}
        result = {name: labels[root(i)] for i, name in enumerate(names)}
    # Even manually supplied group assignments must not put identical strings apart.
    seen = {}
    for name, seq in alignment.raw.items():
        if seq in seen and seen[seq] != result[name]:
            raise ValueError("Identical sequences occur in different groups; merge those groups.")
        seen[seq] = result[name]
    return result


def make_folds(names: list[str], groups: dict[str, str], count: int, seed: int) -> list[list[str]]:
    blocks = defaultdict(list)
    for name in names:
        blocks[groups[name]].append(name)
    count = min(count, len(blocks))
    if count < 2:
        raise ValueError("At least two independent groups are required for inner CV.")
    items = sorted(blocks.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: -len(item[1]))  # stable random tie breaking
    folds = [[] for _ in range(count)]
    for _, members in items:
        destination = min(range(count), key=lambda i: len(folds[i]))
        folds[destination].extend(members)
    return folds


def group_mean(scores: dict[str, float], groups: dict[str, str]) -> float:
    values = defaultdict(list)
    for name, score in scores.items():
        values[groups[name]].append(score)
    return float(np.mean([np.mean(x) for x in values.values()]))


def run_command(command: list[str], log_path: Path, timeout: float) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        try:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                     text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Command timed out. See {log_path}") from error
    if process.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-15:])
        raise RuntimeError(f"Command failed ({process.returncode}): {shlex.join(command)}\n{tail}")


class Backend:
    """Build once per unique subset; cache per-sequence likelihoods on disk."""

    def __init__(self, alignment: Alignment, output: Path, args, tool_version: str):
        self.alignment, self.raw, self.output, self.args = alignment, alignment.raw, output, args
        self.base = digest({"version": VERSION, "source_sha256": SOURCE_SHA256, "sequences": alignment.sequences,
                            "ss": alignment.ss_cons, "rf": alignment.rf,
                            "cmbuild": tool_version, "weighting": args.weighting})
        self.models: OrderedDict[str, CM] = OrderedDict()
        self.builds, self.score_calls = 0, 0

    def key(self, names: Iterable[str]) -> str:
        return digest([self.base, sorted(names)])[:24]

    def model_path(self, names: Iterable[str]) -> Path:
        return self.output / "cache" / self.key(names) / "model.cm"

    def model(self, names: Iterable[str]) -> CM:
        names = tuple(sorted(names))
        key = self.key(names)
        if key in self.models:
            self.models.move_to_end(key)
            return self.models[key]
        path = self.model_path(names)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            seed = path.with_name("seed.sto")
            self.alignment.write_subset(seed, names)
            temporary = path.with_name("building.cm")
            command = [self.args.cmbuild, "-F", "--hand", "--" + self.args.weighting,
                       "-n", "subset_" + key, str(temporary.resolve()), str(seed.resolve())]
            log(f"Building {len(names)}-sequence seed {key[:8]}")
            run_command(command, path.with_name("cmbuild.log"), self.args.command_timeout)
            model = CM.read(temporary)  # Validate before making the cache entry visible.
            temporary.replace(path)
            write_json(path.with_name("build.json"), {"ids": names, "command": command, "metadata": model.metadata})
            self.builds += 1
        else:
            model = CM.read(path)
        self.models[key] = model
        if len(self.models) > 32:
            self.models.popitem(last=False)
        return model

    def score(self, train: Iterable[str], test: Iterable[str]) -> dict[str, float]:
        train, test = tuple(sorted(train)), list(test)
        model = self.model(train)
        cache_path = self.model_path(train).with_name("scores.json")
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        changed = False
        for name in test:
            if name not in cached:
                cached[name] = model.log_probability(self.raw[name], self.args.memory_mb)
                self.score_calls += 1
                changed = True
        if changed:
            write_json(cache_path, cached)
        return {name: float(cached[name]) for name in test}


class Objective:
    """Evaluate every candidate mask on the SAME fixed withheld sequences.

    In fold k, only S minus V_k is used for construction, even if S contains
    members of V_k. Membership itself is tuned on this inner CV objective;
    an outer loop is needed for a less optimistic estimate of the procedure.
    """

    def __init__(self, backend: Backend, pool: list[str], groups: dict[str, str],
                 folds: list[list[str]], args, scope: str):
        self.backend, self.pool, self.groups, self.folds = backend, pool, groups, folds
        self.args, self.scope = args, scope
        self.memo: dict[tuple[str, ...], dict] = {}
        self.evaluations = 0
        self.history = backend.output / f"{scope}_evaluations.jsonl"
        self.history.write_text("")  # Resume replays deterministic search using model/score caches.

    def feasible(self, subset: Iterable[str]) -> bool:
        s = set(subset)
        return (len(s) >= self.args.min_size and s <= set(self.pool)
                and all(s - set(fold) for fold in self.folds))

    def evaluate(self, subset: Iterable[str]) -> dict | None:
        key = tuple(sorted(subset))
        if key in self.memo:
            return self.memo[key]
        if not self.feasible(key) or self.evaluations >= self.args.max_evaluations:
            return None
        self.evaluations += 1
        scores, fold_sizes = {}, []
        for heldout in self.folds:
            train = sorted(set(key) - set(heldout))
            assert not set(train) & set(heldout)
            fold_sizes.append(len(train))
            scores.update(self.backend.score(train, heldout))
        raw = group_mean(scores, self.groups)
        result = {"ids": list(key), "n": len(key), "cv_mean_log2p": raw,
                  "objective": raw - self.args.size_penalty * len(key),
                  "fold_training_sizes": fold_sizes, "evaluation": self.evaluations}
        self.memo[key] = result
        with self.history.open("a") as handle:
            handle.write(json.dumps(json_safe(result), allow_nan=False) + "\n")
        log(f"{self.scope}: evaluation {self.evaluations}/{self.args.max_evaluations}, "
            f"N={len(key)}, CV={raw:.5f} bits, objective={result['objective']:.5f}")
        return result


def better(a: dict | None, b: dict | None, tolerance: float) -> bool:
    if a is None:
        return False
    if b is None:
        return True
    if a["objective"] > b["objective"] + tolerance:
        return True
    return abs(a["objective"] - b["objective"]) <= tolerance and a["n"] < b["n"]


def select_subset(objective: Objective, args, seed: int) -> dict:
    """Budgeted add/delete/swap hill climbing, or exhaustive feasible enumeration."""
    rng = random.Random(seed)
    pool = sorted(objective.pool)
    full = objective.evaluate(pool)  # Always include the full-pool baseline.
    if full is None or not math.isfinite(full["objective"]):
        raise ValueError("Full-pool CV is infeasible/nonfinite. Check group counts and input sequences.")
    best = full
    edges = []
    complete = True
    if args.search == "exhaustive":
        if len(pool) > 20:
            raise ValueError("Exhaustive search is limited to 20 candidates; use local search for larger pools.")
        for size in range(args.min_size, len(pool) + 1):
            for subset in itertools.combinations(pool, size):
                if not objective.feasible(subset):
                    continue
                item = objective.evaluate(subset)
                if item is None:
                    complete = False
                    break
                if better(item, best, args.tie_tolerance):
                    best = item
            if not complete:
                break
    else:
        starts = [pool]
        for _ in range(args.restarts - 1):
            # Generate alternative starting masks, retaining CV feasibility.
            candidates = None
            for _attempt in range(100):
                size = rng.randint(args.min_size, len(pool))
                proposal = sorted(rng.sample(pool, size))
                if objective.feasible(proposal):
                    candidates = proposal
                    break
            starts.append(candidates or pool)
        for start in starts:
            current = objective.evaluate(start)
            if current is None:
                complete = False
                break
            if better(current, best, args.tie_tolerance):
                best = current
            for _round in range(args.max_rounds):
                inside = set(current["ids"])
                outside = set(pool) - inside
                neighbors = [tuple(sorted(inside - {x})) for x in sorted(inside)]
                neighbors += [tuple(sorted(inside | {x})) for x in sorted(outside)]
                swaps = list(itertools.product(sorted(inside), sorted(outside)))
                rng.shuffle(swaps)
                neighbors += [tuple(sorted((inside - {x}) | {y})) for x, y in swaps[:args.swaps]]
                rng.shuffle(neighbors)  # Avoid identifier-order bias under a finite budget.
                next_item = current
                for neighbor in neighbors:
                    item = objective.evaluate(neighbor)
                    if item is None:
                        if objective.feasible(neighbor) and tuple(neighbor) not in objective.memo:
                            complete = False
                        continue
                    if better(item, next_item, args.tie_tolerance):
                        next_item = item
                    if better(item, best, args.tie_tolerance):
                        best = item
                if next_item is current:
                    break
                edges.append({"from": current["ids"], "to": next_item["ids"]})
                current = next_item
                if objective.evaluations >= args.max_evaluations:
                    complete = False
                    break
            else:
                complete = False
    return {"selected": best, "full_pool_baseline": full, "evaluations": objective.evaluations,
            "search": args.search, "finished_requested_search": complete,
            "globally_optimal_over_feasible_masks": args.search == "exhaustive" and complete,
            "accepted_edges": edges}


def paired_outer_summary(rows: list[dict], bootstrap: int, seed: int) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(float(row["delta_log2p"]))
    means = np.array([np.mean(values) for _, values in sorted(groups.items())])
    result = {"independent_groups": len(means), "macro_mean_delta_log2p": float(means.mean())}
    if len(means) >= 2 and np.isfinite(means).all():
        rng = np.random.default_rng(seed)
        estimates = [float(rng.choice(means, size=len(means), replace=True).mean()) for _ in range(bootstrap)]
        result["paired_group_bootstrap_95"] = np.quantile(estimates, [0.025, 0.975]).tolist()
    else:
        result["paired_group_bootstrap_95"] = None
    result["interval_scope"] = "Descriptive group resampling of outer predictions; does not rerun model selection."
    return result


def executable_version(command: str) -> str:
    resolved = shutil.which(command)
    if not resolved:
        raise ValueError(f"Executable not found: {command}")
    result = subprocess.run([resolved, "-h"], text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError(f"Cannot run {command} -h.")
    return result.stdout


def comparator_template(text: str) -> list[str]:
    value = json.loads(text)
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
        raise ValueError("--cmcompare-command must be a JSON array of command arguments.")
    if not any("{query}" in x for x in value) or not any("{target}" in x for x in value):
        raise ValueError("CMCompare command must contain {query} and {target} placeholders.")
    if not shutil.which(value[0]):
        raise ValueError(f"CMCompare executable not found: {value[0]}")
    return value


def parse_comparison(text: str, format_name: str) -> tuple[float, float, float]:
    if format_name == "json":
        value = json.loads(text)
        qa, qb = float(value["score_query"]), float(value["score_target"])
        link = float(value.get("link_score", min(qa, qb)))
    else:
        # Documented non-verbose hsCMCompare output: query target score1 score2 ...
        # Do not silently interpret weak-pair database exports or verbose reports.
        candidates = []
        for line in text.splitlines():
            fields = line.split()
            if len(fields) != 9 or line.startswith("#"):
                continue
            try:
                candidates.append((float(fields[2]), float(fields[3])))
            except ValueError:
                continue
        if len(candidates) != 1:
            raise ValueError("Expected one non-verbose CMCompare result; use a JSON adapter for other versions.")
        qa, qb = candidates[0]
        link = min(qa, qb)
    if not all(math.isfinite(v) for v in (qa, qb, link)) or abs(link - min(qa, qb)) > 0.02:
        raise ValueError("CMCompare returned invalid or inconsistent scores.")
    return qa, qb, link


def compare_competitors(queries: dict[str, Path], targets: list[Path], output: Path, args) -> list[dict]:
    template = comparator_template(args.cmcompare_command)
    rows = []
    for query_label, query in queries.items():
        for index, target in enumerate(targets):
            # Fixed local filenames also avoid ambiguity in legacy space-separated output.
            directory = output / f"{query_label}_{index:04d}"
            directory.mkdir(parents=True, exist_ok=True)
            qcopy, tcopy = directory / "query.cm", directory / "target.cm"
            shutil.copyfile(query, qcopy)
            shutil.copyfile(target, tcopy)
            command = [x.replace("{query}", "query.cm").replace("{target}", "target.cm") for x in template]
            command[0] = str(Path(shutil.which(command[0])).resolve())
            # Resolve explicit auxiliary script paths before changing the working directory.
            for i in range(1, len(command)):
                if command[i] not in {"query.cm", "target.cm"} and Path(command[i]).is_file():
                    command[i] = str(Path(command[i]).resolve())
            log(f"CMCompare: {query_label} vs {target.name}")
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True,
                                    timeout=args.command_timeout, check=False)
            (directory / "stdout.txt").write_text(result.stdout)
            (directory / "stderr.txt").write_text(result.stderr)
            write_json(directory / "command.json", command)
            if result.returncode:
                raise RuntimeError(f"CMCompare failed; see {directory}")
            qa, qb, link = parse_comparison(result.stdout, args.cmcompare_format)
            rows.append({"query": query_label, "target": str(target.resolve()),
                         "target_name": CM.read(target).name,
                         "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                         "score_query": qa, "score_target": qb, "link_score": link})
    return rows


def optimizer(args) -> dict:
    alignment = Alignment.read(args.alignment)
    names = list(alignment.sequences)
    if len(names) < args.min_size:
        raise ValueError("Fewer candidates than --min-size.")
    if max(map(len, alignment.raw.values())) > args.max_length:
        raise ValueError("An input sequence exceeds --max-length.")
    groups = sequence_groups(alignment, args.groups, args.identity_threshold)
    version = executable_version(args.cmbuild)
    if args.calibrate:
        executable_version(args.cmcalibrate)
    if args.competitors:
        comparator_template(args.cmcompare_command)
        for path in args.competitors:
            CM.read(path)  # Fail before expensive selection if inputs are unsupported.
    outer = make_folds(names, groups, args.outer_folds, args.seed) if args.outer_folds else []
    for fold in outer:
        training = set(names) - set(fold)
        if len(training) < args.min_size or len({groups[x] for x in training}) < 2:
            raise ValueError("An outer training fold is too small or has <2 groups. Change grouping/fold design.")
    make_folds(names, groups, args.inner_folds, args.seed)  # Validate full-pool inner CV.
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in {"out", "resume", "func"} and not key.startswith("_")}
    config["competitors"] = [str(p.resolve()) for p in (args.competitors or [])]
    competitor_hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.competitors or [])}
    fingerprint = digest({"version": VERSION, "source_sha256": SOURCE_SHA256,
                          "numpy_version": np.__version__, "alignment": alignment.sequences, "rf": alignment.rf,
                          "ss": alignment.ss_cons, "groups": groups, "config": config, "cmbuild": version,
                          "competitor_hashes": competitor_hashes})
    args.out = args.out.resolve()
    manifest = args.out / "manifest.json"
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume or not manifest.exists():
            raise ValueError("Output directory is nonempty. Use a new directory, or --resume for the same run.")
        if json.loads(manifest.read_text()).get("fingerprint") != fingerprint:
            raise ValueError("Resume configuration/input/version differs; use a new output directory.")
    args.out.mkdir(parents=True, exist_ok=True)
    args._run_started = True
    write_json(manifest, {"program_version": VERSION, "fingerprint": fingerprint,
                          "source_sha256": SOURCE_SHA256, "competitor_hashes": competitor_hashes,
                          "configuration": config, "cmbuild_help": version,
                          "numpy_version": np.__version__, "python_version": sys.version,
                          "status": "running", "probability_model": "normalized_global_full_sequence"})
    write_tsv(args.out / "groups.tsv", [{"sequence_id": n, "group": groups[n]} for n in names],
              ["sequence_id", "group"])
    backend = Backend(alignment, args.out, args, version)
    predictions, fold_reports = [], []
    for index, test in enumerate(outer):
        pool = [x for x in names if x not in set(test)]
        folds = make_folds(pool, groups, args.inner_folds, args.seed + 101 + index)
        scope = f"outer_{index + 1:02d}"
        write_json(args.out / f"{scope}_folds.json", {"training_pool": pool, "outer_test": test, "inner": folds})
        objective = Objective(backend, pool, groups, folds, args, scope)
        selected = select_subset(objective, args, args.seed + 1001 + index)
        train = selected["selected"]["ids"]
        assert not set(train) & set(test)
        # These outer scores are never passed to the search or its tie-breaking.
        selected_scores = backend.score(train, test)
        baseline_scores = backend.score(pool, test)
        for identifier in test:
            predictions.append({"sequence_id": identifier, "group": groups[identifier], "fold": index + 1,
                                "selected_log2p": selected_scores[identifier], "baseline_log2p": baseline_scores[identifier],
                                "delta_log2p": selected_scores[identifier] - baseline_scores[identifier]})
        write_json(args.out / f"{scope}_selection.json", selected)
        fold_reports.append({"fold": index + 1, "selected_ids": train, "n": len(train),
                             "outer_test_ids": test, "evaluations": selected["evaluations"]})
        write_tsv(args.out / "outer_predictions.tsv", predictions,
                  ["sequence_id", "group", "fold", "selected_log2p", "baseline_log2p", "delta_log2p"])
    final_folds = make_folds(names, groups, args.inner_folds, args.seed + 9001)
    write_json(args.out / "final_folds.json", final_folds)
    final_objective = Objective(backend, names, groups, final_folds, args, "final")
    final = select_subset(final_objective, args, args.seed + 10001)
    selected_ids = final["selected"]["ids"]
    selected_model, full_model = backend.model(selected_ids), backend.model(names)
    shutil.copyfile(backend.model_path(selected_ids), args.out / "selected.cm")
    shutil.copyfile(backend.model_path(names), args.out / "full_pool.cm")
    alignment.write_subset(args.out / "selected.sto", selected_ids)
    (args.out / "selected.ids.txt").write_text("\n".join(selected_ids) + "\n")
    write_json(args.out / "selection.json", final)
    summary = {"program_version": VERSION, "selected_ids": selected_ids, "n_selected": len(selected_ids),
               "n_candidates": len(names), "groups": len(set(groups.values())),
               "final_inner_cv": final["selected"], "full_pool_inner_cv": final["full_pool_baseline"],
               "selection_claim": "Best evaluated feasible mask under the configured search and objective.",
               "outer_folds": fold_reports,
               "outer_evaluation": paired_outer_summary(predictions, args.bootstrap, args.seed + 7001) if predictions else None,
               "outer_status": "completed" if predictions else "not_run; inner score is selection-biased",
               "js_status": "pending" if args.js_samples else "not_requested",
               "cmcompare_status": "pending" if args.competitors else "not_requested",
               "calibrated": False, "status": "selection_complete"}
    write_json(args.out / "summary.json", summary)
    stability = []
    if args.js_samples:
        pairs = [("selected_vs_full_pool", selected_ids, names)]
        if args.js_path:
            pairs.extend((f"accepted_step_{i + 1}", edge["from"], edge["to"])
                         for i, edge in enumerate(final["accepted_edges"]))
        for index, (label, left, right) in enumerate(pairs):
            log(f"JS diagnostic {index + 1}/{len(pairs)}: {label}")
            result = js_divergence(backend.model(left), backend.model(right), args.js_samples,
                                   args.seed + 20001 + index, args.memory_mb, args.max_length)
            stability.append({"comparison": label, "n_a": len(left), "n_b": len(right),
                              "model_a": str(backend.model_path(left)), "model_b": str(backend.model_path(right)), **result})
            write_json(args.out / "stability.json", stability)
        write_tsv(args.out / "stability.tsv", stability,
                  ["comparison", "n_a", "n_b", "js_bits", "mc_se_bits", "mc_normal_95_low", "mc_normal_95_high",
                   "kl_a_to_b_bits", "kl_b_to_a_bits", "samples_per_model", "seed", "model_a", "model_b"])
        summary["js_status"] = "completed"
        write_json(args.out / "summary.json", summary)
    if args.competitors:
        rows = compare_competitors({"selected": args.out / "selected.cm", "full_pool": args.out / "full_pool.cm"},
                                   args.competitors, args.out / "cmcompare", args)
        baseline_links = {r["target"]: r["link_score"] for r in rows if r["query"] == "full_pool"}
        for row in rows:
            row["link_delta_vs_full_pool"] = row["link_score"] - baseline_links[row["target"]]
        write_tsv(args.out / "cmcompare.tsv", rows,
                  ["query", "target", "target_name", "score_query", "score_target", "link_score",
                   "link_delta_vs_full_pool", "target_sha256"])
        summary["cmcompare_status"] = "completed"
        write_json(args.out / "summary.json", summary)
    if args.calibrate:
        command = [args.cmcalibrate, "--cpu", str(args.cpu), str(args.out / "selected.cm")]
        run_command(command, args.out / "cmcalibrate.log", args.command_timeout)
        summary["calibrated"] = True
    summary.update({"builds_this_execution": backend.builds, "scored_sequences_this_execution": backend.score_calls,
                    "status": "complete"})
    write_json(args.out / "summary.json", summary)
    manifest_data = json.loads(manifest.read_text())
    manifest_data["status"] = "complete"
    write_json(manifest, manifest_data)
    log(f"Selected {len(selected_ids)}/{len(names)} sequences. Results: {args.out}")
    return summary


def compare_command(args) -> None:
    a, b = CM.read(args.a), CM.read(args.b)
    result = js_divergence(a, b, args.samples, args.seed, args.memory_mb, args.max_length)
    write_json(args.out, {"model_a": str(args.a.resolve()), "model_b": str(args.b.resolve()), **result})
    log(f"JS = {result['js_bits']:.6g} bits; Monte Carlo SE = {result['mc_se_bits']:.3g}")


def score_command(args) -> None:
    model, sequences = CM.read(args.cm), read_fasta(args.fasta)
    rows = []
    for identifier, sequence in sequences.items():
        if len(sequence) > args.max_length:
            raise ValueError("Sequence exceeds --max-length.")
        rows.append({"sequence_id": identifier, "length": len(sequence),
                     "log2_probability": model.log_probability(sequence, args.memory_mb)})
    write_tsv(args.out, rows, ["sequence_id", "length", "log2_probability"])


def selftest_command(args) -> None:
    """Independent exact-enumeration tests are embedded to keep delivery to two files."""
    import unittest
    from types import SimpleNamespace

    def state(kind, children=(), transition=(), emission=()):
        return State(kind, tuple(children), np.array(transition, dtype=float), np.array(emission, dtype=float))

    def categorical(p):
        return CM([state("S", [1], [1]), state("ML", [2], [1], p), state("E")])

    def enumerate_strings(model, v=0, max_length=5):
        """Independent dictionary expansion; used only for acyclic toy grammars."""
        s = model.states[v]
        if s.kind == "E":
            return {"": 1.0}
        result = defaultdict(float)
        if s.kind == "B":
            for x, px in enumerate_strings(model, s.children[0], max_length).items():
                for y, py in enumerate_strings(model, s.children[1], max_length).items():
                    if len(x + y) <= max_length:
                        result[x + y] += px * py
        else:
            for child, pt in zip(s.children, s.transition):
                if not pt:
                    continue
                if child == v:
                    raise ValueError("Enumeration test deliberately excludes cycles.")
                for x, px in enumerate_strings(model, child, max_length).items():
                    if not len(s.emission):
                        result[x] += pt * px
                    else:
                        for letter, pe in enumerate(s.emission):
                            if s.kind == "MP":
                                y = ALPHABET[letter // 4] + x + ALPHABET[letter % 4]
                            elif s.kind in {"ML", "IL"}:
                                y = ALPHABET[letter] + x
                            else:
                                y = x + ALPHABET[letter]
                            if len(y) <= max_length:
                                result[y] += pt * px * pe
        return dict(result)

    class Tests(unittest.TestCase):
        def test_inside_against_exhaustive_grammar_expansion(self):
            pair = np.zeros(16); pair[3] = 0.7; pair[9] = 0.3
            model = CM([state("S", [1], [1]), state("B", [2, 6]),
                        state("S", [3, 4], [0.4, 0.6]),
                        state("ML", [5], [1], [0.8, 0.2, 0, 0]),
                        state("D", [5], [1]), state("E"),
                        state("S", [7, 8, 9], [0.5, 0.3, 0.2]),
                        state("MP", [10], [1], pair),
                        state("MR", [10], [1], [0, 0.2, 0.8, 0]),
                        state("IR", [10], [1], [0.4, 0.6, 0, 0]), state("E")])
            expected = enumerate_strings(model)
            self.assertAlmostEqual(sum(expected.values()), 1)
            for length in range(4):
                for letters in itertools.product(ALPHABET, repeat=length):
                    seq = "".join(letters)
                    self.assertAlmostEqual(2 ** model.log_probability(seq), expected.get(seq, 0), places=12)
            rng = np.random.default_rng(3)
            for _ in range(100):
                self.assertGreater(expected[model.sample(rng)], 0)

        def test_insert_self_loops_and_multiple_parses(self):
            for kind in ("IL", "IR"):
                model = CM([state("S", [1, 2], [0.4, 0.6]),
                            state("ML", [3], [1], [1, 0, 0, 0]),
                            state(kind, [2, 3], [0.5, 0.5], [1, 0, 0, 0]), state("E")])
                self.assertAlmostEqual(2 ** model.log_probability("A"), 0.7)
                self.assertAlmostEqual(2 ** model.log_probability("AA"), 0.15)
                self.assertAlmostEqual(2 ** model.log_probability("AAAA"), 0.0375)
                self.assertEqual(model.log_probability(""), NEG_INF)

        def test_js_identity_and_disjoint_support(self):
            a, b = categorical([1, 0, 0, 0]), categorical([0, 1, 0, 0])
            same = js_divergence(a, a, 20, 1, 64, 20)
            disjoint = js_divergence(a, b, 20, 1, 64, 20)
            self.assertEqual(same["js_bits"], 0)
            self.assertEqual(disjoint["js_bits"], 1)
            self.assertEqual(disjoint["kl_a_to_b_bits"], float("inf"))

        def test_js_against_analytic_value(self):
            pa, pb = np.array([0.9, 0.1, 0, 0]), np.array([0.2, 0.8, 0, 0])
            mix = (pa[:2] + pb[:2]) / 2
            exact = 0.5 * np.sum(pa[:2] * np.log2(pa[:2] / mix) + pb[:2] * np.log2(pb[:2] / mix))
            result = js_divergence(categorical(pa), categorical(pb), 3000, 7, 64, 20)
            self.assertLess(abs(result["js_bits"] - exact), 0.04)

        def test_parser_nonuniform_null_and_rounding(self):
            q, p = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.3, 0.2, 0.1])
            text = "INFERNAL1/a [test]\nNAME toy\nSTATES 3\nALPH RNA\nNULL "
            text += " ".join(f"{math.log2(x / .25):.3f}" for x in q)
            text += "\nCM\nS 0 -1 0 1 1 0 0 0 0 0.000\nML 1 0 1 2 1 0 0 0 0 0.000 "
            text += " ".join(f"{math.log2(x / y):.3f}" for x, y in zip(p, q))
            text += "\nE 2 1 1 -1 0 0 0 0 0\n//\nHMMER3/f [filter ignored]\n//\n"
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "toy.cm"; path.write_text(text)
                model = CM.read(path)
                total = sum(2 ** model.log_probability(x) for x in ALPHABET)
                self.assertAlmostEqual(total, 1)
                self.assertAlmostEqual(2 ** model.log_probability("A"), 0.4, places=3)

        def test_stockholm_interleaving_and_groups(self):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "seed.sto"
                path.write_text("# STOCKHOLM 1.0\na AC\nb AG\n#=GC SS_cons <.\n#=GC RF xx\n\na GU\nb CU\n#=GC SS_cons .>\n#=GC RF xx\n//\n")
                aln = Alignment.read(path)
                self.assertEqual(aln.raw["a"], "ACGU")
                self.assertEqual(len(set(sequence_groups(aln, None, 1).values())), 2)
                aln.write_subset(Path(temporary) / "subset.sto", ["a"])

        def test_objective_keeps_evaluation_fixed_and_masks_training(self):
            with tempfile.TemporaryDirectory() as temporary:
                calls = []
                class FakeBackend:
                    output = Path(temporary)
                    def score(self, train, test):
                        self_test = set(train) & set(test)
                        if self_test:
                            raise AssertionError("Training leak")
                        calls.append((set(train), set(test)))
                        return {x: -float(len(train)) for x in test}
                opts = SimpleNamespace(min_size=2, max_evaluations=10, size_penalty=0)
                objective = Objective(FakeBackend(), ["a", "b", "c", "d"],
                                      {x: x for x in "abcd"}, [["a", "b"], ["c", "d"]], opts, "test")
                objective.evaluate(["a", "c"])
                self.assertEqual(set.union(*(test for _, test in calls)), set("abcd"))
                self.assertFalse(objective.feasible(["a", "b"]))

        def test_subset_search_known_optimum_and_baseline(self):
            class ToyObjective:
                pool = list("abcd")
                evaluations = 0
                memo = {}
                def feasible(self, subset):
                    return len(subset) >= 2
                def evaluate(self, subset):
                    key = tuple(sorted(subset))
                    if key not in self.memo:
                        self.evaluations += 1
                        score = -len(set(key) ^ {"a", "c"})
                        self.memo[key] = {"ids": list(key), "n": len(key), "objective": score}
                    return self.memo[key]
            opts = SimpleNamespace(search="exhaustive", min_size=2, tie_tolerance=1e-8)
            result = select_subset(ToyObjective(), opts, 1)
            self.assertEqual(result["selected"]["ids"], ["a", "c"])
            self.assertEqual(result["full_pool_baseline"]["n"], 4)

        def test_cmcompare_parsers(self):
            self.assertEqual(parse_comparison("q.cm t.cm 27.996 19.500 ACGU .... .... [1,2] [1,2]", "legacy"), (27.996, 19.5, 19.5))
            self.assertEqual(parse_comparison('{"score_query":3,"score_target":4}', "json"), (3, 4, 3))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise RuntimeError("Self-tests failed.")
    if args.integration:
        # Optional real executable integration; no mocks or synthetic tool outputs.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            alignment = path / "toy.sto"
            alignment.write_text("# STOCKHOLM 1.0\na ACGUACACGUAC\nb GCAUGCGCAUGC\nc AGCUACAGCUAC\nd GUACGUGUACGU\n#=GC SS_cons <<..>><<..>>\n#=GC RF xxxxxxxxxxxx\n//\n")
            command = [sys.executable, str(Path(__file__).resolve()), "optimize", "--alignment", str(alignment),
                       "--out", str(path / "run"), "--identity-threshold", "1", "--outer-folds", "2",
                       "--inner-folds", "2", "--max-evaluations", "8", "--js-samples", "4", "--bootstrap", "100",
                       "--cmbuild", args.cmbuild]
            subprocess.run(command, check=True)
            summary = json.loads((path / "run" / "summary.json").read_text())
            if summary["status"] != "complete" or summary["outer_status"] != "completed":
                raise RuntimeError("Infernal integration did not complete.")
        log("Real cmbuild integration passed.")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root.add_argument("--version", action="version", version=VERSION)
    commands = root.add_subparsers(dest="command", required=True)

    def numerical(p):
        p.add_argument("--seed", type=int, default=17, help="Reproducible RNG seed.")
        p.add_argument("--memory-mb", type=float, default=1024, help="Inside memory budget per sequence.")
        p.add_argument("--max-length", type=int, default=500, help="Input/sample length limit; excess causes an error, never censoring.")

    opt = commands.add_parser("optimize", help="Nested grouped CV and explicit seed-subset selection.")
    opt.add_argument("--alignment", type=Path, required=True)
    opt.add_argument("--out", type=Path, required=True)
    opt.add_argument("--groups", type=Path, help="TSV: sequence_id<TAB>group; overrides identity clustering.")
    opt.add_argument("--identity-threshold", type=float, default=0.95)
    opt.add_argument("--inner-folds", type=int, default=3)
    opt.add_argument("--outer-folds", type=int, default=3, help="0 disables independent outer evaluation; then inner scores are optimistic.")
    opt.add_argument("--search", choices=["local", "exhaustive"], default="local")
    opt.add_argument("--min-size", type=int, default=2)
    opt.add_argument("--max-evaluations", type=int, default=200, help="Candidate-mask budget PER outer selection and final selection.")
    opt.add_argument("--max-rounds", type=int, default=50)
    opt.add_argument("--restarts", type=int, default=1, help="Includes the full-pool start; later starts use random feasible subsets.")
    opt.add_argument("--swaps", type=int, default=25, help="Maximum random one-for-one swap proposals per neighborhood.")
    opt.add_argument("--size-penalty", type=float, default=0.0, help="Optional bits-per-sequence penalty on seed size.")
    opt.add_argument("--tie-tolerance", type=float, default=1e-6, help="Numerical score tolerance for preferring smaller masks; not a statistical margin.")
    opt.add_argument("--weighting", choices=["wpb", "wgsc", "wblosum", "wnone"], default="wpb")
    opt.add_argument("--cmbuild", default="cmbuild")
    opt.add_argument("--calibrate", action="store_true", help="Calibrate the final selected CM for subsequent database searches.")
    opt.add_argument("--cmcalibrate", default="cmcalibrate")
    opt.add_argument("--cpu", type=int, default=2, help="cmcalibrate workers; subset evaluation is sequential.")
    opt.add_argument("--command-timeout", type=float, default=3600)
    opt.add_argument("--js-samples", type=int, default=64, help="Samples per model for selected/full comparison; 0 disables diagnostics.")
    opt.add_argument("--js-path", action="store_true", help="Also compare every accepted add/delete/swap step; potentially expensive.")
    opt.add_argument("--competitors", type=Path, nargs="+", help="One ASCII CM per competing-family file.")
    opt.add_argument("--cmcompare-command", default='["hsCMCompare", "{query}", "{target}"]', help="JSON argv template; shell is never invoked.")
    opt.add_argument("--cmcompare-format", choices=["legacy", "json"], default="legacy")
    opt.add_argument("--bootstrap", type=int, default=2000)
    opt.add_argument("--resume", action="store_true", help="Replay identical search, reusing validated model and score caches.")
    numerical(opt)
    opt.set_defaults(func=optimizer)
    comparison = commands.add_parser("compare", help="Estimate sequence-distribution JS and both directional KL values.")
    comparison.add_argument("--a", type=Path, required=True)
    comparison.add_argument("--b", type=Path, required=True)
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--samples", type=int, default=200)
    numerical(comparison)
    comparison.set_defaults(func=compare_command)
    score = commands.add_parser("score", help="Calculate normalized global sequence log probabilities.")
    score.add_argument("--cm", type=Path, required=True)
    score.add_argument("--fasta", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    numerical(score)
    score.set_defaults(func=score_command)
    tests = commands.add_parser("selftest", help="Run embedded mathematical and workflow tests.")
    tests.add_argument("--integration", action="store_true", help="Also run a real cmbuild nested-CV integration test.")
    tests.add_argument("--cmbuild", default="cmbuild")
    tests.set_defaults(func=selftest_command)
    return root


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    try:
        for key, value in vars(args).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"--{key.replace('_', '-')} must be finite.")
        for name in ("memory_mb", "max_length", "command_timeout", "cpu", "max_rounds", "restarts", "max_evaluations", "bootstrap"):
            if hasattr(args, name) and getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive.")
        if hasattr(args, "samples") and args.samples < 2:
            raise ValueError("--samples must be at least 2.")
        if args.command == "optimize":
            if args.inner_folds < 2 or args.outer_folds == 1 or args.outer_folds < 0:
                raise ValueError("Inner folds >=2; outer folds either 0 or >=2.")
            if args.min_size < 2 or args.swaps < 0 or args.size_penalty < 0 or args.tie_tolerance < 0:
                raise ValueError("Minimum seed size is 2; swaps, penalty and tolerance must be nonnegative.")
            if not 0 < args.identity_threshold <= 1 or args.js_samples < 0 or args.js_samples == 1:
                raise ValueError("Identity threshold must be in (0,1]; JS samples must be 0 or >=2.")
        args.func(args)
    except (ValueError, RuntimeError, OSError, MemoryError, ArithmeticError, subprocess.SubprocessError) as error:
        if getattr(args, "_run_started", False):
            for filename in ("manifest.json", "summary.json"):
                path = args.out / filename
                if path.exists():
                    try:
                        data = json.loads(path.read_text())
                        data.update({"status": "failed", "error": str(error)})
                        write_json(path, data)
                    except (OSError, ValueError):
                        pass  # Preserve the original failure if the filesystem also fails.
        log(f"ERROR: {error}")
        sys.exit(2)
    except KeyboardInterrupt:
        log("Interrupted. Completed model and score caches remain available for --resume.")
        sys.exit(130)


if __name__ == "__main__":
    main()
