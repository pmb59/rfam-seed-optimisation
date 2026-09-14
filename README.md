# Selecting an RNA seed subset for Infernal

This document accompanies `cm_subset_optimizer.py`, a single-file Python research
implementation. Given a fixed, structurally annotated alignment of candidate
homologs, it selects an explicit subset of sequence IDs, builds its covariance
model (CM), and records the evidence used for selection.

The implementation includes:

- Grouped inner cross-validation of candidate **subsets**, including additions,
  deletions and swaps; the full candidate pool is always a baseline.
- An outer cross-validation loop that evaluates the selection procedure.
- A global Inside algorithm and matching random sampler operating directly on
  normalized probabilities reconstructed from Infernal ASCII CM files.
- Monte Carlo Jensen-Shannon (JS) divergence and directional KL diagnostics.
- Optional CMCompare calls against supplied competing-family models.
- Persistent model/score caches, reproducible folds, command logs and embedded
  self-tests. No network calls, package installation or Git operations occur.

This is a starting point for a research repository. A bounded heuristic search
does **not** establish the optimum among all possible subsets. Predictive fit to
homologs does **not** by itself establish specificity in genome searches.

## 1. Installation and first run

Requirements:

- Python 3.10 or later.
- NumPy: `python -m pip install numpy`.
- Infernal 1.1.x, with `cmbuild` on `PATH`, for subset optimization.
- `cmcalibrate` if final search calibration is requested.
- A working CMCompare installation or compatible command adapter if competing
  models are supplied. CMCompare is a separate program, not an Infernal command.

Obtain Infernal from the [official repository](https://github.com/EddyRivasLab/infernal).
CMCompare is described on the [authors' software page](https://www.tbi.univie.ac.at/software/cmcompare/).
This delivery has not verified a current CMCompare installation.

Run the mathematical and workflow tests first:

```bash
python cm_subset_optimizer.py selftest
```

After installing Infernal, run the executable integration test:

```bash
python cm_subset_optimizer.py selftest --integration
```

The integration test uses actual `cmbuild` processes on a small artificial
alignment and exercises the nested selection pipeline. It does not substitute
mock outputs for Infernal, and it does not test CMCompare or calibration.

A first exploratory run on your own data:

```bash
python cm_subset_optimizer.py optimize \
  --alignment candidates.sto \
  --out experiment_01 \
  --inner-folds 3 \
  --outer-folds 3 \
  --max-evaluations 200 \
  --js-samples 64 \
  --seed 17
```

The evaluation budget applies separately to each outer training pool and the
final all-candidate selection. This command is a research computation, not an
instantaneous operation. Start with a small budget to measure local runtime,
then increase it. The final subset is in `selected.ids.txt`; the final seed and
model are `selected.sto` and `selected.cm`.

For a more extensive search, for example on 50 candidate sequences:

```bash
python cm_subset_optimizer.py optimize \
  --alignment candidates.sto \
  --groups groups.tsv \
  --out experiment_02 \
  --max-evaluations 1000 \
  --restarts 3 \
  --swaps 50 \
  --js-samples 200 \
  --js-path \
  --calibrate \
  --cpu 4 \
  --seed 17
```

Use `--cmbuild /absolute/path/cmbuild` and
`--cmcalibrate /absolute/path/cmcalibrate` when necessary.

## 2. Input contract and scope

`candidates.sto` must contain exactly one Stockholm alignment with unique IDs,
aligned sequences, `#=GC SS_cons`, and `#=GC RF`. Interleaved Stockholm blocks
are supported. A/C/G/U and T are accepted; T becomes U. Gaps may be `.`, `-`,
or `_`. Ambiguous bases, missing-data symbols in sequences, and all-gap
sequences are rejected. Resolve those cases explicitly before running.

The structural annotation must be balanced and nested. Letter-coded pseudoknots
and crossing pairs are rejected. Both ends of an annotated pair must be RF
consensus columns. These restrictions make the modeled structure explicit.

The same column coordinates, RF annotation and SS_cons are retained in every
subset, including columns that become all-gap. The builder uses `--hand` and
one chosen weighting method (default `--wpb`). Entropy weighting remains at
Infernal's default. Existing alignment sequence weights and incidental
annotations are not copied; Infernal recomputes weights for each subset.
See the [cmbuild documentation](https://manpages.debian.org/trixie/infernal/cmbuild.1.en.html).

All evaluations are conditional on this fixed alignment and structure. If you
inferred or refined them using all candidate sequences, outer CV is **not** an
independent test of structure discovery or alignment construction. A stronger
study would construct/refine them inside each training fold, or use a trusted
external reference. That additional alignment-construction workflow is outside
this implementation.

Treat inputs as complete RNA sequences, with alignment gaps representing
insertions/deletions relative to the consensus. Incomplete experimental
fragments require a separate observation/truncation model. Global scoring here
does not implement Infernal's truncated-search algorithms.

### Grouping related sequences

By default, the program builds single-linkage components at aligned identity
>= 0.95. Identity is the number of matching non-gap residues divided by the
number of alignment columns containing a residue in at least one sequence.
Columns with two gaps are excluded. Every directly connected pair at or above
the threshold remains in one group; transitive links can produce large groups.

Use `--identity-threshold` to specify a scientifically appropriate threshold.
There is no universal cutoff. If the data collapse into too few groups, the
program fails rather than silently treating related sequences as independent.

For taxonomic or curated groups, supply a tab-delimited file:

```text
sequence_id	group
seq_A	clade_1
seq_B	clade_1
seq_C	clade_2
seq_D	clade_3
```

The separators in the actual file must be tab characters. Every candidate must
appear exactly once. Supplied groups override automatic clustering, but
identical ungapped sequences may not be assigned to different groups.
Otherwise the supplied grouping is trusted; the program does not infer a
phylogeny or verify taxonomic annotations.

Groups are assigned intact to folds, approximately balancing sequence counts.
The score averages equally across groups, so a large group does not dominate.
The number of folds is capped by the number of available groups. Each outer
training pool must retain at least two groups for its inner CV.

## 3. What is optimized?

Let the candidate pool be

\[
D=\{x_1,\ldots,x_n\}.
\]

A candidate is a **membership subset** \(S\subseteq D\), not just a number.
Write \(C(S)\) for the CM built from its seed rows. The desired output is

\[
S^*=\arg\max_S Q(S), \qquad N^*=|S^*|.
\]

Two subsets of the same size are evaluated separately. For 50 candidates there
are \(2^{50}-1\) nonempty subsets, so routine exhaustive enumeration is
impractical.

### A fixed evaluation set for every membership mask

Partition the current training pool into fixed grouped folds \(V_k\). For a
candidate mask \(S\), fold \(k\) uses

\[
T_{S,k}=S\setminus V_k
\]

to build its CM, and scores **every** sequence in \(V_k\), including sequences
not selected by \(S\). Thus a mask cannot improve merely by removing difficult
sequences from the evaluation set.

Let \(g\) index the groups in this pool, \(D_g\) be its sequences, and \(k(g)\)
the fold containing that group. The implemented inner score is

\[
\widehat Q(S)=
\frac{1}{G}\sum_{g=1}^{G}
\frac{1}{|D_g|}\sum_{x\in D_g}
\log_2 P_{C(S\setminus V_{k(g)})}(x).
\]

This is group-balanced held-out log probability, in bits per sequence. It is
not divided by RNA length. All candidate masks are evaluated on the same
sequences, so these log scores can be compared directly. If a per-residue or
different biological weighting is desired, change `group_mean` deliberately
and document the resulting objective.

An optional, prespecified size penalty gives

\[
J(S)=\widehat Q(S)-\lambda |S|.
\]

`--size-penalty` sets \(\lambda\); the default is zero. It represents a
preference for seed compactness, not an AIC/BIC parameter-count correction.
`--tie-tolerance` only handles numerical ties, preferring smaller subsets. It
is not a confidence interval, equivalence test or biological tolerance.

**Consequence of this masking design:** the fitted seed in a CV fold has size
\(|S\setminus V_k|\), usually smaller than the final \(|S|\). The code records
these fold training sizes. It evaluates a selected mask under withholding;
it does not estimate the performance of a fixed-N training design in every
fold. Final fitting uses the complete selected subset.

Each fold must leave at least one selected training sequence. Consequently,
selected masks must span at least two folds, and this implementation requires
`--min-size >= 2`. Selecting a single-sequence final seed would require a
different design with external fixed validation sequences; it is not silently
approximated here. The default feasible search space should be reported in a
publication.

### Subset search

The default search starts from the full candidate pool and evaluates feasible
one-sequence deletions, additions, and randomly sampled one-for-one swaps. It
moves to the best improving neighbor, then repeats. Optional additional starts
use random feasible subsets. These are alternative starting masks, not
bootstrap replicates or confidence-interval calculations.

Controls are `--max-evaluations`, `--max-rounds`, `--restarts`, and `--swaps`.
Proposals are reproducibly shuffled to reduce identifier-order bias when the
budget ends partway through a neighborhood. The result is the best evaluated
mask, which can remain the full pool.

`--search exhaustive` enumerates all feasible masks for pools of at most 20
sequences, still subject to the evaluation budget. Only a completed exhaustive
search marks `globally_optimal_over_feasible_masks: true`. A completed local
search is never a certificate of global optimality.

## 4. Outer CV separates selection from evaluation

Membership is tuned using inner CV scores, so the winning inner score is
optimistic. The outer procedure is:

1. Hold out an entire outer fold.
2. Construct inner folds using only the outer training pool.
3. Run the complete subset search inside that pool.
4. Refit its winning subset and score the untouched outer fold.
5. Also fit all outer training sequences as the baseline and score that same
   outer fold.

The outer predictions are not fed back to the optimizer. After the outer loop,
the same selection procedure is run on all candidates to obtain the final
sequence-ID list and CM. Outer folds assess the **procedure**; they do not
independently assess that final CM trained using all the available groups.

For each outer test sequence, the output includes

\[
\Delta_x=\log_2 P_{C_{\mathrm{selected}}}(x)
-\log_2 P_{C_{\mathrm{full\ training\ pool}}}(x).
\]

The summary averages these differences within groups, then across groups.
Positive values favor subset selection under the predictive objective.
Group bootstrap resampling provides a descriptive interval for the paired
differences. It resamples existing outer predictions, **not** the full fitting
and selection procedure; overlapping training sets and few independent groups
limit the interpretation. It is not a proof that one final CM is superior.

`--outer-folds 0` is supported for exploratory runs and is explicitly labelled
as lacking outer evaluation. Changing grouping, search settings, weighting,
or penalties after examining outer scores effectively tunes on those scores.
Reserve an additional independent test set for repeated methodological tuning.

The statistical reason to use predictive log probability is

\[
\mathbb E_{P_*}[\log_2P_C(X)]
=-H(P_*)-D_{\mathrm{KL}}(P_*\|P_C),
\]

where \(P_*\) is the target family distribution. Maximizing the expectation
minimizes divergence from that distribution. Here the group weighting defines
which empirical distribution the CV estimate is intended to represent.

## 5. Probability calculations are explicit

### Reading a CM

The parser supports one `INFERNAL1/a` ASCII CM per file, with the standard
S/D/B/E/ML/MR/MP/IL/IR states. It stops at the CM terminator and ignores the
following filter HMM. Binary files, multiple-CM libraries, and unsupported
formats fail with an error.

For a nucleotide background \(q\), file emission scores are converted as

\[
e_v(a)=q(a)2^{s_v(a)},\qquad
e_v(a,b)=q(a)q(b)2^{s_v(a,b)}.
\]

Transition scores are converted using \(t_{vu}=2^{s_{vu}}\). The NULL line
stores background log odds relative to uniform nucleotide probabilities.
Each reconstructed categorical distribution is renormalized to account for
ASCII rounding; grossly inconsistent sums are rejected. An asterisk denotes
zero probability. These encodings follow Infernal's
[CM writer implementation](https://github.com/EddyRivasLab/infernal/blob/master/src/cm_file.c).

The scoring distribution is therefore the normalized **global grammar
reconstructed from the saved model**, with its natural distribution over
sequence lengths. It is not claimed to reproduce search E-values or local
score corrections. No length conditioning, null2/null3 correction, HMM filter,
truncation handling or search threshold is applied.

### Inside recurrence

Let \(F_v(i,d)\) be the probability that state \(v\) generates the substring
\(x_i\ldots x_{i+d-1}\). For an ordinary state, let \(\ell_v,r_v\in\{0,1\}\)
indicate whether it emits a left/right residue. Its emission factor is
\(E_v(x,i,d)\), or 1 for a silent state. Then

\[
F_v(i,d)=E_v(x,i,d)
\sum_u t_{vu}
F_u(i+\ell_v,d-\ell_v-r_v).
\]

For a bifurcation with left/right children \(b,c\),

\[
F_v(i,d)=\sum_{k=0}^{d}F_b(i,k)F_c(i+k,d-k).
\]

For an end state, \(F_E(i,0)=1\) and \(F_E(i,d>0)=0\). Finally,

\[
P_C(x)=F_{\mathrm{root}}(0,|x|).
\]

The implementation works in base-2 log space with stable log-sum-exp, summing
over parses rather than choosing the highest-scoring parse. Forward state
ordering and residue-consuming insert self-loops determine evaluation order.
Non-forward transitions, silent self-loops, and nonterminating reachable
insert states are rejected.

The sampler uses exactly the same normalized transitions and emissions. A
bifurcation emits both subtrees in sequence order. This matching definition is
essential for valid divergence estimates. The sampler is internal, so
`cmemit` is not a dependency of this implementation.

### Cost and numerical limits

Memory is approximately \(O(M L^2)\), where \(M\) is state count and \(L\)
is sequence length. Non-bifurcation evaluation is \(O(M L^2)\) for bounded
out-degree; bifurcations add \(O(B L^3)\) work. The reference implementation
uses NumPy and runs sequentially. `--cpu` only controls optional calibration.
Long RNAs and large search budgets can take hours and substantial memory.

`--memory-mb` bounds estimated per-sequence working memory. `--max-length`
limits inputs and generated samples. If a sample exceeds the limit, the run
fails: it never redraws, rejects or truncates that sample. Those alternatives
would change the distribution being estimated. Increasing the limit requires
a new optimizer output directory because it changes the recorded configuration.

## 6. JS and KL diagnostics

For sequence distributions \(P_A,P_B\), define their equal mixture

\[
M(x)=\tfrac12(P_A(x)+P_B(x)).
\]

Jensen-Shannon divergence is

\[
\operatorname{JS}(A,B)=
\tfrac12\mathbb E_{P_A}\!\left[\log_2\frac{P_A(X)}{M(X)}\right]
+\tfrac12\mathbb E_{P_B}\!\left[\log_2\frac{P_B(X)}{M(X)}\right].
\]

Draw \(m\) independent samples from each model, evaluate both model
probabilities for every sample, and replace the expectations by sample means.
No common topology or state correspondence is required: the comparison is
over emitted sequence distributions, not state indices or parse trees.

For the two sets of log-ratio contributions \(Y_A,Y_B\), the reported Monte
Carlo standard error is

\[
\widehat{\mathrm{SE}}(\widehat{\operatorname{JS}})
=\tfrac12\sqrt{s_A^2/m+s_B^2/m}.
\]

The theoretical JS range is [0,1] bits. Finite-sample estimates or normal
intervals can fall below zero; these are preserved rather than silently
clipped. The normal interval is an approximate Monte Carlo interval
conditional on the fitted models, not uncertainty about biological truth or
seed selection. Increase `--js-samples` for a more precise diagnostic.

Directional KL estimates use the same samples:

\[
\widehat D_{\mathrm{KL}}(A\|B)
=\frac1m\sum_{x\sim A}\log_2\frac{P_A(x)}{P_B(x)}.
\]

If a sampled A sequence has zero probability under B, the corresponding KL is
infinite. Missing rare support mismatches can still make a finite Monte Carlo
estimate misleading; JS remains finite. IEEE infinities are stored as explicit
strings in JSON, not nonstandard JSON numeric values.

By default the selected full-data CM is compared with the full-pool CM.
`--js-path` additionally compares models before and after every accepted
final-search move. These moves can grow, shrink or swap the seed; this output
is a subset-change trajectory, not necessarily a monotonic learning curve in
N. JS is a post-selection diagnostic and does not influence search or stopping.
Small JS means stability under a particular change, not demonstrated accuracy.

Standalone comparison and scoring are also available without `cmbuild`:

```bash
python cm_subset_optimizer.py compare \
  --a A.cm --b B.cm --samples 500 --seed 17 --out comparison.json

python cm_subset_optimizer.py score \
  --cm A.cm --fasta withheld_homologs.fa --out probabilities.tsv
```

## 7. CMCompare integration and interpretation

Supply individual competing-family CM files:

```bash
python cm_subset_optimizer.py optimize \
  --alignment candidates.sto --groups groups.tsv --out experiment_03 \
  --competitors family_B.cm family_C.cm \
  --cmcompare-command '["hsCMCompare", "{query}", "{target}"]'
```

This runs comparisons for both the selected model and the full-pool model.
The command is a JSON argument array, executed without a shell. Use absolute
paths for executables/scripts when necessary. Each pair runs in its own
directory with inputs named `query.cm` and `target.cm`, preserving raw stdout,
stderr and command arguments for inspection.

Default `--cmcompare-format legacy` expects the documented non-verbose
`hsCMCompare` layout, with exactly nine whitespace-delimited fields:

```text
query target score_query score_target sequence structure_query structure_target nodes_query nodes_target
```

Verbose output, precomputed weak-pair tables and other layouts are not valid
inputs to that parser. If your version differs, use a command adapter with
`--cmcompare-format json`. The adapter must emit exactly one JSON object on
stdout, sending other logging to stderr:

```json
{"score_query": 27.996, "score_target": 19.5, "link_score": 19.5}
```

The `link_score` key is optional; if supplied it must equal the smaller score.
CMCompare's link score for its link sequence is

\[
L(A,B)=\min\{s_A(x_{\mathrm{link}}),s_B(x_{\mathrm{link}})\}.
\]

It identifies possible shared high-scoring sequences. The output includes the
change in link score relative to the full-pool model against each competitor.
No automatic universal specificity cutoff or link-score penalty is imposed.
Shared ancestry or legitimate structural similarity can explain high scores,
and a link score is not a false-positive probability. A weak model can also
have low overlap simply because it recognizes very little.

Missing executables, nonzero exit statuses, ambiguous output and invalid
scores fail explicitly. The program does not claim to have run CMCompare when
competitors were not supplied. It is an optional post-selection diagnostic;
if you use its results to revise the selection method, obtain new independent
evaluation data for the revised procedure.

Method references: [Höner zu Siederdissen and Hofacker, 2010](https://academic.oup.com/bioinformatics/article/26/18/i453/205243)
and [Eggenhofer et al., 2013](https://academic.oup.com/nar/article/41/W1/W499/1094235).

## 8. Output files and reproducibility

| Output | Purpose |
|---|---|
| `selected.ids.txt` | Explicit chosen candidate sequence IDs |
| `selected.sto` | Selected seed with fixed RF and SS_cons |
| `selected.cm` | CM fitted to the complete selected subset |
| `full_pool.cm` | Full-data baseline model |
| `selection.json` | Winning inner score, evaluated baseline, search status and accepted moves |
| `summary.json` | Selected IDs, outer results, diagnostic/calibration status |
| `groups.tsv` | Actual sequence groups used |
| `final_folds.json`, `outer_*_folds.json` | Fold membership for audit/reproduction |
| `outer_predictions.tsv` | Paired held-out scores for selected and full-training models |
| `outer_*_selection.json` | Selected IDs and search result within each outer training pool |
| `*_evaluations.jsonl` | Every newly evaluated mask, score and fold-specific training sizes |
| `stability.json`, `stability.tsv` | JS/KL estimates when requested |
| `cmcompare.tsv`, `cmcompare/` | Competitor results and raw command outputs when requested |
| `manifest.json` | Input/configuration fingerprint and executable/version information |
| `cache/` | Models, seed files, build logs and per-sequence likelihood caches |

Models do not need E-value calibration for normalized probability scoring.
`--calibrate` calibrates only the final selected CM for later database searches.
Check `summary.json` before assuming calibration or any optional diagnostic
completed. A run can finish selection and subsequently fail a diagnostic;
its selected files remain available, and status fields distinguish that case.

To resume, rerun the **same** optimize command with `--resume`. Search is
replayed deterministically while cached model builds and likelihoods are
reused; this is not an instruction-pointer checkpoint. Inner history files
are rewritten during replay. Changes in input, groups, settings, builder help
text or program version cause resume to fail. Use a new directory for changed
experiments. Do not run two processes against the same output directory, and
do not edit cache files manually.

The inference engine is deterministic for fixed files. RNG seeds control fold
ties, search proposals and Monte Carlo draws. Tool-version changes and small
floating-point differences can alter near-tied subset choices. Retain the
two source files, original alignment, group file, configuration and output
manifest when moving work to GitHub or a compute cluster. The program source
SHA-256, NumPy version and competing-model file hashes are also incorporated
in the run fingerprint. Editing the Python source invalidates resume/cache
identity, even when its VERSION string remains unchanged.

## 9. Validation performed and remaining checks

The embedded self-tests verify:

- Inside probabilities against independent exhaustive grammar expansion,
  including paired emissions, silent/deletion states, bifurcations and both
  sequence directions.
- Insert self-loops and multiple parses producing the same sequence against
  hand-calculated probabilities.
- JS=0 for identical distributions, JS=1 for disjoint supports, and a sampled
  JS estimate against an analytically solvable categorical example.
- CM ASCII probability conversion with a nonuniform NULL background and
  rounding, Stockholm interleaving, fixed evaluation sets, training exclusion,
  exhaustive subset selection on a known objective, and CMCompare parsing.

During preparation, the supplied 286-state, 94-consensus-position CM was also
parsed and scored; generated sequences received finite probabilities, and
self-comparison produced JS=0. That check uses the example as a file-format
fixture, not as an endorsement of its seed composition or biology.

Command-line orchestration, report generation, the JSON competitor adapter,
and cache reuse on resume were additionally exercised with explicitly labelled
test-double executables. Those tests verify software wiring, not Infernal's
model construction or CMCompare's scientific results.

Infernal and CMCompare executables were unavailable in the preparation
environment. Therefore actual `cmbuild`/`cmcalibrate`/CMCompare end-to-end
compatibility has **not** been claimed as tested. Run `selftest --integration`
with your Infernal installation and inspect a small real competitor comparison
before a large study. For publication-grade use, add an independent
implementation comparison of global Inside probabilities and test additional
real CM architectures; keep the reference global-distribution convention
consistent when doing so.

The program is intentionally conditional on curated inputs and a defined
predictive objective. It does not infer homology labels, prove structure,
estimate empirical genome-search false-hit rates, provide Bayesian model
evidence, or guarantee a global optimum from a local search. Those claims
require additional data or methodology. The immediate scientific output is
an auditable subset, its fitted CM, grouped predictive evidence, and model
comparison diagnostics.
