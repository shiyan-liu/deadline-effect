# Reproduction scope and numerical conventions

The main command `./reproduce.sh` performs a computational replay from fixed, released numerical snapshots. It is distinct from making new language-model judgments or claiming a new independent scientific replication.

## Statistical replay

The main rating outcome is the leave-one-out contrast, equal to n/(n-1) times the mean-centred rating. Hour-cutoff, soundness and score-group specifications use mean-centred outcomes. Pooled review quantiles define relative Early/Mid/Late buckets; timestamp ties need not yield equally sized groups. The latest observed review is not assumed to be the official deadline.

The primary regression inference clusters by paper (CR1, asymptotic normal p-values). The NeurIPS historical relative-timing comparison additionally records its original independent-error p-value and a clustered alternative, so the inference convention is visible. Official-deadline comparisons use clustered inference. Paper bootstrap has 500 resamples and retains all reviews of each sampled paper.

OTCA fits raw-score squared loss against binary decisions within a feasible family. Evaluation additionally uses normalised score-derived Brier values and thresholded accuracy. These objectives are not equivalent. Borderline papers are fixed within one raw point of the year's median individual rating. Scores within 1e-12 raw points of the threshold are acceptance ties. Joint is timing weight multiplied by confidence, not OTCA combined with ATD.

Content coefficients are supplied for exact replay. `scripts/refit_content.py` separately recomputes the historical exploratory search. That search drops papers with nonpositive weights during fitting and floors weights at evaluation, and it is not a constrained-OTCA ablation.

## External model experiments

The six pairwise summaries in the paper are archived as the author's reported results from the OpenReview discussion. Optional pairwise code contains the rubric and both-ordering consistency protocol; it requires source text and paid API access. A new run does not overwrite the historical summary file. The archived implementation's default pilot model is not a claim that historical large-run results used that default; select the intended GPT-4.1 or DeepSeek model explicitly.

The metareview labels are included, enabling direct recalculation of the reported enthusiasm thresholds without new API calls. The coding script also exposes its prompt, temperature 0, truncation and response limit for new evaluations. Treating an instruction to ignore verdict words as equivalent to masking them would overstate target independence.

ATD code preserves the original review-level split, training order, architecture and legacy AUC convention. Its reported results are exploratory, with no independent post-training adversary and no repeated-seed interval. A new CPU replay is stored separately from historical values; the historical result remains separately identified. The 50-epoch CPU replay obtains symmetric timing AUC 0.7561 and acceptance AUC 0.7195, compared with historical values 0.625 and 0.720. This backend sensitivity strengthens the need for grouped splits and repeated-seed validation.

## Checks

`verify_results.py` checks released input hashes and compares the core numerical outputs against the manuscript reference JSON. It does not verify unarchived per-pair LLM responses or convert historical observations into causal evidence. Timing, backend, versions and measured resource use for the release replay are recorded under `results/`.

The archived lexical script mislabeled the two presence proportions returned by its late-first helper call. The released validation uses an explicit Late minus Early subtraction for both density and presence. All three years use the same decided/filtered snapshots as the primary analysis.
