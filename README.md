# The Deadline Effect

Code and numerical research artifacts for **The Deadline Effect: Identifying and Correcting Temporal Bias in Human Evaluation**, Shiyan Liu, NeurIPS 2026.

[Paper and discussion](https://openreview.net/forum?id=At4p3kXMIV) · [Data and provenance](docs/DATA_CARD.md) · [Reproduction scope](docs/REPRODUCIBILITY.md)

## Reproduce the statistical results

Use Python 3.11 or later. The reference environment uses Python 3.14, NumPy 2.4.1, SciPy 1.17.0, and Matplotlib 3.10.8.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
./reproduce.sh
```

The numerical inputs are included, so this replay needs **no API key, paid model calls, or access to the author's local files**. It recomputes the timing specifications, clustered uncertainty and paper bootstrap, aggregation rules, full-sample losses, cross-year transfer, metareview thresholds, text-feature effects, confidence controls, official-deadline checks, selected NeurIPS comparison, and lexical markers. The verifier compares the core outputs to the reference values at numerical tolerance and checks input hashes. Figures are written to `figures/`.

Outputs are in `results/recomputed/`; `results/reference/` contains the manuscript reference outputs. Input snapshots and the shared threshold-tie convention are fixed. A new API download can contain later review edits and need not give identical numbers.

## Data

`data/*_numeric.json.gz` contains structured ratings, timestamps, confidence, soundness, source forum/review IDs, and precomputed numerical text features. There are no author names, author emails, reviewer identities, full review prose, paper PDFs, credentials, or private discussion exports. Source IDs support attribution and record lookup. The primary retained ICLR sample contains **32,955 papers and 130,023 reviews**; the selected NeurIPS sample contains 5,541 papers and 22,373 reviews.

`data/metareview_labels.json` provides the 2,378 stored enthusiasm scores needed for the target-sensitivity replay. `data/content_coefficients.json` records the exploratory content-weight fits used in the manuscript. The content fitting/clipping mismatch is documented rather than treated as a matched OTCA comparison.

## Optional representation experiment

```sh
python -m pip install -r requirements-atd.txt
ATD_DEVICE=cpu python scripts/train_atd.py --epochs 50
```

This is the archived review-level ATD protocol, with seed 42. It permits paper overlap, uses full-snapshot preprocessing, and is not an independent-adversary invariance test. The historical timing AUC used descending-rank orientation; interpret its symmetric value. A fresh run is saved separately from the reported reference summary. Backend and precision can change its numerical result.

## Optional LLM assessments

```sh
python -m pip install -r requirements-llm.txt
export OPENROUTER_API_KEY='your-own-key'
python scripts/rerun_pairwise_judge.py --help
python scripts/rerun_metareview_judge.py --help
```

These optional scripts require review/metareview text snapshots in the documented source schema and explicitly require `--execute-paid-api`. **Running them can incur provider charges.** Keys are read only from the environment and are never included in this repository. The pairwise script evaluates both review orderings with a configurable model. Model routing IDs must be checked with the provider at execution time.

The six published judge-by-year results are included as **aggregate summaries** in `results/reference/judge_summaries.json`. They are not synthetic per-pair records. The repository does not claim that a new API rerun will recover every historical response. The returned judgments, prompt settings, and complete source snapshot should be archived when rerunning.

## Licenses and attribution

Original code is provided under the MIT license in `LICENSE`. This does not relicense third-party records, hosted models, or paper PDFs. OpenReview's [Terms of Use](https://openreview.net/legal/terms) specify CC BY 4.0 for review/metareview comments and CC0 for metadata. See [DATA_LICENSE.md](DATA_LICENSE.md) for the scope of the released numerical inputs and outputs. Model service terms and dependency credits are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

```bibtex
@inproceedings{liu2026deadline,
  title={The Deadline Effect: Identifying and Correcting Temporal Bias in Human Evaluation},
  author={Liu, Shiyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=At4p3kXMIV}
}
```
