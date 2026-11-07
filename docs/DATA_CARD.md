# Numerical data card

## Sources and population

The snapshots originate from OpenReview records for ICLR 2024, 2025, 2026 and NeurIPS 2025. We credit OpenReview, the conference organisers, and the reviewers and area chairs contributing these records. Every released paper and review retains its source forum/review identifier; corresponding URLs are defined in `data/manifest.json`. Stable reviewer identities are not available and are not reconstructed.

The primary filter retains papers with at least two valid timestamped ratings and at least one hour between earliest and latest review creation. ICLR 2024/2025 records not explicitly marked acceptance, including empty decision strings, are non-acceptance; these are not all adjudicated rejections. ICLR 2026 is the explicit decided snapshot. The NeurIPS sample is selected and approximately 95.4% accepted; it is not representative of all submissions.

## Files and fields

Each compressed JSON file is a list of paper records:

- `id`, `forum`: source record IDs, used for provenance and joins.
- `decision`: recorded decision string, retaining empty strings.
- `reviews`: list of structured observations; `id`, `cdate` (creation time in milliseconds), `rating`, `confidence`, and `soundness` retain their source meanings.
- `engagement_features`: total words, weakness words, weakness/strength word ratio, numerical-token count divided by total words, and question-mark count. The text concatenation order is summary, strengths, weaknesses, questions. Word counts use Python whitespace splitting. Numeric matches use `\d+(\.\d+)?`; a zero strength count gives ratio 0.
- `neurips_features`: word count, numeric density, question count using summary, weaknesses, and questions. This follows the historical cross-venue script, whose field set differs from the primary engagement analysis.
- `ai_features`: total words, marker hits, marker rate per 1,000 words. Matching uses case-insensitive whole words from `data/ai_marker_words.json`; analyses retain texts of at least 20 words. These are weak lexical indicators, not AI-authorship labels.

No prose can be reconstructed from these counts. Data generation code is in `scripts/prepare_numeric.py`. Exact input hashes, collection-level counts before filtering, and source URL templates are in the manifest.

`metareview_labels.json` contains source paper ID, year, and the stored 1-to-5 LLM enthusiasm label. Coding requests that the model disregard verdict words; it does not remove them. Labels are derived model outputs and not independent expert annotations. The two calibrated cutoff policies select the same cutoff and are not independent replications.

## Uses and restrictions

The numerical package supports reproducible statistical evaluation. Do not infer reviewer identities, treat timing as a validated measure of personal effort, or infer that historical decision agreement establishes better scientific decisions. Respect record-level access controls when obtaining source text; a licence does not override access restrictions. Retain source attribution with derivative distributions. No article PDFs or full review/metareview prose are redistributed here.

## Intended scope of the release

Core statistical outputs are recomputable from the included numerical inputs. Text-feature extraction from prose can be repeated with separately obtained source text using `prepare_numeric.py`. Paid LLM reruns require source text and a user-owned API account. Six historical pairwise comparisons are available at aggregate summary granularity, not as a complete per-pair response archive. Original execution hardware and total historical API costs are not retrospectively reconstructed.
