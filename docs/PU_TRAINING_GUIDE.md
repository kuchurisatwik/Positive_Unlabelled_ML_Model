# PU Training Guide For CSE Phishing Detection

This guide explains how to build the dataset, extract features, and train the Positive-Unlabeled model step by step.

The important idea:

- We are not training "phishing vs normal internet".
- We are training "confirmed phishing vs CSE-looking suspicious/unlabeled URLs".
- Random legitimate websites are not used as negatives.
- The old `PS-02...xlsx` files are not needed for this PU flow.

## What PU Means Here

PU means Positive-Unlabeled.

In your case:

- Positive means a URL is confirmed phishing.
- Unlabeled means the URL is suspicious or CSE-like, but we do not know if it is truly phishing.
- Reliable negative means a row from the unlabeled set that the first model thinks is very unlikely to be phishing.

So the model learns in two stages:

1. Train a temporary model using confirmed phishing as `1` and all unlabeled rows as temporary `0`.
2. Pick only the safest low-score unlabeled rows as reliable negatives.
3. Train the final model using confirmed phishing vs reliable negatives.

The final output is a risk score, not a perfect truth label.

## Files In This Flow

Use these files:

```text
collect_cse_pu_urls.py
run_pipeline.py
train_pu_model.py
PU_TRAINING_GUIDE.md
input/cse/cse_list.csv
input/dataset/*.csv
input/dataset/*.xlsx
input/dataset/*.txt
```

Generated files:

```text
input/pu_candidates/cse_pu_candidates.csv
output/enriched_features_v2.csv
output/enriched_features_v2_audit.csv
output/pu_model/pu_model.joblib
output/pu_model/pu_scored_rows.csv
output/pu_model/pu_training_summary.json
```

Use `output/enriched_features_v2_audit.csv` for PU training because it keeps review columns like `url`, `domain`, `target_brand_domain`, and `source_label`.

## Step 0. Install Requirements

Local PowerShell:

```powershell
cd C:\Users\sathwik.kusuri\Documents\ai-challenge-ijmm\scripts\model_training_datasets
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Step 1. Prepare Raw Or Collected Candidate URLs

`run_pipeline.py` now performs CSE matching directly from `input/cse/cse_list.csv`, which must contain:

```text
domains,cse
```

That means raw feed files can go into `run_pipeline.py` without first being CSE-matched by `collect_cse_pu_urls.py`. The pipeline accepts CSV/XLSX rows and one-URL-per-line TXT feed files. Use the collector only when you want it to fetch OpenPhish/PhishTank or generate dnstwist candidates first.

Run:

```powershell
.\venv\Scripts\python.exe collect_cse_pu_urls.py --fetch-feeds
```

This script does four things when you use it:

1. Loads official CSE domains from `input/cse/cse_list.csv`.
2. Loads every CSV/XLSX candidate file from `input/dataset`, skipping audit/diagnostic files.
3. Optionally downloads phishing feed URLs and keeps only feed URLs that match your CSE names/domains.
4. Optionally runs dnstwist for CSE domains and keeps DNS-active lookalikes that lexically match the CSE list.

Expected output:

```text
input/pu_candidates/cse_pu_candidates.csv
```

That CSV can be the input for `run_pipeline.py`, but it is not required if your feed/local URL files are already in the pipeline input folder.

The output rows will have labels like:

```text
Unlabeled
Phishing
```

Important:

- `Unlabeled` does not mean safe.
- `Unlabeled` means "unknown, suspicious, needs model/review".
- `Phishing` comes from phishing feeds or confirmed extractor evidence.

## Step 1A. If Live Feed Download Fails

If your network blocks feed download, download feed files manually and put them here:

```text
input/data/openphish_feed.txt
input/data/phishtank_online_valid.csv
```

Then run:

```powershell
.\venv\Scripts\python.exe collect_cse_pu_urls.py `
  --openphish-file input\data\openphish_feed.txt `
  --phishtank-file input\data\phishtank_online_valid.csv
```

## Step 1B. Add DNS-Active dnstwist URLs

Install requirements first so the `dnstwist` command is available:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then run:

```powershell
.\venv\Scripts\python.exe collect_cse_pu_urls.py `
  --dnstwist `
  --require-dns-active `
  --dnstwist-workers 2 `
  --cpu-gate-threshold 85 `
  --dns-timeout 1.0 `
  --dns-lifetime 1.5 `
  --max-active-urls 5000
```

For a smaller first run, fuzz only selected CSE domains:

```powershell
.\venv\Scripts\python.exe collect_cse_pu_urls.py `
  --dnstwist `
  --dnstwist-domain examplebank.in,examplepay.in `
  --require-dns-active `
  --dnstwist-workers 2 `
  --cpu-gate-threshold 85 `
  --dns-timeout 1.0 `
  --dns-lifetime 1.5 `
  --max-active-urls 5000
```

dnstwist rows are labelled `Unlabeled` by default. That is intentional: DNS-active lookalikes are suspicious training candidates, but they are not confirmed phishing unless you verify them separately.

The collector runs dnstwist, feed matching, local matching, and DNS checks with bounded parallel workers and progress bars. DNS active checks use a fast active-only resolver with caching. The CPU gate is strict: new work waits while CPU is above `--cpu-gate-threshold`, and the run fails after `--cpu-gate-max-wait-seconds` if CPU does not cool down. `--max-active-urls 5000` writes at most 5000 final rows after DNS-active filtering. Use `--cpu-gate-threshold 100` only to disable the gate.

## Step 1C. If You Get Too Few Positives

Check the collector output. You want more than a handful of `Phishing` rows.

If it says too few phishing rows, try:

```powershell
.\venv\Scripts\python.exe collect_cse_pu_urls.py --fetch-feeds --feed-threshold 0.78
```

Default `--feed-threshold` is stricter. Lowering it catches more CSE-related feed URLs.

Do not go very low. If you use something like `0.50`, unrelated phishing URLs may enter your CSE dataset.

Suggested values:

```text
0.82 = strict
0.78 = reasonable fallback
0.75 = only if positives are still too low
```

## Step 2. Run Feature Extraction

`run_pipeline.py` now CSE-matches the input rows itself. Point `PIPELINE_INPUT_DIR` at either a raw feed/local folder or at `input/pu_candidates` if you generated `cse_pu_candidates.csv`.

You can still set `PIPELINE_INPUT_DIR` explicitly if you want to be extra clear in the terminal.

Before expensive feature extraction starts, `run_pipeline.py` now runs a DNS precheck:

- DNS-active rows continue into HTTP/RDAP/TLS/page feature extraction.
- DNS-failed rows are written to `output/dns_failed_urls.csv`.
- DNS-failed rows are not written to `enriched_features_v2.csv`.
- DNS-failed rows are not written to `enriched_features_v2_audit.csv`.
- DNS-failed rows are not used by `train_pu_model.py`.

Local safe CPU settings:

```powershell
$env:PIPELINE_INPUT_DIR = "input\pu_candidates"
$env:PIPELINE_OUTPUT_DIR = "output"
$env:PIPELINE_CONCURRENCY = "8"
$env:PIPELINE_HTTP_CONCURRENCY = "8"
$env:PIPELINE_DNS_CONCURRENCY = "64"
$env:PIPELINE_RDAP_CONCURRENCY = "20"
$env:PIPELINE_TLS_CONCURRENCY = "4"
$env:PIPELINE_ASN_CONCURRENCY = "4"
$env:PIPELINE_THREAD_WORKERS = "2"

.\venv\Scripts\python.exe run_pipeline.py
```

Expected outputs:

```text
output/enriched_features_v2.csv
output/enriched_features_v2_audit.csv
output/dns_failed_urls.csv
```

Use this one for training:

```text
output/enriched_features_v2_audit.csv
```

Why audit file?

The normal `enriched_features_v2.csv` is compact. The audit file keeps the metadata needed by PU training and review:

```text
url
domain
target_brand_domain
source_label
validation_status
validation_reason
```

## Step 3. Train The PU Model

Run:

```powershell
.\venv\Scripts\python.exe train_pu_model.py --features output\enriched_features_v2_audit.csv
```

Local Sophos-safe mode:

```powershell
.\venv\Scripts\python.exe train_pu_model.py `
  --features output\enriched_features_v2_audit.csv `
  --n-jobs 1
```

Expected outputs:

```text
output/pu_model/pu_model.joblib
output/pu_model/pu_scored_rows.csv
output/pu_model/pu_training_summary.json
```

## Step 4. Read The Training Summary

Open:

```text
output/pu_model/pu_training_summary.json
```

Important fields:

```text
positive_rows
unlabeled_rows
reliable_negative_rows
threshold
validation.tp
validation.fp
validation.tn
validation.fn
validation.accuracy
validation.precision
validation.recall
validation.specificity
validation.f1
kfold_cross_validation.overall.tp/fp/tn/fn
```

How to interpret:

- `positive_rows` should not be tiny. Aim for at least `50+`, better `100+`.
- `unlabeled_rows` can be large.
- `reliable_negative_rows` are mined from unlabeled rows.
- `threshold` is the selected cutoff for `pu_score`.
- `tp`, `fp`, `tn`, and `fn` show the confusion matrix for the validation split or k-fold rows.
- `recall` matters more than precision for detection, because missing phishing is costly.

If `positive_rows` is very low, do not trust the model yet. Collect more CSE-matched phishing positives first.

## Step 5. Read The Scored Rows

Open:

```text
output/pu_model/pu_scored_rows.csv
```

Important columns:

```text
url
domain
target_brand_domain
source_label
validation_status
pu_training_role
pu_score
pu_prediction
```

Meaning:

- `pu_score`: model risk score from `0` to `1`.
- `pu_prediction`: final prediction using the selected threshold.
- `pu_training_role`: how the row was used during PU training.

Possible `pu_training_role` values:

```text
positive
positive_validation
unlabeled
reliable_negative
reliable_negative_validation
ignored
```

For review, sort by highest `pu_score`.

## Step 6. Suggested Review Workflow

Use the model to prioritize review:

```text
pu_score >= 0.90  -> urgent phishing review
pu_score 0.70-0.90 -> suspicious, review next
pu_score 0.40-0.70 -> watchlist
pu_score < 0.40 -> lower priority
```

Do not blindly call every high score confirmed phishing. Treat high score as "needs fast review".

## Step 7. Feature Groups Used By The Model

The trainer automatically keeps numeric features and removes review/leakage columns.

Useful feature groups:

```text
Lexical:
lexical_similarity_score
min_edit_distance_to_brand_domain
homoglyph_similarity_score
brand_token_in_subdomain
brand_token_in_path
subdomain_depth

Infrastructure:
dns_check_pass
domain_age_days
domain_expiry_days
registrar_reputation_score
nameserver_reputation_score
asn_reputation_score
ip_seen_with_many_brands

TLS/HTTP:
https_enabled
ssl_valid
cert_age_days
url_fetch_success
redirect_count
redirect_cross_domain_count

Page evidence:
has_login_form
has_password_input
form_action_external
brand_token_or_logo_present
visual_brand_domain_mismatch

Hash evidence:
favicon_hash_matches_target_brand
favicon_hash_matches_known_phish
html_dom_hash_matches_known_phish
same_html_hash_domain_count_7d
```

Review-only columns, not model features:

```text
url
domain
target_brand_domain
critical_sector_entity_name
source_label
validation_status
validation_reason
raw favicon hash
raw DOM hash
source files
errors
old label column
```

## Full Local Run

Copy and run this from PowerShell:

```powershell
cd C:\Users\sathwik.kusuri\Documents\ai-challenge-ijmm\scripts\model_training_datasets

.\venv\Scripts\python.exe -m pip install -r requirements.txt

.\venv\Scripts\python.exe collect_cse_pu_urls.py --fetch-feeds

$env:PIPELINE_INPUT_DIR = "input\pu_candidates"
$env:PIPELINE_OUTPUT_DIR = "output"
$env:PIPELINE_CONCURRENCY = "8"
$env:PIPELINE_HTTP_CONCURRENCY = "8"
$env:PIPELINE_DNS_CONCURRENCY = "64"
$env:PIPELINE_RDAP_CONCURRENCY = "20"
$env:PIPELINE_TLS_CONCURRENCY = "4"
$env:PIPELINE_ASN_CONCURRENCY = "4"
$env:PIPELINE_THREAD_WORKERS = "2"

.\venv\Scripts\python.exe run_pipeline.py

.\venv\Scripts\python.exe train_pu_model.py --features output\enriched_features_v2_audit.csv --n-jobs 1
```

## Troubleshooting

### Problem: `Need at least 10 positive rows`

Meaning:

The trainer found too few confirmed phishing rows.

Fix:

```powershell
.\venv\Scripts\python.exe collect_cse_pu_urls.py --fetch-feeds --feed-threshold 0.78
```

Then rerun:

```powershell
.\venv\Scripts\python.exe run_pipeline.py
.\venv\Scripts\python.exe train_pu_model.py --features output\enriched_features_v2_audit.csv
```

### Problem: Most rows are unlabeled

This is normal for PU learning.

Unlabeled does not mean negative. The final trainer mines reliable negatives from the unlabeled pool.

### Problem: Sophos kills Python locally

Use lower settings:

```powershell
$env:PIPELINE_CONCURRENCY = "4"
$env:PIPELINE_HTTP_CONCURRENCY = "4"
$env:PIPELINE_DNS_CONCURRENCY = "8"
$env:PIPELINE_RDAP_CONCURRENCY = "20"
$env:PIPELINE_TLS_CONCURRENCY = "2"
$env:PIPELINE_ASN_CONCURRENCY = "2"
$env:PIPELINE_THREAD_WORKERS = "1"
```

Also train with:

```powershell
.\venv\Scripts\python.exe train_pu_model.py --features output\enriched_features_v2_audit.csv --n-jobs 1
```

### Problem: `source_label` missing during training

Use:

```text
output/enriched_features_v2_audit.csv
```

Do not use only:

```text
output/enriched_features_v2.csv
```

The compact file may not include enough metadata for clean PU review.

## Final Mental Model

The full flow is:

```text
CSE official domains
        +
current lexical suspicious URLs
        +
CSE-matched phishing feed URLs
        |
        v
input/pu_candidates/cse_pu_candidates.csv
        |
        v
run_pipeline.py DNS precheck
        |
        v
output/dns_failed_urls.csv for DNS-failed review only
        |
        v
run_pipeline.py feature extraction for DNS-active rows only
        |
        v
output/enriched_features_v2_audit.csv
        |
        v
train_pu_model.py
        |
        v
pu_score for prioritizing phishing review
```

## References

- OpenPhish community feed: https://openphish.com/phishing_feeds.html
- PhishTank developer feed fields: https://phishtank.org/developer_info.php
- PU learning Python package/docs: https://pulearn.github.io/pulearn/doc/pulearn/
