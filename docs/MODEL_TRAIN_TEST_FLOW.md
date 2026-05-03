# Standard + Hardened PU Model Train/Test Flow

This is the current production flow for training and validating both PU models:

- Standard model: conservative high-confidence phishing model.
- Hardened model: excludes rule/evidence leakage features and catches suspicious infrastructure patterns.
- Final classifier: uses both models together.

Run all commands from this folder:

```powershell
cd C:\Users\sathwik.kusuri\Documents\ai-challenge-ijmm\scripts\model_training_datasets
```

Use the venv Python:

```powershell
.\venv\Scripts\python.exe
```

## 1. Training Flow

### Step 1. Prepare feed/local candidate input

`run_pipeline.py` now performs CSE matching directly from the unified CSE list:

```text
input\cse\cse_list.csv
```

The CSE list must contain:

```text
domains,cse
```

So the candidate input can be raw feed/local URLs in CSV/XLSX files, or one-URL-per-line TXT feed files. The pipeline fills `target_brand_domain` and `critical_sector_entity_name` by matching each candidate against `input\cse\cse_list.csv`, then drops rows below `PIPELINE_CSE_MATCH_THRESHOLD` before DNS and feature extraction.

Use `collect_cse_pu_urls.py` only when you still want it to fetch feeds or dnstwist candidates into a candidate file first.

```powershell
.\venv\Scripts\python.exe collect_cse_pu_urls.py --fetch-feeds
```

Expected output:

```text
input\pu_candidates\cse_pu_candidates.csv
```

Important: `collect_cse_pu_urls.py` recreates this CSV each time. It does not append to the existing `input\pu_candidates\cse_pu_candidates.csv`.

When you run multiple source options together, the output file is a fresh combined dataset from those sources. For example:

```powershell
.\venv\Scripts\python.exe collect_cse_pu_urls.py `
  --fetch-feeds `
  --dnstwist `
  --require-dns-active `
  --dnstwist-workers 2 `
  --feed-workers 8 `
  --cpu-gate-threshold 85 `
  --dns-timeout 1.0 `
  --dns-lifetime 1.5 `
  --max-active-urls 5000
```

This rebuilds `input\pu_candidates\cse_pu_candidates.csv` from:

- local input files from `input\dataset`, unless `--no-local` is passed
- OpenPhish and PhishTank live feeds, because `--fetch-feeds` is passed
- DNS-active dnstwist variants that lexically match the CSE list, because `--dnstwist` is passed
- only final rows that pass DNS lookup, because `--require-dns-active` is passed
- at most 5000 final active URLs, because `--max-active-urls 5000` is passed

If live feed download is blocked, use saved feed files instead:

```powershell
.\venv\Scripts\python.exe collect_cse_pu_urls.py `
  --openphish-file input\data\openphish_feed.txt `
  --phishtank-file input\data\phishtank_online_valid.csv
```

To add DNS-active dnstwist variants for CSE domains:

```powershell
.\venv\Scripts\python.exe collect_cse_pu_urls.py `
  --dnstwist `
  --require-dns-active `
  --dnstwist-workers 2 `
  --feed-workers 8 `
  --cpu-gate-threshold 85 `
  --dns-timeout 1.0 `
  --dns-lifetime 1.5 `
  --max-active-urls 5000
```

For a smaller first run, fuzz specific CSE domains:

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

dnstwist rows are written as `Unlabeled` by default because DNS-active lookalikes are suspicious but not automatically confirmed phishing. Use `--dnstwist-label Phishing` only after independent verification.

Parallel + CPU gate notes:

- `--dnstwist-workers` controls how many CSE domains are processed by dnstwist at once.
- `--feed-workers` controls parallel lexical matching for OpenPhish/PhishTank URLs.
- `--local-workers` controls parallel lexical matching for local input files.
- `--dnstwist-dns-workers` and `--dns-workers` control DNS active checks.
- DNS active checks use a fast active-only resolver with caching. `--dns-timeout 1.0` and `--dns-lifetime 1.5` keep dead domains from blocking too long.
- `--cpu-gate-threshold 85` means new parallel work waits while system CPU is above 85%.
- `--cpu-gate-max-wait-seconds 600` fails the run if CPU remains above threshold for too long.
- `--max-active-urls 5000` writes at most 5000 final rows after DNS-active filtering; use `--max-active-urls 0` to disable this cap.
- Progress bars show the active stage, such as dnstwist domain generation, DNS checks, feed matching, or lexical matching.
- Use `--cpu-gate-threshold 100` only if you intentionally want to disable the gate.

The strict CPU gate uses `psutil`, so install requirements first:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you want only dnstwist URLs:

```powershell
.\venv\Scripts\python.exe collect_cse_pu_urls.py `
  --no-local `
  --dnstwist `
  --require-dns-active `
  --dnstwist-workers 2 `
  --cpu-gate-threshold 85 `
  --dns-timeout 1.0 `
  --dns-lifetime 1.5 `
  --max-active-urls 5000
```

Without `--fetch-feeds`, OpenPhish and PhishTank are not included.

If you only want local input files from `input\dataset`:

```powershell
.\venv\Scripts\python.exe collect_cse_pu_urls.py
```

### Step 2. Extract training features

By default, `run_pipeline.py` reads raw feed/local files from `input\dataset`. The pipeline loads `input\cse\cse_list.csv`, fills the CSE target fields, and filters rows below `PIPELINE_CSE_MATCH_THRESHOLD` before feature extraction. Set `PIPELINE_INPUT_DIR` only when you want to use another folder, such as `input\pu_candidates`.

```powershell
$env:PIPELINE_INPUT_DIR = "input\dataset"
$env:PIPELINE_OUTPUT_DIR = "output"
$env:PIPELINE_OUTPUT_FILE = "output\enriched_features_v2.csv"
$env:PIPELINE_OUTPUT_AUDIT_FILE = "output\enriched_features_v2_audit.csv"
$env:PIPELINE_OUTPUT_DNS_FAILED_FILE = "output\dns_failed_urls.csv"

.\venv\Scripts\python.exe run_pipeline.py
```

Expected outputs:

```text
output\enriched_features_v2.csv
output\enriched_features_v2_audit.csv
output\dns_failed_urls.csv
```

Use this file for model training:

```text
output\enriched_features_v2_audit.csv
```

### Step 3. Train the standard model

Use 5-fold CV because the dataset is small.

```powershell
.\venv\Scripts\python.exe train_pu_model.py `
  --features output\enriched_features_v2_audit.csv `
  --cv-folds 5 `
  --n-jobs 1
```

Expected outputs:

```text
output\pu_model\pu_model.joblib
output\pu_model\pu_training_summary.json
output\pu_model\pu_scored_rows.csv
output\pu_model\pu_kfold_cv_scored_rows.csv
```

### Step 4. Train the hardened model

The hardened model removes rule/evidence-style columns used to create `validation_status`.

```powershell
.\venv\Scripts\python.exe train_pu_model.py `
  --features output\enriched_features_v2_audit.csv `
  --hardened `
  --cv-folds 5 `
  --n-jobs 1
```

Expected outputs:

```text
output\pu_model_hardened\pu_model.joblib
output\pu_model_hardened\pu_training_summary.json
output\pu_model_hardened\pu_scored_rows.csv
output\pu_model_hardened\pu_kfold_cv_scored_rows.csv
```

## 2. Test Dataset Setup

The dedicated raw test input folder is:

```text
input\test_dataset
```

The default test file is:

```text
input\test_dataset\test_dataset.csv
```

Minimum format:

```csv
url
https://example.com
https://example2.com
```

This is valid for production-style unknown URL testing. In this mode, `target_brand_domain` may be empty.

## 3. Production-Style Validation

Run this when you want to extract fresh live features and classify with both models:

```powershell
.\venv\Scripts\python.exe validate_test_dataset.py
```

This executes:

```text
input\test_dataset\test_dataset.csv
  -> run_pipeline.py
  -> output\test_results\enriched_features_audit.csv
  -> score_pu_models.py
  -> output\test_results\model_classification_results.csv
```

Expected outputs:

```text
output\test_results\enriched_features.csv
output\test_results\enriched_features_audit.csv
output\test_results\dns_failed_urls.csv
output\test_results\model_classification_results.csv
output\test_results\model_scores_long.csv
output\test_results\model_evaluation_metrics.json
```

`model_evaluation_metrics.json` is written when `validate_test_dataset.py` runs. If the test data has usable ground-truth labels, it includes TP, FP, TN, FN, accuracy, precision, recall, specificity, F1, balanced accuracy, and threshold sweeps for the standard and hardened models. If the test data is unlabeled, the JSON records why metrics were skipped.

## 4. Reuse Existing Test Features

Use this if you already ran feature extraction and only want to rerun model scoring:

```powershell
.\venv\Scripts\python.exe validate_test_dataset.py --skip-extraction
```

Or call the scorer directly:

```powershell
.\venv\Scripts\python.exe score_pu_models.py `
  --features output\test_results\enriched_features_audit.csv `
  --output output\test_results\model_classification_results.csv `
  --long-output output\test_results\model_scores_long.csv `
  --metrics-output output\test_results\model_evaluation_metrics.json
```

## 5. CSE-Targeted Validation

Use this when you want the test flow to include the same CSE matching step used before training.

Create a separate CSE-matched test input:

```powershell
New-Item -ItemType Directory -Force input\test_dataset_cse

.\venv\Scripts\python.exe collect_cse_pu_urls.py `
  --input input\test_dataset\test_dataset.csv `
  --output input\test_dataset_cse\test_dataset_cse.csv
```

Then validate into a separate output folder:

```powershell
.\venv\Scripts\python.exe validate_test_dataset.py `
  --input-dir input\test_dataset_cse `
  --output-dir output\test_results_cse
```

Expected outputs:

```text
output\test_results_cse\enriched_features.csv
output\test_results_cse\enriched_features_audit.csv
output\test_results_cse\dns_failed_urls.csv
output\test_results_cse\model_classification_results.csv
output\test_results_cse\model_scores_long.csv
```

## 6. Final Classification Rules

The production classifier in `score_pu_models.py` uses this logic:

```text
standard positive
  -> confirmed_phishing, high confidence

pipeline status phishing + hardened positive
  -> confirmed_phishing, high confidence

hardened positive only
  -> suspicious_needs_review, medium confidence

pipeline status suspected + both models below threshold
  -> suspicious_needs_review, low confidence

both models below threshold and no suspicious pipeline status
  -> low_risk, low confidence
```

Production meaning:

- `confirmed_phishing`: block or escalate as phishing.
- `suspicious_needs_review`: do not auto-block as final phishing; review or monitor.
- `low_risk`: no phishing action from the model.

## 7. Check Results

Main result file:

```text
output\test_results\model_classification_results.csv
```

Count final classes:

```powershell
Import-Csv output\test_results\model_classification_results.csv |
  Group-Object final_classification |
  Select-Object Name,Count |
  Format-Table -AutoSize
```

Show both-model positive rows:

```powershell
Import-Csv output\test_results\model_classification_results.csv |
  Where-Object { [int]$_.standard_prediction -eq 1 -and [int]$_.hardened_prediction -eq 1 } |
  Select-Object url,validation_status,standard_score,hardened_score,final_classification |
  Format-Table -AutoSize
```

Show hardened-only rows:

```powershell
Import-Csv output\test_results\model_classification_results.csv |
  Where-Object { [int]$_.standard_prediction -eq 0 -and [int]$_.hardened_prediction -eq 1 } |
  Select-Object url,validation_status,standard_score,hardened_score,final_classification,classification_confidence |
  Format-Table -AutoSize
```

Check specific domains:

```powershell
$check = @(
  "axismedlife.com",
  "arcanara.org",
  "aamsanchar.com",
  "aiims-edu.com",
  "icicilombard.shop",
  "vodafoneidea.store",
  "sewa99.org",
  "nigamtravels.com",
  "icicibanks.net"
)

Import-Csv output\test_results\model_classification_results.csv |
  Where-Object {
    $host = ([uri]$_.url).Host -replace "^www\.", ""
    $check -contains $host
  } |
  Select-Object url,validation_status,standard_prediction,hardened_prediction,standard_score,hardened_score,final_classification,classification_confidence |
  Format-Table -AutoSize
```

## 8. Clean Flow Summary

Training:

```text
input\dataset\*.csv / *.xlsx / *.txt
run_pipeline.py
  -> output\enriched_features_v2_audit.csv
train_pu_model.py
  -> output\pu_model
train_pu_model.py --hardened
  -> output\pu_model_hardened
```

Testing:

```text
input\test_dataset\test_dataset.csv
validate_test_dataset.py
  -> output\test_results\model_classification_results.csv
```

Optional CSE-targeted testing:

```text
collect_cse_pu_urls.py --input input\test_dataset\test_dataset.csv --output input\test_dataset_cse\test_dataset_cse.csv
validate_test_dataset.py --input-dir input\test_dataset_cse --output-dir output\test_results_cse
```

## 9. Important Notes

- Do not use the old ad hoc `output\url_tests` folder for validation.
- Use `output\test_results` for raw production testing.
- Use `output\test_results_cse` for CSE-targeted testing.
- Use `output\enriched_features_v2_audit.csv` for training, not the compact CSV.
- Live validation can change slightly between runs because DNS, HTTP, redirects, page content, and TLS data can change.
- A hardened-only positive is suspicious, not automatic final phishing.
- Standard positive is the safer high-confidence phishing signal.
