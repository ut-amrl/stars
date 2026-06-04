#!/usr/bin/env bash
set -euo pipefail

# This script downloads the real dataset and pre-trained model files for SWM/STARS
# from public Google Drive links and caches them in the data directory.

DATA_DIR="data"
mkdir -p "$DATA_DIR"

# Google Drive File IDs (Replace these with your actual uploaded file IDs)
COMBINED_H5_ID="1EljmDHsf_m1Yxn9sGW53B5t0yGkfZPjI"     # combined_v1.h5 (563MB)
SNS_EVAL_H5_ID="13q4wvD7khmwEYIfhI1D8971JoGi7gNy3"     # social_nav_eval_combined.h5 (4.2MB)
SPLIT_INDICES_ID="1jL3gxXDlR9UBjowxZMhbucsfs7BV9B64"   # split_indices_final.pt (18KB)
BEST_MODEL_ID="1fuX-a70Tc8jKPhK63Moc46JtxqwY1oTD"      # best_model_optuna.pt (16MB)
DATASET_SPLITS_ID="18CI8lNHGhcl6gBpHE31U_Su0f4bAR6m1"  # dataset_splits.zip (500MB, optional)

download_from_gdrive() {
  local file_id="$1"
  local output_name="$2"
  
  if [[ "$file_id" == "PLACEHOLDER_"* ]]; then
    echo "Skipping download of $output_name: Please replace placeholder with your actual Google Drive File ID."
    return 0
  fi

  echo "Downloading $output_name from Google Drive..."
  python3 -c "
import urllib.request, urllib.parse, re, sys, http.cookiejar
file_id = '${file_id}'
output_name = '${output_name}'
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
try:
    res = opener.open(f'https://drive.google.com/uc?export=download&id={file_id}')
    content_type = res.headers.get_content_type()
    if 'html' in content_type.lower():
        html = res.read().decode('utf-8', errors='ignore')
        if 'id=\"download-form\"' in html:
            action = re.search(r'action=\"([^\"]+)\"', html).group(1)
            inputs = re.findall(r'<input type=\"hidden\" name=\"([^\"]+)\" value=\"([^\"]+)\"', html)
            query = urllib.parse.urlencode({name: val for name, val in inputs})
            download_url = f'{action}?{query}'
            res = opener.open(download_url)
        else:
            print(f'Warning: Google Drive returned HTML instead of file: {html[:200]}', file=sys.stderr)
            sys.exit(1)
            
    with open(output_name, 'wb') as out_file:
        chunk_size = 1024 * 1024
        while True:
            chunk = res.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
except Exception as e:
    print(f'Error downloading file {output_name}: {e}', file=sys.stderr)
    sys.exit(1)
"
}

process_file() {
  local filename="$1"
  local gdrive_id="$2"
  local is_zip="${3:-false}"
  
  local target_name="$filename"
  if [[ "$is_zip" == "true" ]]; then
    target_name="${filename%.zip}"
  fi

  if [[ -e "$DATA_DIR/$target_name" ]]; then
    echo "$target_name already exists in $DATA_DIR; skipping."
    return 0
  fi
  
  # Download from Google Drive
  download_from_gdrive "$gdrive_id" "$DATA_DIR/$filename"
  
  if [[ "$is_zip" == "true" && -e "$DATA_DIR/$filename" ]]; then
    echo "Unzipping $DATA_DIR/$filename..."
    unzip -q -o "$DATA_DIR/$filename" -d "$DATA_DIR"
    echo "Removing $DATA_DIR/$filename..."
    rm -f "$DATA_DIR/$filename"
  fi
}

echo "Processing real dataset files..."
process_file "combined_v1.h5" "$COMBINED_H5_ID"
process_file "social_nav_eval_combined.h5" "$SNS_EVAL_H5_ID"
process_file "split_indices_final.pt" "$SPLIT_INDICES_ID"
process_file "best_model_optuna.pt" "$BEST_MODEL_ID"
process_file "dataset_splits.zip" "$DATASET_SPLITS_ID" true

echo "Done. All dataset files processed."
