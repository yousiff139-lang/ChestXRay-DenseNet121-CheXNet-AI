"""
Real Clinical Dataset Downloader & Benchmark Suite.

Downloads verified, real clinical chest radiographs with ground truth diagnoses
from the Cohen et al. open medical collection (curated from NEJM, Lancet, etc.):
- Real Normal / Healthy Controls ("No Finding")
- Real Viral Pneumonia / Infiltrates
- Real Bacterial Pneumonia / Consolidation
- Real Severe ARDS / Opacities

Generates:
- backend/data/real_cohort/images/*.jpg
- backend/data/real_cohort/ground_truth.csv
"""

import os
import io
import urllib.request
import pandas as pd
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real_cohort")
IMG_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
BASE_URL = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/"

def main():
    print("=" * 65)
    print("Downloading Real Clinical Chest X-Ray Cohort with Ground Truth")
    print("=" * 65)

    meta_url = BASE_URL + "metadata.csv"
    req = urllib.request.Request(meta_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()))

    xrays = df[(df["modality"] == "X-ray") & (df["view"].isin(["PA", "AP"]))].copy()

    # 1. Select distinct clinical categories
    normals = xrays[xrays["finding"] == "No Finding"].head(6)
    virals = xrays[xrays["finding"].str.contains("Viral|COVID", na=False)].head(6)
    bacterials = xrays[xrays["finding"].str.contains("Bacterial", na=False)].head(4)
    pneumonias = xrays[xrays["finding"] == "Pneumonia"].head(4)

    selected = pd.concat([normals, virals, bacterials, pneumonias]).reset_index(drop=True)
    print(f"Total curated cases: {len(selected)}")

    records = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for idx, row in selected.iterrows():
        raw_fn = row["filename"]
        img_url = BASE_URL + "images/" + raw_fn
        ext = os.path.splitext(raw_fn)[1].lower() or ".jpg"
        clean_fn = f"case_{idx+1:02d}_{'normal' if row['finding'] == 'No Finding' else 'pathology'}{ext}"
        local_path = os.path.join(IMG_DIR, clean_fn)

        print(f"[{idx+1:02d}/{len(selected):02d}] Fetching {raw_fn[:30]}... -> {clean_fn}")
        try:
            r = urllib.request.Request(img_url, headers=headers)
            with urllib.request.urlopen(r, timeout=10) as img_resp:
                img_data = img_resp.read()

            # Ensure valid image by opening with PIL
            pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
            # Save standardized as JPEG
            jpeg_fn = f"case_{idx+1:02d}_{'normal' if row['finding'] == 'No Finding' else 'pathology'}.jpg"
            jpeg_path = os.path.join(IMG_DIR, jpeg_fn)
            pil_img.save(jpeg_path, "JPEG", quality=92)

            # Map finding to 14 ChestX-ray14 labels
            if row["finding"] == "No Finding":
                labels = "No Finding"
            elif "Bacterial" in str(row["finding"]):
                labels = "Pneumonia|Consolidation"
            elif "Viral" in str(row["finding"]) or "COVID" in str(row["finding"]):
                labels = "Pneumonia|Infiltration"
            else:
                labels = "Pneumonia"

            records.append({
                "Image Index": jpeg_fn,
                "Finding Labels": labels,
                "Clinical Diagnosis": str(row["finding"]),
                "Patient ID": str(row["patientid"]),
                "View": str(row["view"]),
                "Notes": str(row.get("clinical_notes", ""))[:120]
            })

        except Exception as e:
            print(f"    [!] Error downloading {raw_fn}: {e}")

    # Save Ground Truth CSV
    gt_df = pd.DataFrame(records)
    gt_df.to_csv(CSV_PATH, index=False)
    print(f"\n[✓] Successfully downloaded {len(records)} real clinical X-rays.")
    print(f"[✓] Ground truth saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()

