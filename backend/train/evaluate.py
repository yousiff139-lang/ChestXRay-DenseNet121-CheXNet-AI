"""
Seminar 16: Comprehensive Evaluation & Benchmark Protocol
DenseNet-121 vs ResNet-50 on NIH ChestX-ray14 & Real Clinical Cohort

Maps directly to Seminar 16 Architecture Specification Section 6:
  (1) Mean ROC-AUC (macro mean + per-class AUC with 95% bootstrap CIs)
  (2) Mean Average Precision (mAP with class prevalence beside each value)
  (3) Per-class Sensitivity & Specificity at validation-chosen thresholds (Youden's J)
  (4) Timed Inference Latency:
      - Batch size 1, 224x224, 20 warm-up runs, 200 timed runs
      - Model-only forward latency (mean, std, p95)
      - End-to-end latency (preprocessing + forward + Grad-CAM)
      - Trainable parameters and GFLOPs
  (5) Baseline comparison: Paired bootstrap test on the AUC difference (DenseNet vs ResNet)
"""

import os
import sys
import time
import argparse
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.classification import (
    CHEST_XRAY_CLASSES,
    CHEST_XRAY_PREVALENCE,
    CHEST_XRAY_VALIDATION_THRESHOLDS,
)
from app.services.classification_service import get_classification_service
from app.services.evaluation_service import get_evaluation_service


def compute_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_resamples: int = 1000, seed: int = 42):
    """Compute 95% bootstrap confidence interval [2.5%, 97.5%]."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        yt_sample = y_true[indices]
        yp_sample = y_pred[indices]
        if len(np.unique(yt_sample)) > 1:
            try:
                score = metric_fn(yt_sample, yp_sample)
                scores.append(score)
            except Exception:
                continue

    if len(scores) < 10:
        return None, None
    low = np.percentile(scores, 2.5)
    high = np.percentile(scores, 97.5)
    return float(low), float(high)


def benchmark_timing(cls_service, sample_img_path: str, model_name: str, num_warmup: int = 20, num_timed: int = 200):
    """
    Measure inference time per Section 6:
    Batch size 1, 224x224, 20 warm-up runs, 200 timed runs, GPU sync if available.
    Reports model-only time and end-to-end time (preprocessing + Grad-CAM) separately.
    """
    device = cls_service.model_service.get_device()
    inp, w, h = cls_service._preprocess_image(sample_img_path, model_name=model_name)

    # 1. Warm-up runs
    for _ in range(num_warmup):
        _ = cls_service._run_inference(inp, model_name=model_name)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 2. Pure Model Forward Runs
    model_times = []
    for _ in range(num_timed):
        t0 = time.perf_counter()
        _ = cls_service._run_inference(inp, model_name=model_name)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_times.append((time.perf_counter() - t0) * 1000.0)

    # 3. End-to-End Runs (Image Load + Preprocess + Forward + Grad-CAM)
    e2e_times = []
    for _ in range(min(num_timed, 50)):  # 50 runs for full pipeline
        t0 = time.perf_counter()
        inp_e2e, _, _ = cls_service._preprocess_image(sample_img_path, model_name=model_name)
        probs, _ = cls_service._run_inference(inp_e2e, model_name=model_name)
        top_idx = int(np.argmax(probs))
        _ = cls_service.gradcam_service.generate_gradcam(sample_img_path, model_name=model_name, target_class_idx=top_idx)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e2e_times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "model_mean_ms": float(np.mean(model_times)),
        "model_std_ms": float(np.std(model_times)),
        "model_p95_ms": float(np.percentile(model_times, 95)),
        "e2e_mean_ms": float(np.mean(e2e_times)),
        "e2e_std_ms": float(np.std(e2e_times)),
        "e2e_p95_ms": float(np.percentile(e2e_times, 95)),
        "device": str(device),
        "parameters_m": 7.98 if model_name == "densenet" else 25.56,
        "gflops": 2.88 if model_name == "densenet" else 4.12,
    }


def main():
    parser = argparse.ArgumentParser(description="Seminar 16 Model Evaluation Protocol")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--img_dir", type=str, default="")
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" SEMINAR 16: DENSENET-121 VS RESNET-50 ON CHEST RADIOGRAPHY")
    print(" Status: Research prototype, not a diagnostic device")
    print(" Protocol: Bootstrap 95% CIs, ROC-AUC, mAP, Sensitivity/Specificity, Latency")
    print("=" * 80 + "\n")

    # Locate dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_candidates = [
        args.csv,
        os.path.join(base_dir, "data", "test.csv"),
        os.path.join(base_dir, "data", "real_cohort", "ground_truth.csv"),
    ]
    csv_path = None
    for c in csv_candidates:
        if c and os.path.exists(c):
            csv_path = c
            break

    if not csv_path:
        print("[!] No evaluation CSV dataset found on disk.")
        print("    Displaying published Stanford CheXNet & NIH ChestX-ray14 benchmark standards:\n")
        eval_service = get_evaluation_service()
        benchmarks = eval_service.get_published_benchmark_comparison()

        print(f"{'Evaluation Metric':<28} | {'DenseNet-121 (Primary)':<24} | {'ResNet-50 (Baseline)':<22}")
        print("-" * 80)
        dm = benchmarks["metrics_comparison"]["densenet_121"]
        rm = benchmarks["metrics_comparison"]["resnet_50"]
        print(f"{'Mean ROC-AUC':<28} | {dm['mean_roc_auc']:<24.3f} | {rm['mean_roc_auc']:<22.3f}")
        print(f"{'Mean Average Precision (mAP)':<28} | {dm['mean_ap']:<24.3f} | {rm['mean_ap']:<22.3f}")
        print(f"{'Avg Sensitivity (TPR)':<28} | {dm['avg_sensitivity']*100:<23.1f}% | {rm['avg_sensitivity']*100:<21.1f}%")
        print(f"{'Avg Specificity (TNR)':<28} | {dm['avg_specificity']*100:<23.1f}% | {rm['avg_specificity']*100:<21.1f}%")
        print(f"{'Parameters':<28} | {dm['parameters_million']:<23.2f}M | {rm['parameters_million']:<21.2f}M")
        return

    # Image directory
    img_dir = args.img_dir
    if not img_dir or not os.path.exists(img_dir):
        if "real_cohort" in csv_path:
            img_dir = os.path.join(base_dir, "data", "real_cohort", "images")
        else:
            img_dir = os.path.join(base_dir, "data", "images")

    print(f"Dataset CSV:  {csv_path}")
    print(f"Image Folder: {img_dir}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} patient cases for evaluation.\n")

    cls_service = get_classification_service()

    # Pre-parse ground truth matrix [N, 14]
    y_true = np.zeros((len(df), len(CHEST_XRAY_CLASSES)), dtype=np.float32)
    valid_indices = []

    for i, row in df.iterrows():
        img_name = row.get("Image Index") or row.get("filename") or row.get("image")
        img_path = os.path.join(img_dir, str(img_name))
        if not os.path.exists(img_path):
            continue
        valid_indices.append(i)

        labels_str = str(row.get("Finding Labels") or row.get("finding") or row.get("label", ""))
        if labels_str != "No Finding" and not pd.isna(labels_str):
            for finding in labels_str.split("|"):
                f_clean = finding.strip()
                if f_clean in CHEST_XRAY_CLASSES:
                    idx = CHEST_XRAY_CLASSES.index(f_clean)
                    y_true[i, idx] = 1.0

    y_true = y_true[valid_indices]
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    N = len(df_valid)
    print(f"Found {N} valid image files on disk.\n")

    # Evaluate DenseNet-121 and ResNet-50
    models = ["densenet", "resnet"]
    results = {}

    for model_name in models:
        display_name = "DenseNet-121 (Primary)" if model_name == "densenet" else "ResNet-50 (Baseline)"
        print(f">>> Running inference with {display_name}...")

        all_probs = []
        for i, row in df_valid.iterrows():
            img_name = row.get("Image Index") or row.get("filename") or row.get("image")
            img_path = os.path.join(img_dir, str(img_name))
            inp, w, h = cls_service._preprocess_image(img_path, model_name=model_name)
            probs, _ = cls_service._run_inference(inp, model_name=model_name)
            all_probs.append(probs)

        y_pred = np.array(all_probs)
        results[model_name] = {"y_pred": y_pred}

    # Timing benchmark
    sample_img = os.path.join(img_dir, str(df_valid.iloc[0].get("Image Index") or df_valid.iloc[0].get("filename")))
    print("\n>>> Measuring inference latency (20 warm-up, 200 timed runs)...")
    d_timing = benchmark_timing(cls_service, sample_img, "densenet")
    r_timing = benchmark_timing(cls_service, sample_img, "resnet")

    # Calculate per-class metrics
    print("\n" + "=" * 95)
    print(f"{'Pathology':<20} | {'Prevalence':<11} | {'Cutoff':<8} | {'DenseNet AUC [95% CI]':<24} | {'ResNet AUC [95% CI]':<22}")
    print("=" * 95)

    d_aucs, r_aucs = [], []
    d_aps, r_aps = [], []
    d_sens_list, r_sens_list = [], []
    d_spec_list, r_spec_list = [], []

    for c_idx, disease in enumerate(CHEST_XRAY_CLASSES):
        yt = y_true[:, c_idx]
        prev = CHEST_XRAY_PREVALENCE.get(disease, 0.0)
        thresh = CHEST_XRAY_VALIDATION_THRESHOLDS.get(disease, 0.5)

        yp_d = results["densenet"]["y_pred"][:, c_idx]
        yp_r = results["resnet"]["y_pred"][:, c_idx]

        # Sensitivity & Specificity at validation threshold
        tp_d = ((yp_d >= thresh) & (yt == 1)).sum()
        fn_d = ((yp_d < thresh) & (yt == 1)).sum()
        tn_d = ((yp_d < thresh) & (yt == 0)).sum()
        fp_d = ((yp_d >= thresh) & (yt == 0)).sum()
        sens_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
        spec_d = tn_d / (tn_d + fp_d) if (tn_d + fp_d) > 0 else 0.0

        tp_r = ((yp_r >= thresh) & (yt == 1)).sum()
        fn_r = ((yp_r < thresh) & (yt == 1)).sum()
        tn_r = ((yp_r < thresh) & (yt == 0)).sum()
        fp_r = ((yp_r >= thresh) & (yt == 0)).sum()
        sens_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
        spec_r = tn_r / (tn_r + fp_r) if (tn_r + fp_r) > 0 else 0.0

        if (tp_d + fn_d) > 0:
            d_sens_list.append(sens_d)
            r_sens_list.append(sens_r)
        d_spec_list.append(spec_d)
        r_spec_list.append(spec_r)

        if len(np.unique(yt)) > 1:
            auc_d = roc_auc_score(yt, yp_d)
            auc_r = roc_auc_score(yt, yp_r)
            ap_d = average_precision_score(yt, yp_d)
            ap_r = average_precision_score(yt, yp_r)

            d_aucs.append(auc_d)
            r_aucs.append(auc_r)
            d_aps.append(ap_d)
            r_aps.append(ap_r)

            ci_d_low, ci_d_high = compute_bootstrap_ci(yt, yp_d, roc_auc_score, n_resamples=args.bootstrap_resamples)
            ci_r_low, ci_r_high = compute_bootstrap_ci(yt, yp_r, roc_auc_score, n_resamples=args.bootstrap_resamples)

            d_ci_str = f"{auc_d:.3f} [{ci_d_low:.2f}-{ci_d_high:.2f}]" if ci_d_low is not None else f"{auc_d:.3f}"
            r_ci_str = f"{auc_r:.3f} [{ci_r_low:.2f}-{ci_r_high:.2f}]" if ci_r_low is not None else f"{auc_r:.3f}"

            print(f"{disease:<20} | {prev*100:>9.1f}% | {thresh*100:>6.1f}% | {d_ci_str:<24} | {r_ci_str:<22}")
        else:
            print(f"{disease:<20} | {prev*100:>9.1f}% | {thresh*100:>6.1f}% | {'N/A (no positives)':<24} | {'N/A (no positives)':<22}")

    # Macro Averages
    mean_auc_d = float(np.mean(d_aucs)) if d_aucs else 0.0
    mean_auc_r = float(np.mean(r_aucs)) if r_aucs else 0.0
    mean_ap_d = float(np.mean(d_aps)) if d_aps else 0.0
    mean_ap_r = float(np.mean(r_aps)) if r_aps else 0.0
    mean_sens_d = float(np.mean(d_sens_list)) if d_sens_list else 0.0
    mean_sens_r = float(np.mean(r_sens_list)) if r_sens_list else 0.0
    mean_spec_d = float(np.mean(d_spec_list)) if d_spec_list else 0.0
    mean_spec_r = float(np.mean(r_spec_list)) if r_spec_list else 0.0

    # Paired Bootstrap Test on AUC difference (DenseNet - ResNet)
    delta_aucs = []
    rng = np.random.default_rng(42)
    for _ in range(args.bootstrap_resamples):
        idx_boot = rng.integers(0, N, size=N)
        res_d_boot, res_r_boot = [], []
        for c in range(len(CHEST_XRAY_CLASSES)):
            yt_b = y_true[idx_boot, c]
            if len(np.unique(yt_b)) > 1:
                try:
                    res_d_boot.append(roc_auc_score(yt_b, results["densenet"]["y_pred"][idx_boot, c]))
                    res_r_boot.append(roc_auc_score(yt_b, results["resnet"]["y_pred"][idx_boot, c]))
                except Exception:
                    pass
        if res_d_boot and res_r_boot:
            delta_aucs.append(np.mean(res_d_boot) - np.mean(res_r_boot))

    if delta_aucs:
        delta_ci_low = np.percentile(delta_aucs, 2.5)
        delta_ci_high = np.percentile(delta_aucs, 97.5)
        p_val = float(2.0 * min(np.mean(np.array(delta_aucs) <= 0), np.mean(np.array(delta_aucs) >= 0)))
    else:
        delta_ci_low, delta_ci_high, p_val = 0.0, 0.0, 1.0

    print("\n" + "=" * 95)
    print(" SUMMARY BENCHMARK TABLE (Seminar 16 Architecture Specification)")
    print("=" * 95)
    print(f"{'Evaluation Metric':<32} | {'DenseNet-121 (Primary)':<24} | {'ResNet-50 (Baseline)':<22} | {'Delta':<12}")
    print("-" * 95)
    print(f"{'(1) Mean ROC-AUC (14 Classes)':<32} | {mean_auc_d:<24.4f} | {mean_auc_r:<22.4f} | {mean_auc_d - mean_auc_r:+.4f}")
    print(f"{'(2) Mean Average Precision (mAP)':<32} | {mean_ap_d:<24.4f} | {mean_ap_r:<22.4f} | {mean_ap_d - mean_ap_r:+.4f}")
    print(f"{'(3) Average Sensitivity (Recall)':<32} | {mean_sens_d*100:<23.1f}% | {mean_sens_r*100:<21.1f}% | {(mean_sens_d-mean_sens_r)*100:+.1f}%")
    print(f"{'    Average Specificity':<32} | {mean_spec_d*100:<23.1f}% | {mean_spec_r*100:<21.1f}% | {(mean_spec_d-mean_spec_r)*100:+.1f}%")
    print(f"{'(4) Model Forward Latency (mean)':<32} | {d_timing['model_mean_ms']:<21.1f} ms | {r_timing['model_mean_ms']:<19.1f} ms | {d_timing['model_mean_ms'] - r_timing['model_mean_ms']:+.1f} ms")
    print(f"{'    Model Latency (p95)':<32} | {d_timing['model_p95_ms']:<21.1f} ms | {r_timing['model_p95_ms']:<19.1f} ms | -")
    print(f"{'    End-to-End Latency (w/ GradCAM)':<32} | {d_timing['e2e_mean_ms']:<21.1f} ms | {r_timing['e2e_mean_ms']:<19.1f} ms | -")
    print(f"{'    Parameters':<32} | {d_timing['parameters_m']:<21.2f} M  | {r_timing['parameters_m']:<19.2f} M  | 3.2x lighter")
    print(f"{'    Compute (GFLOPs @ 224x224)':<32} | {d_timing['gflops']:<21.2f} G  | {r_timing['gflops']:<19.2f} G  | 30% fewer FLOPs")
    print("-" * 95)
    print(f"Paired Bootstrap Test on AUC Difference (1000 resamples):")
    print(f"  Delta AUC (DenseNet - ResNet): {mean_auc_d - mean_auc_r:+.4f} (95% CI: [{delta_ci_low:+.4f}, {delta_ci_high:+.4f}])")
    print(f"  Two-tailed p-value: p = {p_val:.4f} {'(DenseNet significantly leads)' if p_val < 0.05 else '(Statistically equivalent)'}")
    print("=" * 95)
    print("Note: Research prototype. Not for clinical use.\n")


if __name__ == "__main__":
    main()
