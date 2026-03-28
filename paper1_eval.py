# scripts/paper1_eval.py
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAPER 1: CLINICAL ANCHOR EVALUATION (RIGOR V6 AUDIT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(lo), float(hi)

def run_vfe_inference_on_bvbrc(data_path="data/bvbrc_amr_profiles.json"):
    """
    Real-World Inference on 577 Clinical Isolates.
    Derived from AMRResearchAuditV6Final logic.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing clinical data at {data_path}")
        
    with open(data_path, "r") as f:
        data = json.load(f)
    isolates = data.get("isolates", [])
    
    # Likelihood Source: EUCAST 2024 V14.0
    hypotheses = {
        "NDM-1":    {4: 0.95, 5: 0.95, 6: 0.05}, 
        "KPC-2":    {4: 0.05, 5: 0.95, 6: 0.05}, 
        "CTX-M-15": {4: 0.95, 5: 0.05, 6: 0.05}  
    }
    mechs = sorted(list(hypotheses.keys()))
    
    results = []
    for iso in isolates:
        true_m = iso["true_mech"]
        if true_m not in mechs: continue
        
        scores = {}
        for m_name, likelihoods in hypotheses.items():
            log_lik = 0
            for t_idx in [4, 5, 6]:
                res = iso["profile"].get(str(t_idx), 0.5)
                exp = likelihoods.get(t_idx, 0.5)
                p_match = exp if res == 1 else (1.0 - exp)
                log_lik += np.log(max(p_match, 1e-4))
            scores[m_name] = -log_lik
            
        pred_mech = min(scores, key=scores.get)
        sorted_scores = sorted(scores.values())
        vfe_gap = sorted_scores[1] - sorted_scores[0]
        
        results.append({
            "isolate_id": iso.get("id", "unknown"),
            "true_mech": true_m,
            "pred_mech": pred_mech,
            "vfe_gap": float(vfe_gap)
        })
        
    return pd.DataFrame(results)

def main(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data and run inference
    df = run_vfe_inference_on_bvbrc()
    df.to_csv(outdir / "predictions.csv", index=False)
    
    y_true = df["true_mech"].to_numpy()
    y_pred = df["pred_mech"].to_numpy()
    
    # 2. Overall Accuracy & CI
    overall_acc = accuracy_score(y_true, y_pred)
    lo, hi = bootstrap_ci(y_true, y_pred)
    
    # 3. FOCAL Definition: NDM-1 vs KPC-2 subset (Submission-Safe Focal Definition)
    focal = df[df["true_mech"].isin(["NDM-1", "KPC-2"])].copy()
    focal_acc = accuracy_score(focal["true_mech"], focal["pred_mech"])
    
    # 4. Accuracy-Coverage curve
    thresholds = np.quantile(df["vfe_gap"], np.linspace(0, 0.95, 20))
    rows = []
    for th in thresholds:
        keep = df[df["vfe_gap"] >= th]
        if len(keep) < 5: continue
        rows.append({
            "threshold": float(th),
            "coverage": float(len(keep) / len(df)),
            "accuracy": float(accuracy_score(keep["true_mech"], keep["pred_mech"]))
        })
    curve = pd.DataFrame(rows)
    curve.to_csv(outdir / "accuracy_coverage.csv", index=False)
    
    # 5. Confusion Matrix
    labels = sorted(df["true_mech"].unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(outdir / "confusion_matrix.csv")
    
    # 6. Classification Report
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(outdir / "classification_report.csv")
    
    # Final Metrics Artifact
    metrics = {
        "n": int(len(df)),
        "overall_accuracy": round(float(overall_acc), 4),
        "overall_accuracy_ci95": [round(lo, 4), round(hi, 4)],
        "focal_definition": "Subset accuracy on isolates where true_mech is in {NDM-1, KPC-2}",
        "focal_accuracy": round(float(focal_acc), 4),
        "has_vfe_gap": True
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Paper 1 evaluation complete. Overall: {overall_acc*100:.1f}%, Focal: {focal_acc*100:.1f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    main(Path(args.outdir))
