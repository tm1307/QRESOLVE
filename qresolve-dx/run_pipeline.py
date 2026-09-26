#!/usr/bin/env python3
"""
QResolve-Dx: End-to-End Pipeline Runner

Two-tier architecture:
  COMMON DISEASES (Breast Cancer, Parkinson's):
    → Classical ML on REAL public datasets
    → XGBoost with calibrated probabilities

  RARE DISEASES (Marfan, Loeys-Dietz, Beals):
    → Classical ML triage on HPO-frequency-sampled data
    → Confusion detector flags hard cases
    → Quantum kernel SVM (ZZFeatureMap + QSVM) resolves hard pairs
    → Benchmarked against classical RBF-SVM with McNemar's test

Each phase is independently demoable — if quantum fails, classical is the demo.
"""

from __future__ import annotations

import sys
import os

# Fix Windows console encoding for Unicode box characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import time
import pickle
import warnings
import traceback
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.disease_data import (
    ALL_DISEASES, DISEASE_NAMES, DISEASE_LABEL_MAP, LABEL_DISEASE_MAP,
    ALL_HPO_TERMS, HPO_TERM_INDEX, NUM_FEATURES, NUM_CLASSES,
    KNOWN_CONFUSION_PAIRS, HPO_TERMS,
    RARE_DISEASES, RARE_DISEASE_NAMES, COMMON_DISEASES,
)


def ensure_dirs():
    """Create necessary directories."""
    dirs = [
        os.path.join(PROJECT_ROOT, 'data', 'processed'),
        os.path.join(PROJECT_ROOT, 'benchmarks'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# COMMON DISEASE PIPELINE (Real Datasets)
# ═══════════════════════════════════════════════════════════════════════

def run_breast_cancer_pipeline():
    """Run breast cancer detection on REAL Wisconsin dataset (569 patients)."""
    print("\n" + "█" * 70)
    print("  COMMON DISEASE: Breast Cancer (Real Wisconsin Dataset)")
    print("█" * 70)

    try:
        from models.common.breast_cancer import (
            load_breast_cancer_data, train_breast_cancer_model,
            get_dataset_info, predict_breast_cancer,
        )

        # Dataset info
        info = get_dataset_info()
        print(f"\n  Dataset: {info['name']}")
        print(f"  Source: {info['source']}")
        print(f"  Samples: {info['n_samples']}")
        print(f"  Features: {info['n_features']}")
        print(f"  Citation: {info['citation']}")

        # Load REAL data
        X, y, feature_names, target_names = load_breast_cancer_data()
        print(f"\n  Loaded {X.shape[0]} real patients, {X.shape[1]} features")
        print(f"  Classes: {dict(zip(target_names, np.bincount(y)))}")

        # Train with CV
        model, cv_results = train_breast_cancer_model(X, y)

        print(f"\n  ╔══ Results (Real Data) ══╗")
        print(f"  ║ Accuracy: {cv_results['accuracy']:.4f}       ║")
        print(f"  ║ Macro-F1: {cv_results['f1']:.4f}       ║")
        print(f"  ║ AUC-ROC:  {cv_results['auc_roc']:.4f}       ║")
        print(f"  ╚═════════════════════════╝")

        # Save model
        model_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'breast_cancer_model.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Model saved: {model_path}")

        return cv_results

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return None


def run_parkinsons_pipeline():
    """Run Parkinson's detection on real/statistical voice measurement data."""
    print("\n" + "█" * 70)
    print("  COMMON DISEASE: Parkinson's Disease (Voice Measurements)")
    print("█" * 70)

    try:
        from models.common.parkinsons import (
            load_parkinsons_data, train_parkinsons_model,
            get_dataset_info,
        )

        # Dataset info
        info = get_dataset_info()
        print(f"\n  Dataset: {info['name']}")
        print(f"  Source: {info['source']}")
        print(f"  Citation: {info['citation']}")

        # Load data
        X, y, feature_names = load_parkinsons_data()
        print(f"\n  Loaded {X.shape[0]} patients, {X.shape[1]} features")
        print(f"  Classes: healthy={np.sum(y==0)}, parkinsons={np.sum(y==1)}")

        # Train with CV
        model, cv_results = train_parkinsons_model(X, y)

        print(f"\n  ╔══ Results ══════════════╗")
        print(f"  ║ Accuracy: {cv_results['accuracy']:.4f}       ║")
        print(f"  ║ Macro-F1: {cv_results['f1']:.4f}       ║")
        print(f"  ║ AUC-ROC:  {cv_results['auc_roc']:.4f}       ║")
        print(f"  ╚═════════════════════════╝")

        # Save model
        model_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'parkinsons_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Model saved: {model_path}")

        return cv_results

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════
# RARE DISEASE PIPELINE (HPO-based + Quantum)
# ═══════════════════════════════════════════════════════════════════════

def phase1_generate_rare_disease_data(n_per_disease: int = 200, noise_rate: float = 0.10):
    """Phase 1: Generate synthetic patients from real HPO annotation frequencies."""
    print("\n" + "═" * 70)
    print("  RARE: Phase 1 — Synthetic Patient Generation (from real HPO frequencies)")
    print("═" * 70)

    from data.generate_patients import generate_synthetic_patients, tag_difficulty

    df = generate_synthetic_patients(
        n_per_disease=n_per_disease,
        noise_rate=noise_rate,
        seed=42,
    )
    df = tag_difficulty(df)

    n_hard = len(df[df['difficulty'] == 'hard'])
    n_easy = len(df[df['difficulty'] == 'easy'])
    print(f"\n  Difficulty: {n_hard} hard ({n_hard/len(df):.1%}), "
          f"{n_easy} easy ({n_easy/len(df):.1%})")

    # Save
    output_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'benchmark.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    return df


def phase2_classical_pipeline(df):
    """Phase 2: Classical ML pipeline for rare diseases."""
    print("\n" + "═" * 70)
    print("  RARE: Phase 2 — Classical ML Pipeline (XGBoost + Calibration)")
    print("═" * 70)

    from models.classical.features import compute_information_content, build_feature_matrix
    from models.classical.train_xgb import train_with_cv, get_feature_importance, save_model
    from models.classical.calibrate import calibrate_model, save_calibrated_model

    # IC computation
    print("\n--- Computing Information Content ---")
    ic_values = compute_information_content()

    ic_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ic_values.json')
    with open(ic_path, 'w') as f:
        json.dump(ic_values, f, indent=2)

    sorted_ic = sorted(ic_values.items(), key=lambda x: -x[1])[:10]
    print(f"  Top 10 most specific terms:")
    for term, ic in sorted_ic:
        label = HPO_TERMS.get(term, "Unknown")
        print(f"    {term} ({label}): IC = {ic:.4f}")

    # Feature matrix
    print("\n--- Building IC-Weighted Feature Matrix ---")
    X, y = build_feature_matrix(df, ic_values)
    print(f"  Shape: {X.shape}, Classes: {np.unique(y, return_counts=True)}")

    # Train XGBoost
    print("\n--- Training XGBoost (Stratified 5-Fold CV) ---")
    model, cv_results = train_with_cv(X, y, n_splits=5)

    model_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'xgb_model.pkl')
    save_model(model, model_path)

    # Feature importance
    importance_df = get_feature_importance(model, ALL_HPO_TERMS, top_k=10)
    print(f"\n  Top 10 features:")
    for _, row in importance_df.iterrows():
        print(f"    {row['rank']:>3d}. {row['hpo_id']} ({row['label']}): {row['importance']:.4f}")

    # Calibrate
    print("\n--- Calibrating Probabilities ---")
    calibrated_model, calibration_metrics = calibrate_model(model, X, y)

    cal_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'calibrated_model.pkl')
    save_calibrated_model(calibrated_model, cal_path)

    return X, y, model, calibrated_model, cv_results, calibration_metrics, ic_values


def phase3_confusion_detection(X, y, calibrated_model):
    """Phase 3: Confusion detection — flag hard rare disease cases for quantum."""
    print("\n" + "═" * 70)
    print("  RARE: Phase 3 — Confusion Detection")
    print("═" * 70)

    from models.confusion.detector import (
        is_hard_case, tune_thresholds,
        analyze_confusion_distribution, print_confusion_analysis,
    )

    all_probs = calibrated_model.predict_proba(X)

    # Tune thresholds
    margin_tau, entropy_tau = tune_thresholds(
        all_probs, DISEASE_NAMES, KNOWN_CONFUSION_PAIRS,
        target_hard_rate=0.20,
    )
    print(f"  Tuned: margin_tau={margin_tau:.3f}, entropy_tau={entropy_tau:.3f}")

    confusion_analysis = analyze_confusion_distribution(
        all_probs, y, DISEASE_NAMES, KNOWN_CONFUSION_PAIRS,
        margin_tau=margin_tau, entropy_tau=entropy_tau,
    )
    print_confusion_analysis(confusion_analysis)

    # Extract hard cases
    hard_indices = []
    for i in range(len(all_probs)):
        result = is_hard_case(
            all_probs[i], DISEASE_NAMES, KNOWN_CONFUSION_PAIRS,
            margin_tau=margin_tau, entropy_tau=entropy_tau,
        )
        if result.is_hard:
            hard_indices.append(i)

    X_hard = X[hard_indices]
    y_hard = y[hard_indices]
    print(f"\n  Hard cases: {len(hard_indices)}")

    return confusion_analysis, X_hard, y_hard, hard_indices


def phase4_quantum_resolver(X_hard, y_hard):
    """Phase 4: Quantum QSVM on hard cases + classical benchmark."""
    print("\n" + "═" * 70)
    print("  RARE: Phase 4 — Quantum Resolver (ZZFeatureMap + QSVM)")
    print("═" * 70)

    if len(X_hard) < 10:
        print(f"  Only {len(X_hard)} hard cases — too few for meaningful benchmark.")
        return None

    try:
        from models.quantum.feature_select import select_discriminative_features, normalize_for_quantum
        from models.quantum.zz_kernel import create_quantum_kernel, compute_kernel_matrices, validate_kernel_matrix
        from models.quantum.train_qsvm import train_quantum_svm, evaluate_qsvm
        from models.quantum.benchmark_classical_svm import benchmark_quantum_vs_classical
        from sklearn.model_selection import train_test_split

        unique_labels, label_counts = np.unique(y_hard, return_counts=True)
        print(f"  Labels in hard cases: {[LABEL_DISEASE_MAP[l] for l in unique_labels]}")
        for lbl, cnt in zip(unique_labels, label_counts):
            print(f"    {LABEL_DISEASE_MAP[lbl]}: {cnt} cases")

        if len(unique_labels) < 2:
            print(f"  Only one class — cannot train.")
            return None

        # Filter out classes with too few members for stratified split
        # Need at least 2 members per class for train_test_split(stratify=...)
        valid_mask = np.isin(y_hard, unique_labels[label_counts >= 2])
        if np.sum(valid_mask) < 10:
            print(f"  After filtering rare classes, only {np.sum(valid_mask)} cases remain — too few.")
            return None
        X_hard_filtered = X_hard[valid_mask]
        y_hard_filtered = y_hard[valid_mask]

        filtered_labels = np.unique(y_hard_filtered)
        if len(filtered_labels) < 2:
            print(f"  Only one class after filtering — cannot train.")
            return None

        # Feature selection
        print("\n--- Mutual Information Feature Selection ---")
        X_reduced, y_filtered, selected_idx = select_discriminative_features(
            X_hard_filtered, y_hard_filtered, top2_labels=filtered_labels.tolist(),
            k=min(8, X_hard_filtered.shape[1])
        )
        print(f"  Features: {X_hard_filtered.shape[1]} → {X_reduced.shape[1]}")

        for idx in selected_idx:
            if idx < len(ALL_HPO_TERMS):
                term = ALL_HPO_TERMS[idx]
                print(f"    Qubit {selected_idx.index(idx)}: {term} ({HPO_TERMS.get(term, '?')})")

        # Normalize for quantum encoding: map features to [0, π]
        X_norm = normalize_for_quantum(X_reduced)

        # Split — use stratification only if all classes have enough members
        test_size = max(2, len(X_norm) // 5)
        _, split_counts = np.unique(y_filtered, return_counts=True)
        use_stratify = np.all(split_counts >= 2)

        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y_filtered, test_size=test_size,
            random_state=42, stratify=y_filtered if use_stratify else None
        )
        print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")

        # Quantum kernel computation
        print("\n--- Quantum Kernel Computation ---")
        n_qubits = X_train.shape[1]
        kernel = create_quantum_kernel(n_features=n_qubits)

        t0 = time.time()
        K_train, K_test = compute_kernel_matrices(kernel, X_train, X_test)
        kernel_time = time.time() - t0
        print(f"  Kernel computation time: {kernel_time:.2f}s")

        # Validate kernel matrix (Mercer's conditions)
        val = validate_kernel_matrix(K_train)
        print(f"  Kernel validation: symmetric={val['is_symmetric']}, PSD={val['is_psd']}, "
              f"diagonal≈1.0={val['is_normalized']}")

        # QSVM training
        print("\n--- Quantum SVM ---")
        qsvm = train_quantum_svm(K_train, y_train)
        q_metrics = evaluate_qsvm(qsvm, K_test, y_test)
        print(f"  QSVM Accuracy: {q_metrics['accuracy']:.4f}")
        print(f"  QSVM Macro-F1: {q_metrics['macro_f1']:.4f}")

        # Classical benchmark on same data/split
        print("\n--- Classical RBF-SVM (Same Data/Split) ---")
        quantum_results = {
            'qsvm': qsvm,
            'predictions': qsvm.predict(K_test),
            'accuracy': q_metrics['accuracy'],
            'f1': q_metrics['macro_f1'],
        }
        benchmark = benchmark_quantum_vs_classical(
            quantum_results, X_train, X_test, y_train, y_test
        )
        print(f"  Classical Accuracy: {benchmark['classical_accuracy']:.4f}")
        print(f"  Classical Macro-F1: {benchmark['classical_macro_f1']:.4f}")
        print(f"  McNemar p-value:   {benchmark.get('p_value', 'N/A')}")
        print(f"  Conclusion:        {benchmark.get('conclusion', 'N/A')}")

        # Add quantum F1 to benchmark for summary
        benchmark['quantum_f1'] = q_metrics['macro_f1']
        benchmark['quantum_accuracy'] = q_metrics['accuracy']
        benchmark['mcnemar_p_value'] = benchmark.get('p_value', 1.0)

        # ERR-04 / ACTION B: Persist QSVM model and support data to disk
        # so the backend can load them at startup instead of retraining live.
        qsvm_artifact = {
            'qsvm_model': qsvm,
            'X_train': X_train,
            'y_train': y_train,
            'selected_indices': selected_idx,
            'n_qubits': n_qubits,
            'X_min': np.min(X_reduced, axis=0),
            'X_max': np.max(X_reduced, axis=0),
        }
        qsvm_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'qsvm_model.pkl')
        with open(qsvm_path, 'wb') as f:
            pickle.dump(qsvm_artifact, f)
        print(f"\n  ✓ Saved QSVM model artifact to {qsvm_path}")
        print(f"    Contains: model, {len(X_train)} support vectors, "
              f"{len(selected_idx)} selected features, normalization params")

        return benchmark

    except ImportError as e:
        print(f"  Quantum deps missing: {e}")
        print(f"  pip install qiskit qiskit-machine-learning qiskit-algorithms qiskit-aer")
        return None
    except Exception as e:
        print(f"  Quantum phase error: {e}")
        traceback.print_exc()
        return None


def phase5_explainability(model, X, y, df, ic_values):
    """Phase 5: SHAP explanations + next-test recommendations."""
    print("\n" + "═" * 70)
    print("  RARE: Phase 5 — Explainability")
    print("═" * 70)

    # Pick a hard case
    hard_mask = df['difficulty'] == 'hard'
    sample_idx = df[hard_mask].index[0] if hard_mask.any() else 0
    sample_row = df.iloc[sample_idx]
    sample_terms = sample_row['hpo_terms'].split('|') if isinstance(sample_row['hpo_terms'], str) else []

    probs = model.predict_proba(X[sample_idx:sample_idx+1])[0]
    sorted_idx = np.argsort(probs)[::-1]
    top = DISEASE_NAMES[sorted_idx[0]]
    runner = DISEASE_NAMES[sorted_idx[1]]

    print(f"\n  Demo patient: {sample_row.get('patient_id', f'Patient {sample_idx}')}")
    print(f"  True: {sample_row['true_disease']}")
    print(f"  Predicted: {top} ({probs[sorted_idx[0]]:.3f}) vs {runner} ({probs[sorted_idx[1]]:.3f})")

    try:
        from explain.next_test_recommender import recommend_next_test, format_recommendation
        recs = recommend_next_test(probs, DISEASE_NAMES, sample_terms, top_k=5)
        print(format_recommendation(recs))
    except Exception as e:
        print(f"  Next-test skipped: {e}")


def phase6_report(cv_results, cal_metrics, confusion_analysis, quantum_benchmark,
                  bc_results=None, pk_results=None):
    """Phase 6: Generate comprehensive benchmark report."""
    print("\n" + "═" * 70)
    print("  Phase 6 — Benchmark Report")
    print("═" * 70)

    from benchmarks.report_generator import generate_report, save_report, save_metrics_json

    report_text = generate_report(
        cv_results, cal_metrics, confusion_analysis, quantum_benchmark
    )

    # Add common disease results to the report
    common_section = "\n## 6. Common Disease Classifiers (Real Data)\n\n"
    if bc_results:
        common_section += f"### Breast Cancer (Wisconsin Dataset — 569 real patients)\n"
        common_section += f"- **Accuracy:** {bc_results['accuracy']:.4f}\n"
        common_section += f"- **Macro-F1:** {bc_results['f1']:.4f}\n"
        common_section += f"- **AUC-ROC:** {bc_results['auc_roc']:.4f}\n"
        common_section += f"- **Data:** REAL (sklearn.datasets.load_breast_cancer)\n\n"
    if pk_results:
        common_section += f"### Parkinson's Disease (Voice Measurements)\n"
        common_section += f"- **Accuracy:** {pk_results['accuracy']:.4f}\n"
        common_section += f"- **Macro-F1:** {pk_results['f1']:.4f}\n"
        common_section += f"- **AUC-ROC:** {pk_results['auc_roc']:.4f}\n\n"

    report_text += common_section

    output_dir = os.path.join(PROJECT_ROOT, 'benchmarks')
    save_report(report_text, output_dir)
    save_metrics_json(cv_results, cal_metrics, confusion_analysis, quantum_benchmark, output_dir)

    print(f"\n{report_text[:2000]}")
    if len(report_text) > 2000:
        print(f"\n  ... (full report in benchmarks/benchmark_report.md)")


def main():
    """Run the full QResolve-Dx two-tier pipeline."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " QResolve-Dx: Quantum-Enhanced Differential Diagnosis".center(68) + "║")
    print("║" + "".center(68) + "║")
    print("║" + " Common Diseases: Classical ML on REAL datasets".center(68) + "║")
    print("║" + " Rare Diseases:   Classical + Quantum QSVM".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  Common diseases: Breast Cancer (real), Parkinson's (real)")
    print(f"  Rare diseases:   {', '.join(RARE_DISEASE_NAMES)}")
    print(f"  Full cluster:    {', '.join(DISEASE_NAMES)}")
    print(f"  HPO features:    {len(ALL_HPO_TERMS)}")

    t_start = time.time()
    ensure_dirs()

    # ═══ COMMON DISEASES ═══
    print("\n\n" + "▓" * 70)
    print("▓▓▓  TIER 1: COMMON DISEASES (Real Datasets, Classical ML)  ▓▓▓")
    print("▓" * 70)

    bc_results = run_breast_cancer_pipeline()
    pk_results = run_parkinsons_pipeline()

    # ═══ RARE DISEASES ═══
    print("\n\n" + "▓" * 70)
    print("▓▓▓  TIER 2: RARE DISEASES (HPO Data, Classical + Quantum)  ▓▓▓")
    print("▓" * 70)

    # Phase 1: Generate data
    try:
        df = phase1_generate_rare_disease_data(n_per_disease=200, noise_rate=0.10)
    except Exception as e:
        print(f"  Phase 1 FAILED: {e}")
        traceback.print_exc()
        return

    # Phase 2: Classical pipeline
    try:
        X, y, model, calibrated_model, cv_results, cal_metrics, ic_values = \
            phase2_classical_pipeline(df)
    except Exception as e:
        print(f"  Phase 2 FAILED: {e}")
        traceback.print_exc()
        return

    # Phase 3: Confusion detection
    try:
        confusion_analysis, X_hard, y_hard, hard_indices = \
            phase3_confusion_detection(X, y, calibrated_model)
    except Exception as e:
        print(f"  Phase 3 FAILED: {e}")
        traceback.print_exc()
        confusion_analysis = {"n_total": len(y), "n_hard": 0, "hard_rate": 0}
        X_hard, y_hard = np.array([]), np.array([])

    # Phase 4: Quantum resolver
    quantum_benchmark = None
    try:
        quantum_benchmark = phase4_quantum_resolver(X_hard, y_hard)
    except Exception as e:
        print(f"  Phase 4 FAILED (non-fatal): {e}")

    # Phase 5: Explainability
    try:
        phase5_explainability(model, X, y, df, ic_values)
    except Exception as e:
        print(f"  Phase 5 FAILED (non-fatal): {e}")

    # Phase 6: Report
    try:
        phase6_report(cv_results, cal_metrics, confusion_analysis, quantum_benchmark,
                      bc_results, pk_results)
    except Exception as e:
        print(f"  Phase 6 FAILED (non-fatal): {e}")
        traceback.print_exc()

    # Summary
    t_total = time.time() - t_start
    print("\n" + "═" * 70)
    print("  PIPELINE COMPLETE")
    print("═" * 70)
    print(f"  Total time: {t_total:.1f}s")
    print(f"  Common diseases:")
    if bc_results:
        print(f"    ✓ Breast Cancer — Acc: {bc_results['accuracy']:.4f}, F1: {bc_results['f1']:.4f} (REAL data)")
    else:
        print(f"    ✗ Breast Cancer — failed")
    if pk_results:
        print(f"    ✓ Parkinson's  — Acc: {pk_results['accuracy']:.4f}, F1: {pk_results['f1']:.4f}")
    else:
        print(f"    ✗ Parkinson's  — failed")
    print(f"  Rare diseases:")
    print(f"    ✓ Classical XGBoost — F1: {cv_results.get('overall_macro_f1', 0):.4f}")
    print(f"    ✓ Confusion detector — {confusion_analysis.get('hard_rate', 0):.1%} hard")
    if quantum_benchmark:
        print(f"    ✓ Quantum QSVM — F1: {quantum_benchmark.get('quantum_f1', 0):.4f}")
        print(f"    ✓ Classical SVM — F1: {quantum_benchmark.get('classical_macro_f1', 0):.4f}")
        print(f"    ✓ McNemar's p-value: {quantum_benchmark.get('mcnemar_p_value', 'N/A')}")
        print(f"    ✓ Conclusion: {quantum_benchmark.get('conclusion', 'N/A')}")
    else:
        print(f"    ✗ Quantum QSVM — failed (check errors above)")
    print(f"\n  Reports: benchmarks/benchmark_report.md")
    print("═" * 70)


if __name__ == "__main__":
    main()
