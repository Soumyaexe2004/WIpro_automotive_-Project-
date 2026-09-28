"""
================================================================================
Comprehensive Multi-Model Benchmarking & Diagnostics Analytics Suite
Module 3: Connected Vehicle Telemetry & Health Analytics
================================================================================
"""

import os
import shutil
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import xgboost as xgb

from model import generate_telemetry_data, VehicleHealthDiagnosticSystem

ARTIFACT_DIR = r"C:\Users\csoum\.gemini\antigravity-ide\brain\2922d189-6a59-44ae-82a9-33cad4f6989f"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("=" * 70)
print("1. GENERATING TELEMETRY DATASET (N = 6,000)")
print("=" * 70)
df = generate_telemetry_data(n_samples=6000, random_seed=42)
df.to_csv("telemetry_dataset_sample.csv", index=False)
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['failure_target'].value_counts(normalize=True).to_dict()}\n")

feature_names = [
    "engine_temperature", "rpm", "oil_pressure", "vibration",
    "battery_voltage", "coolant_temperature", "fuel_consumption",
    "vehicle_speed", "operating_hours"
]

feature_display_names = {
    "engine_temperature": "Engine Temp (°C)",
    "rpm": "Engine RPM",
    "oil_pressure": "Oil Pressure (psi)",
    "vibration": "Vibration (mm/s)",
    "battery_voltage": "Battery Voltage (V)",
    "coolant_temperature": "Coolant Temp (°C)",
    "fuel_consumption": "Fuel Cons. (L/100km)",
    "vehicle_speed": "Vehicle Speed (km/h)",
    "operating_hours": "Operating Hours (h)"
}

X = df[feature_names]
y = df["failure_target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

print("=" * 70)
print("2. MULTI-MODEL BENCHMARKING (6 ARCHITECTURES)")
print("=" * 70)

candidate_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42, eval_metric="logloss"),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

benchmark_records = []
trained_models = {}
roc_dict = {}
pr_dict = {}
cm_dict = {}

for name, model in candidate_models.items():
    # Training latency
    t0_train = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    t_train = time.perf_counter() - t0_train
    
    # Inference latency
    t0_infer = time.perf_counter()
    y_pred = model.predict(X_test_scaled)
    t_infer_total = time.perf_counter() - t0_infer
    t_infer_per_sample = (t_infer_total / len(X_test)) * 1000 # ms
    
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_prob)
    
    trained_models[name] = model
    roc_dict[name] = (fpr, tpr, auc)
    pr_dict[name] = (recall_pts, precision_pts)
    cm_dict[name] = cm
    
    benchmark_records.append({
        "Model": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Precision (%)": round(prec * 100, 2),
        "Recall (%)": round(rec * 100, 2),
        "Specificity (%)": round(spec * 100, 2),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Train Time (s)": round(t_train, 4),
        "Inference Latency (ms/sample)": round(t_infer_per_sample, 5)
    })
    print(f" -> {name:20s} | Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}")

bench_df = pd.DataFrame(benchmark_records)
bench_df.to_csv("model_evaluation_metrics.csv", index=False)
print("\n" + "=" * 70)
print("BENCHMARK COMPARISON TABLE")
print("=" * 70)
print(bench_df.to_string(index=False))

# 3. GENERATE VISUALIZATIONS
print("\n" + "=" * 70)
print("3. GENERATING HIGH-RESOLUTION PLOTS")
print("=" * 70)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
x_pos = np.arange(len(bench_df))
width = 0.22
r1 = ax.bar(x_pos - 1.5*width, bench_df["Accuracy (%)"], width, label="Accuracy (%)", color="#1E40AF")
r2 = ax.bar(x_pos - 0.5*width, bench_df["Precision (%)"], width, label="Precision (%)", color="#059669")
r3 = ax.bar(x_pos + 0.5*width, bench_df["Recall (%)"], width, label="Recall (%)", color="#D97706")
r4 = ax.bar(x_pos + 1.5*width, bench_df["Specificity (%)"], width, label="Specificity (%)", color="#7C3AED")

ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
ax.set_title("Connected Vehicle Prognostics: Multi-Model Benchmark Comparison", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(bench_df["Model"], rotation=15, ha="right", fontsize=10, fontweight="bold")
ax.set_ylim([90, 101])
ax.legend(loc="lower right", frameon=True, fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, "model_comparison.png")
plt.savefig(fig1_path)
plt.close()
print("Saved:", fig1_path)

# Figure 2: ROC Curves
fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
colors = ["#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED", "#475569"]
for idx, (name, (fpr, tpr, auc_val)) in enumerate(roc_dict.items()):
    ax.plot(fpr, tpr, lw=2.2, color=colors[idx % len(colors)], label=f"{name} (AUC = {auc_val:.4f})")
ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
ax.set_xlim([-0.01, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
ax.set_title("Receiver Operating Characteristic (ROC) Multi-Model Curves", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right", frameon=True, fontsize=9.5)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, "roc_curves.png")
plt.savefig(fig2_path)
plt.close()
print("Saved:", fig2_path)

# Figure 3: Confusion Matrices (2x3 grid)
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=300)
axes = axes.flatten()
for i, (name, cm) in enumerate(cm_dict.items()):
    ax = axes[i]
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    for (r, c), val in np.ndenumerate(cm):
        text_color = "white" if val > cm.max()/2 else "black"
        ax.text(c, r, f"{val:,}", ha="center", va="center", color=text_color, fontsize=13, fontweight="bold")
    ax.set_title(name, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Condition", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Ground Truth Condition", fontsize=9.5, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Nominal (0)", "Failure (1)"], fontsize=9)
    ax.set_yticklabels(["Nominal (0)", "Failure (1)"], fontsize=9)
plt.suptitle("Confusion Matrix Grid across Evaluated Classifiers (Test N = 1,200)", fontsize=14, fontweight="bold", y=1.00)
plt.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
plt.savefig(fig3_path)
plt.close()
print("Saved:", fig3_path)

# Figure 4: Global Feature Importance
rf_imp = trained_models["Random Forest"].feature_importances_
gb_imp = trained_models["Gradient Boosting"].feature_importances_
xgb_imp = trained_models["XGBoost"].feature_importances_

feat_df = pd.DataFrame({
    "Feature": [feature_display_names[f] for f in feature_names],
    "Random Forest": rf_imp,
    "Gradient Boosting": gb_imp,
    "XGBoost": xgb_imp
}).sort_values(by="Gradient Boosting", ascending=True)

fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)
y_pos = np.arange(len(feat_df))
h = 0.26
ax.barh(y_pos - h, feat_df["Random Forest"], h, label="Random Forest", color="#2563EB")
ax.barh(y_pos, feat_df["Gradient Boosting"], h, label="Gradient Boosting", color="#059669")
ax.barh(y_pos + h, feat_df["XGBoost"], h, label="XGBoost", color="#7C3AED")

ax.set_yticks(y_pos)
ax.set_yticklabels(feat_df["Feature"], fontsize=10, fontweight="bold")
ax.set_xlabel("Relative Predictive Importance (Gini / Gain)", fontsize=11, fontweight="bold")
ax.set_title("Global Feature Importance Comparison across Tree Ensembles", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right", frameon=True, fontsize=10)
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, "feature_importance.png")
plt.savefig(fig4_path)
plt.close()
print("Saved:", fig4_path)

# Figure 5: Telemetry Correlation Matrix
corr = df[feature_names + ["failure_target"]].corr()
fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=300)
cax = ax.matshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
fig.colorbar(cax, fraction=0.046, pad=0.04)
ticks = np.arange(len(corr.columns))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
disp_labels = [feature_display_names.get(c, "Failure Target") for c in corr.columns]
ax.set_xticklabels(disp_labels, rotation=45, ha="left", fontsize=9, fontweight="bold")
ax.set_yticklabels(disp_labels, fontsize=9, fontweight="bold")
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.55 else "black", fontsize=8.5)
ax.set_title("Pearson Correlation Matrix: Connected Powertrain Telemetry Streams", fontsize=12, fontweight="bold", pad=24)
plt.tight_layout()
fig5_path = os.path.join(FIGURES_DIR, "telemetry_correlation_matrix.png")
plt.savefig(fig5_path)
plt.close()
print("Saved:", fig5_path)

# Figure 6: Local XAI Scenario Breakdown
scenarios = [
    {
        "name": "Scenario 1: Critical Thermal & Mechanical",
        "telemetry": {
            "engine_temperature": 114.5, "rpm": 4650.0, "oil_pressure": 15.2,
            "vibration": 5.42, "battery_voltage": 12.1, "coolant_temperature": 105.8,
            "fuel_consumption": 14.8, "vehicle_speed": 110.0, "operating_hours": 4350.0
        }
    },
    {
        "name": "Scenario 2: Incipient Wear (Warning Profile)",
        "telemetry": {
            "engine_temperature": 98.0, "rpm": 3100.0, "oil_pressure": 28.0,
            "vibration": 3.60, "battery_voltage": 13.4, "coolant_temperature": 90.0,
            "fuel_consumption": 8.2, "vehicle_speed": 72.0, "operating_hours": 2200.0
        }
    },
    {
        "name": "Scenario 3: Nominal Cruising",
        "telemetry": {
            "engine_temperature": 86.4, "rpm": 2100.0, "oil_pressure": 38.6,
            "vibration": 1.72, "battery_voltage": 13.9, "coolant_temperature": 79.5,
            "fuel_consumption": 5.8, "vehicle_speed": 55.0, "operating_hours": 950.0
        }
    }
]

gb_model = trained_models["Gradient Boosting"]
diag_system = VehicleHealthDiagnosticSystem()
diag_system.train(df)

scenario_eval_results = []
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

for idx, sc in enumerate(scenarios):
    diag = diag_system.diagnose_vehicle(sc["telemetry"])
    scenario_eval_results.append({
        "scenario": sc["name"],
        "telemetry": sc["telemetry"],
        "diagnosis": diag
    })
    
    # Compute z-score anomalies for plot
    df_row = pd.DataFrame([pd.Series(sc["telemetry"])[feature_names]])
    scaled_z = scaler.transform(df_row)[0]
    
    ax = axes[idx]
    y_indices = np.arange(len(feature_names))
    colors_z = ["#DC2626" if abs(z) > 2.0 else "#D97706" if abs(z) > 1.0 else "#059669" for z in scaled_z]
    ax.barh(y_indices, scaled_z, color=colors_z, alpha=0.85)
    ax.axvline(0, color="black", linestyle="-", lw=1)
    ax.axvline(2.0, color="#DC2626", linestyle="--", lw=1, alpha=0.7, label="+2σ (Alarm)")
    ax.axvline(-2.0, color="#DC2626", linestyle="--", lw=1, alpha=0.7)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([feature_display_names[f] for f in feature_names], fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Normalized Deviation (z-score)", fontsize=9.5, fontweight="bold")
    ax.set_title(f"{sc['name']}\nHealth: {diag['Vehicle Health']} | Prob: {diag['Failure Probability']} | {diag['Status']}",
                 fontsize=10, fontweight="bold", pad=8)
    ax.set_xlim([-3.5, 4.5])
    ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.suptitle("Local XAI Anomaly Breakdown across Field Diagnostic Scenarios", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig6_path = os.path.join(FIGURES_DIR, "xai_scenario_breakdown.png")
plt.savefig(fig6_path)
plt.close()
print("Saved:", fig6_path)

with open("diagnostic_results.json", "w") as f:
    json.dump(scenario_eval_results, f, indent=2)
print("Saved: diagnostic_results.json")

# Copy all figures to the artifact directory for embedding
for fname in os.listdir(FIGURES_DIR):
    if fname.endswith(".png"):
        src = os.path.join(FIGURES_DIR, fname)
        dst = os.path.join(ARTIFACT_DIR, fname)
        shutil.copy2(src, dst)
        print(f"Copied figure to artifact directory: {dst}")

print("\nAll benchmark artifacts and figures generated successfully!")
