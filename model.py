"""
================================================================================
Connected Vehicle Telemetry & Predictive Health Analytics System
Module 3: Automotive Connectivity & AI/ML Prognostics
================================================================================
Description:
    Real-time in-vehicle telemetry stream processing, predictive degradation
    modeling, multi-component health score estimation, and Explainable AI (XAI)
    root-cause diagnostics for connected powertrain and chassis systems.

Compliant with:
    - ISO 13374 (Condition Monitoring and Diagnostics of Machine Systems)
    - ISO 26262 (Functional Safety for Road Vehicles)
    - OBD-II / UDS (ISO 14229) Automotive Diagnostic Protocols
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def generate_telemetry_data(n_samples: int = 6000, random_seed: int = 42) -> pd.DataFrame:
    """
    Synthesize realistic connected vehicle telemetry streams.
    Includes nominal operating dynamics, engine load coupling, and multi-modal
    physical degradation modes (thermal stress, lubrication failure, mechanical vibration).
    """
    np.random.seed(random_seed)
    
    # 1. Operational duty cycle baseline
    operating_hours = np.random.uniform(100, 5000, n_samples)
    vehicle_speed = np.random.uniform(0, 140, n_samples)
    rpm = np.clip(
        vehicle_speed * np.random.uniform(25, 45, n_samples) + np.random.normal(800, 80, n_samples),
        700, 6500
    )
    
    # 2. Thermodynamic & Mechanical physical baselines
    engine_temp = 80 + (rpm / 6500) * 15 + (operating_hours / 5000) * 6 + np.random.normal(0, 2.0, n_samples)
    coolant_temp = engine_temp - np.random.uniform(5, 10, n_samples)
    
    oil_pressure = 44 - (engine_temp / 110) * 8 - (operating_hours / 5000) * 5 + np.random.normal(0, 2.0, n_samples)
    oil_pressure = np.clip(oil_pressure, 12, 60)
    
    vibration = 1.3 + (rpm / 6500) * 1.6 + (operating_hours / 5000) * 1.2 + np.random.normal(0, 0.25, n_samples)
    battery_voltage = 14.2 - (operating_hours / 5000) * 1.2 + np.random.normal(0, 0.2, n_samples)
    fuel_consumption = np.clip(0.05 * vehicle_speed + (rpm / 1000) * 1.1 + np.random.normal(0, 0.3, n_samples), 0.5, 25.0)
    
    # 3. Multi-modal automotive failure taxonomy injection:
    # 0 = Nominal (86%), 1 = Mechanical Vibration (4.5%), 2 = Thermal Stress (4.0%),
    # 3 = Lubrication Starvation (3.5%), 4 = Incipient Multi-component Wear (2.0%)
    mode_assign = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.86, 0.045, 0.040, 0.035, 0.020])
    
    # Mode 1: Mechanical bearing / drivetrain fault
    vibration += (mode_assign == 1) * np.random.uniform(1.8, 3.5, n_samples)
    
    # Mode 2: Thermal runaway / cooling circuit breakdown
    engine_temp += (mode_assign == 2) * np.random.uniform(18, 35, n_samples)
    coolant_temp += (mode_assign == 2) * np.random.uniform(15, 30, n_samples)
    
    # Mode 3: Lubrication pressure loss / pump cavitation
    oil_pressure -= (mode_assign == 3) * np.random.uniform(16, 26, n_samples)
    oil_pressure = np.clip(oil_pressure, 6, 60)
    
    # Mode 4: Incipient multi-component degradation (early warning phase)
    vibration += (mode_assign == 4) * np.random.uniform(0.9, 1.6, n_samples)
    engine_temp += (mode_assign == 4) * np.random.uniform(8, 16, n_samples)
    coolant_temp += (mode_assign == 4) * np.random.uniform(7, 14, n_samples)
    oil_pressure -= (mode_assign == 4) * np.random.uniform(7, 13, n_samples)
    
    failure_label = (mode_assign > 0).astype(int)
    
    df = pd.DataFrame({
        "engine_temperature": np.round(engine_temp, 2),
        "rpm": np.round(rpm, 1),
        "oil_pressure": np.round(oil_pressure, 2),
        "vibration": np.round(vibration, 3),
        "battery_voltage": np.round(battery_voltage, 2),
        "coolant_temperature": np.round(coolant_temp, 2),
        "fuel_consumption": np.round(fuel_consumption, 2),
        "vehicle_speed": np.round(vehicle_speed, 2),
        "operating_hours": np.round(operating_hours, 1),
        "failure_target": failure_label
    })
    return df

class VehicleHealthDiagnosticSystem:
    """
    Production-grade Connected Vehicle Health Diagnostic & Prognostic System.
    Integrates supervised gradient boosting classification, ISO 13374 Condition
    Monitoring Composite Health Index calculation, and Local XAI root-cause attribution.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            random_state=42
        )
        self.feature_names = [
            "engine_temperature", "rpm", "oil_pressure", "vibration",
            "battery_voltage", "coolant_temperature", "fuel_consumption",
            "vehicle_speed", "operating_hours"
        ]
        self.feature_labels = {
            "vibration": "High vibration",
            "engine_temperature": "Rising temperature",
            "oil_pressure": "Abnormal oil pressure",
            "coolant_temperature": "High coolant temperature",
            "battery_voltage": "Low battery voltage",
            "fuel_consumption": "Elevated fuel consumption",
            "operating_hours": "Extended operational duty cycle",
            "rpm": "Engine speed instability (RPM)",
            "vehicle_speed": "Vehicle speed variance"
        }
        self.baseline_means = None
        self.baseline_stds = None

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train predictive model on scaled telemetry and compute validation metrics.
        """
        X = df[self.feature_names]
        y = df["failure_target"]
        
        self.baseline_means = X.mean()
        self.baseline_stds = X.std()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        # Fit scaler on DataFrame to preserve feature metadata and avoid warnings
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.feature_names)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=self.feature_names)
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "specificity": specificity,
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        return metrics

    def compute_composite_health_index(self, raw_instance: pd.Series) -> float:
        """
        ISO 13374 Condition Monitoring Composite Health Index (CHI).
        Quantifies remaining operational margin (0% = critical failure, 100% = pristine nominal).
        Evaluates physical degradation margins across key automotive subsystems:
        - Mechanical / Structural (Vibration)
        - Thermodynamic (Engine & Coolant Temperature)
        - Lubrication (Oil Pressure)
        - Electrical & Duty Cycle (Battery Voltage & Operating Hours)
        """
        vib_deg = max(0.0, (raw_instance["vibration"] - 2.0) / 3.0)
        temp_deg = max(0.0, (raw_instance["engine_temperature"] - 88.0) / 28.0)
        oil_deg = max(0.0, (40.0 - raw_instance["oil_pressure"]) / 25.0)
        volt_deg = max(0.0, (13.5 - raw_instance["battery_voltage"]) / 2.0)
        hours_deg = max(0.0, (raw_instance["operating_hours"] - 2500.0) / 2500.0)
        
        weighted_penalty = (
            0.35 * vib_deg +
            0.32 * temp_deg +
            0.25 * oil_deg +
            0.04 * volt_deg +
            0.04 * hours_deg
        )
        health_score = max(0.0, min(100.0, (1.0 - weighted_penalty) * 100.0))
        return health_score

    def compute_local_xai_importance(self, raw_instance: pd.Series, top_k: int = 3):
        """
        Directional Local XAI Attribution.
        Attributes risk to physical parameters moving towards hazardous failure boundaries.
        Multiplies the positive hazard deviation by the model's global Gini/Gain importance.
        """
        df_row = pd.DataFrame([raw_instance[self.feature_names]])
        instance_scaled = self.scaler.transform(df_row)[0]
        global_importance = self.model.feature_importances_
        
        directions = {
            "vibration": 1.0,
            "engine_temperature": 1.0,
            "coolant_temperature": 1.0,
            "oil_pressure": -1.0,
            "battery_voltage": -1.0,
            "fuel_consumption": 1.0,
            "operating_hours": 1.0,
            "rpm": 1.0,
            "vehicle_speed": 0.0
        }
        
        attributions = {}
        for idx, feat in enumerate(self.feature_names):
            z = instance_scaled[idx]
            dir_mult = directions.get(feat, 1.0)
            hazard_deviation = max(0.0, z * dir_mult)
            attributions[feat] = hazard_deviation * global_importance[idx]
            
        ranked = sorted(attributions.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_factors = [self.feature_labels.get(k, k) for k, _ in ranked]
        return top_factors

    def diagnose_vehicle(self, telemetry_record: dict) -> dict:
        """
        Execute in-vehicle health diagnostic on a single telemetry record.
        Outputs exact format specified in Module 3 Project specifications:
            Vehicle Health: XX%
            Failure Probability: XX%
            Status: NORMAL | WARNING | CRITICAL
            Major contributing factors: [1, 2, 3]
        """
        row = pd.Series(telemetry_record)
        df_row = pd.DataFrame([row[self.feature_names]])
        scaled_features = pd.DataFrame(self.scaler.transform(df_row), columns=self.feature_names)
        
        raw_prob = float(self.model.predict_proba(scaled_features)[0, 1])
        health_score = self.compute_composite_health_index(row)
        
        # When failure probability is low but health index indicates incipient wear,
        # calibrate failure risk to reflect developing fault hazard:
        effective_failure_prob = max(raw_prob, (100.0 - health_score) / 100.0 * 0.95) if health_score < 80.0 else raw_prob
        
        # Automotive Alert Thresholds
        if effective_failure_prob >= 0.75 or health_score < 40.0:
            status = "CRITICAL"
        elif effective_failure_prob >= 0.35 or health_score < 80.0:
            status = "WARNING"
        else:
            status = "NORMAL"
            
        contributing_factors = self.compute_local_xai_importance(row, top_k=3)
        
        return {
            "Vehicle Health": f"{round(health_score)}%",
            "Failure Probability": f"{round(effective_failure_prob * 100)}%",
            "Status": status,
            "Major contributing factors": contributing_factors
        }

if __name__ == "__main__":
    print("=" * 70)
    print("CONNECTED VEHICLE TELEMETRY PREDICTIVE HEALTH ANALYTICS SYSTEM")
    print("=" * 70)
    print("1. Generating synthetic connected vehicle telemetry dataset...")
    dataset = generate_telemetry_data(n_samples=6000, random_seed=42)
    print(f"Generated {len(dataset)} records.")
    dist = dataset['failure_target'].value_counts(normalize=True).to_dict()
    print(f"Class distribution: Nominal: {dist[0]*100:.1f}%, Failure: {dist[1]*100:.1f}%\n")
    
    diagnostic_system = VehicleHealthDiagnosticSystem()
    metrics = diagnostic_system.train(dataset)
    
    print("=" * 70)
    print("MODEL VALIDATION METRICS (Test Set = 1,200 records)")
    print("=" * 70)
    print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:    {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:       {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"Specificity:  {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"F1-Score:     {metrics['f1']:.4f}")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("=" * 70)
    
    # Diagnostic Edge Cases & Field Scenarios
    test_cases = [
        {
            "name": "Scenario 1: Imminent Severe Breakdown",
            "description": "Acute thermal stress and bearing seizure under extreme operating load.",
            "telemetry": {
                "engine_temperature": 114.5,
                "rpm": 4650.0,
                "oil_pressure": 15.2,
                "vibration": 5.42,
                "battery_voltage": 12.1,
                "coolant_temperature": 105.8,
                "fuel_consumption": 14.8,
                "vehicle_speed": 110.0,
                "operating_hours": 4350.0
            }
        },
        {
            "name": "Scenario 2: Incipient Multi-component Wear (Module 3 Sample Output)",
            "description": "Early stage mechanical bearing friction, rising thermal trend, and declining oil pressure.",
            "telemetry": {
                "engine_temperature": 98.0,
                "rpm": 3100.0,
                "oil_pressure": 28.0,
                "vibration": 3.60,
                "battery_voltage": 13.4,
                "coolant_temperature": 90.0,
                "fuel_consumption": 8.2,
                "vehicle_speed": 72.0,
                "operating_hours": 2200.0
            }
        },
        {
            "name": "Scenario 3: Nominal Cruising Operating Profile",
            "description": "Standard expressway driving within optimal thermodynamic and vibrational margins.",
            "telemetry": {
                "engine_temperature": 86.4,
                "rpm": 2100.0,
                "oil_pressure": 38.6,
                "vibration": 1.72,
                "battery_voltage": 13.9,
                "coolant_temperature": 79.5,
                "fuel_consumption": 5.8,
                "vehicle_speed": 55.0,
                "operating_hours": 950.0
            }
        }
    ]
    
    print("\n" + "=" * 70)
    print("FIELD DIAGNOSTIC INFERENCE REPORTS")
    print("=" * 70)
    for case in test_cases:
        diag = diagnostic_system.diagnose_vehicle(case["telemetry"])
        print(f"\n--- {case['name']} ---")
        print(f"Vehicle Health: {diag['Vehicle Health']}")
        print(f"Failure Probability: {diag['Failure Probability']}")
        print(f"Status: {diag['Status']}")
        print("Major contributing factors:")
        for i, factor in enumerate(diag['Major contributing factors'], 1):
            print(f"  {i}. {factor}")
    print("\n" + "=" * 70)