"""
F1 Driver Style Classifier
==========================

Classifies F1 drivers into 3 performance categories using supervised machine
learning with minimal seed labels.

Categories:
1. AGGRESSIVE: High risk scores, pushing limits, penalties and crashes
2. CONSISTENT: Solid performance, moderate risk, reliable points
3. STRUGGLING: Poor results, high incidents, or off-pace vs teammate

Features (7 total):
- Risk Score: Penalties, crashes, collisions
- Teammate Performance: Points/qualifying/position delta vs teammate
- Race Performance: Consistency, position change, tyre degradation

Approach:
- Uses 8 seed labels (38% of 21 drivers) to train Random Forest classifier
- Predicts remaining 13 drivers with confidence scores
- PointsDelta is most important feature (27.8% importance)

Design Principles:
- Minimal pre-training to let model learn patterns
- Teammate comparison normalises for car performance
- Fair cross-team comparisons (e.g., TSU vs VER in same car)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from risk_score_simple import calculate_risk_scores
from teammate_performance import calculate_teammate_performance

logging.getLogger("fastf1").setLevel(logging.WARNING)
fastf1.Cache.enable_cache("data/")

print("=" * 80)
print("F1 DRIVER STYLE CLASSIFIER - 3 CATEGORIES")
print("=" * 80)


# Step 1: Calculate risk scores
print("\n[STEP 1] Calculating risk scores (including crashes)...")
risk_df = calculate_risk_scores(2025)
print(f"✓ Risk scores calculated for {len(risk_df)} drivers")


# Step 1.5: Calculate teammate performance
print("\n[STEP 1.5] Calculating teammate performance...")
teammate_df = calculate_teammate_performance(2025)
print(f"✓ Teammate performance calculated for {len(teammate_df)} drivers")


# Step 2: Extract race performance features
print("\n[STEP 2] Extracting race performance features...")

year = 2025
schedule = fastf1.get_event_schedule(year)
session_dates = pd.to_datetime(schedule["Session5DateUtc"], utc=True).dt.tz_localize(None)
completed_races = schedule[session_dates < datetime.now()]

all_features = []

for _, race in completed_races.iterrows():
    race_name = race['EventName']

    try:
        session = fastf1.get_session(year, race_name, 'R')
        session.load()

        laps = session.laps
        results = session.results

        for driver_abbr in results['Abbreviation'].unique():
            driver_laps = laps[laps['Driver'] == driver_abbr]

            if len(driver_laps) < 3:
                continue

            lap_times = driver_laps['LapTime'].dt.total_seconds().dropna()
            if len(lap_times) < 3:
                continue

            # Consistency: Standard deviation of lap times (lower = better)
            consistency = float(lap_times.std())

            # Position change: Grid to finish (positive = gained places)
            driver_result = results[results['Abbreviation'] == driver_abbr].iloc[0]
            position_change = int(driver_result['GridPosition'] - driver_result['Position'])

            # Tyre degradation: Compare first 3 vs last 3 laps on same compound
            tyre_deg = 0.0
            for compound in driver_laps['Compound'].dropna().unique():
                compound_laps = driver_laps[driver_laps['Compound'] == compound]
                if len(compound_laps) >= 6:
                    first_3 = compound_laps.head(3)['LapTime'].dt.total_seconds().mean()
                    last_3 = compound_laps.tail(3)['LapTime'].dt.total_seconds().mean()
                    tyre_deg = max(tyre_deg, float(last_3 - first_3))

            all_features.append({
                'Driver': driver_abbr,
                'Race': race_name,
                'Consistency': consistency,
                'PositionChange': position_change,
                'TyreDegradation': tyre_deg,
            })

    except Exception:
        continue

print(f"✓ Extracted {len(all_features)} driver-race records")


# Step 3: Aggregate features per driver
print("\n[STEP 3] Aggregating features...")

features_df = pd.DataFrame(all_features)

driver_profiles = features_df.groupby('Driver', as_index=False).agg({
    'Consistency': 'mean',
    'PositionChange': 'mean',
    'TyreDegradation': 'mean',
})

# Add race count
driver_race_counts = features_df.groupby('Driver')['Race'].nunique()
driver_profiles['RacesCompleted'] = driver_profiles['Driver'].map(driver_race_counts).fillna(0).astype(int)

# Merge with risk scores
driver_profiles = driver_profiles.merge(
    risk_df[['Driver', 'RiskScore']],
    on='Driver',
    how='left'
)
driver_profiles['RiskScore'] = driver_profiles['RiskScore'].fillna(0)

# Merge with teammate performance
driver_profiles = driver_profiles.merge(
    teammate_df[['Driver', 'PointsDelta', 'QualifyingDelta', 'PositionDelta']],
    on='Driver',
    how='left'
)
driver_profiles['PointsDelta'] = driver_profiles['PointsDelta'].fillna(0)
driver_profiles['QualifyingDelta'] = driver_profiles['QualifyingDelta'].fillna(0)
driver_profiles['PositionDelta'] = driver_profiles['PositionDelta'].fillna(0)

# Filter to drivers who raced
race_drivers = driver_profiles[driver_profiles['RacesCompleted'] > 0].copy()

print(f"✓ {len(race_drivers)} race participants")
print("\nDriver Profiles:")
print(race_drivers[['Driver', 'RacesCompleted', 'RiskScore', 'PointsDelta',
                    'QualifyingDelta', 'PositionDelta']].to_string(index=False))


# Step 4: Apply seed labels
print("\n[STEP 4] Applying seed labels (3 categories)...")

# Minimal seed set: Only the most obvious examples from each category
# 8 seeds out of 21 drivers = 38% labelled (good train/predict ratio)
SEED_LABELS = {
    # AGGRESSIVE: High risk + good results
    "VER": "Aggressive",  # 18 risk, +245 pts vs TSU
    "ALB": "Aggressive",  # 24 risk, +41 pts vs SAI

    # CONSISTENT: Good performance + low risk
    "LEC": "Consistent",  # 5 risk, +53 pts vs HAM
    "NOR": "Consistent",  # 10 risk, -16 pts vs PIA
    "ALO": "Consistent",  # 10 risk, +8 pts vs STR

    # STRUGGLING: Poor vs teammate OR high risk + poor results
    "TSU": "Struggling",  # 37 risk, -245 pts vs VER (worst on grid)
    "ANT": "Struggling",  # 23 risk, -143 pts vs RUS (rookie struggles)
    "SAI": "Struggling",  # 36 risk, -41 pts vs ALB
}

race_drivers['Label'] = race_drivers['Driver'].map(SEED_LABELS)
labelled = race_drivers[race_drivers['Label'].notna()].copy()
unlabelled = race_drivers[race_drivers['Label'].isna()].copy()

print(f"✓ {len(labelled)} labelled drivers")
print(f"✓ {len(unlabelled)} unlabelled drivers")


# Step 5: Train classifier
print("\n[STEP 5] Training classifier...")

feature_cols = ['RiskScore', 'PointsDelta', 'QualifyingDelta', 'PositionDelta',
                'Consistency', 'PositionChange', 'TyreDegradation']
X_train = labelled[feature_cols].values
y_train = labelled['Label'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

clf = RandomForestClassifier(
    n_estimators=30,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train_scaled, y_train)

train_accuracy = clf.score(X_train_scaled, y_train)
print(f"✓ Training accuracy: {train_accuracy*100:.1f}%")

# Feature importance
importances = clf.feature_importances_
feature_importance = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

print("\nFeature Importance:")
for feature, importance in feature_importance:
    bar = "█" * int(importance * 50)
    print(f"  {feature:20s} {importance*100:5.1f}% {bar}")


# Step 6: Predict unlabelled drivers
print("\n[STEP 6] Predicting unlabelled drivers...")

if len(unlabelled) > 0:
    X_unlabelled = unlabelled[feature_cols].values
    X_unlabelled_scaled = scaler.transform(X_unlabelled)

    predictions = clf.predict(X_unlabelled_scaled)
    confidence = clf.predict_proba(X_unlabelled_scaled).max(axis=1)

    unlabelled['PredictedStyle'] = predictions
    unlabelled['Confidence'] = confidence

    print(f"✓ Predicted {len(unlabelled)} drivers\n")
    for _, row in unlabelled.sort_values('Confidence', ascending=False).iterrows():
        conf_pct = row['Confidence'] * 100
        marker = "✓" if conf_pct > 70 else "?" if conf_pct > 50 else "⚠"
        print(f"  {marker} {row['Driver']:3s}: {row['PredictedStyle']:15s} ({conf_pct:5.1f}%) "
              f"[Risk: {row['RiskScore']:.0f}]")


# Step 7: Final results and save
print("\n" + "=" * 80)
print("FINAL DRIVER CLASSIFICATIONS (3 CATEGORIES)")
print("=" * 80)

labelled['PredictedStyle'] = labelled['Label']
labelled['Confidence'] = 1.0

all_results = pd.concat([labelled, unlabelled], ignore_index=True)
all_results = all_results.sort_values(['PredictedStyle', 'RiskScore'], ascending=[True, False])

for style in sorted(all_results['PredictedStyle'].unique()):
    style_drivers = all_results[all_results['PredictedStyle'] == style]
    print(f"\n{style.upper()}:")
    for _, row in style_drivers.iterrows():
        is_seed = row['Driver'] in SEED_LABELS
        marker = "✓" if is_seed else "→"
        risk = row['RiskScore']
        pos = row['PositionChange']
        print(f"  {marker} {row['Driver']}: {row['Confidence']*100:5.1f}% confidence "
              f"(Risk: {risk:3.0f}, PosΔ: {pos:+.2f})")

# Save results as JSON
output = {
    'season': year,
    'categories': 3,
    'drivers': len(all_results),
    'seed_count': len(labelled),
    'predicted_count': len(unlabelled),
    'training_accuracy': float(train_accuracy),
    'feature_importance': {feat: float(imp) for feat, imp in feature_importance},
    'results': {}
}

for style in all_results['PredictedStyle'].unique():
    style_drivers = all_results[all_results['PredictedStyle'] == style]
    output['results'][style] = [
        {
            'driver': row['Driver'],
            'confidence': round(float(row['Confidence']), 4),
            'is_seed': row['Driver'] in SEED_LABELS,
            'features': {
                'consistency': round(float(row['Consistency']), 4),
                'position_change': round(float(row['PositionChange']), 4),
                'tyre_degradation': round(float(row['TyreDegradation']), 4),
                'risk_score': round(float(row['RiskScore']), 4)
            }
        }
        for _, row in style_drivers.iterrows()
    ]

output_path = Path('driver_classifications.json')
output_path.write_text(json.dumps(output, indent=2))

print(f"\n✓ Results saved to {output_path}")
print("\n" + "=" * 80)
