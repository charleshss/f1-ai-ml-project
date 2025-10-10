import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache('data/')

print("=" * 60)
print("DRIVER STYLE CLASSIFIER - FEATURE EXTRACTION")
print("=" * 60)

# We'll analyze multiple 2025 races for better classification
races_to_analyze = ['Bahrain', 'Saudi Arabia', 'Australia']  # First 3 races of 2025

all_driver_features = []

for race_name in races_to_analyze:
    print(f"\nProcessing {race_name}...")
    
    try:
        session = fastf1.get_session(2025, race_name, 'R')
        session.load()
        
        laps = session.laps
        results = session.results
        race_control = session.race_control_messages
        
        # Calculate features for each driver in this race
        for driver_abbr in results['Abbreviation'].unique():
            driver_laps = laps[laps['Driver'] == driver_abbr].copy()
            
            if len(driver_laps) < 5:  # Skip if too few laps
                continue
            
            # Feature 1: Lap time consistency (lower = more consistent)
            lap_times_seconds = driver_laps['LapTime'].dt.total_seconds()
            consistency = lap_times_seconds.std()
            
            # Feature 2: Position change (aggression indicator)
            driver_result = results[results['Abbreviation'] == driver_abbr].iloc[0]
            position_change = driver_result['GridPosition'] - driver_result['Position']
            
            # Feature 3: Tire degradation rate
            # Calculate avg lap time increase per tire life lap
            tire_deg = 0
            for compound in driver_laps['Compound'].unique():
                compound_laps = driver_laps[driver_laps['Compound'] == compound]
                if len(compound_laps) > 5:
                    # Simple degradation: difference between first and last 3 laps on stint
                    first_laps = compound_laps.head(3)['LapTime'].dt.total_seconds().mean()
                    last_laps = compound_laps.tail(3)['LapTime'].dt.total_seconds().mean()
                    tire_deg = max(tire_deg, last_laps - first_laps)
            
            # Feature 4: Penalty count
            driver_penalties = race_control[
                (race_control['Message'].str.contains(driver_abbr, case=False, na=False)) &
                (race_control['Message'].str.contains('PENALTY', case=False, na=False))
            ]
            penalty_count = len(driver_penalties)
            
            all_driver_features.append({
                'Driver': driver_abbr,
                'Race': race_name,
                'Consistency': consistency,
                'PositionChange': position_change,
                'TireDegradation': tire_deg,
                'PenaltyCount': penalty_count
            })
    
    except Exception as e:
        print(f"Could not load {race_name}: {e}")

# Create DataFrame
features_df = pd.DataFrame(all_driver_features)

# Aggregate features across all races per driver
driver_profiles = features_df.groupby('Driver').agg({
    'Consistency': 'mean',
    'PositionChange': 'mean',
    'TireDegradation': 'mean',
    'PenaltyCount': 'sum'
}).reset_index()

print("\n" + "=" * 60)
print("DRIVER PROFILES:")
print("=" * 60)
print(driver_profiles)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("\n" + "=" * 60)
print("APPLYING K-MEANS CLUSTERING")
print("=" * 60)

# Prepare features for clustering (exclude Driver column)
features_for_clustering = driver_profiles[['Consistency', 'PositionChange', 'TireDegradation', 'PenaltyCount']]

# Normalise features (important! They're on different scales)
scaler = StandardScaler()
features_normalised = scaler.fit_transform(features_for_clustering)

# Apply K-Means clustering (let's try 4 clusters)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
driver_profiles['Cluster'] = kmeans.fit_predict(features_normalised)

# Show results grouped by cluster
print(f"\nDrivers grouped into {n_clusters} clusters:\n")
for cluster_id in range(n_clusters):
    cluster_drivers = driver_profiles[driver_profiles['Cluster'] == cluster_id]
    print(f"--- CLUSTER {cluster_id} ---")
    print(cluster_drivers[['Driver', 'Consistency', 'PositionChange', 'TireDegradation', 'PenaltyCount']])
    print(f"Average stats: Pos Change={cluster_drivers['PositionChange'].mean():.1f}, "
          f"Tyre Deg={cluster_drivers['TireDegradation'].mean():.1f}, "
          f"Penalties={cluster_drivers['PenaltyCount'].mean():.1f}\n")
    
# Assign meaningful labels to clusters
cluster_labels = {
    0: "Penalty Prone",
    1: "Clean & Consistent", 
    2: "Struggling/Unlucky",
    3: "Clean Overtakers"
}

driver_profiles['Style'] = driver_profiles['Cluster'].map(cluster_labels)

print("\n" + "=" * 60)
print("FINAL DRIVER CLASSIFICATIONS:")
print("=" * 60)
print(driver_profiles[['Driver', 'Style', 'PositionChange', 'PenaltyCount']].sort_values('Style'))

# Save to JSON for web visualisation later
import json

output_data = driver_profiles.to_dict('records')
with open('driver_classifications.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("\n✅ Results saved to driver_classifications.json")