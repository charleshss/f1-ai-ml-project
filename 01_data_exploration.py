import fastf1
import pandas as pd

fastf1.Cache.enable_cache('data/')

year = 2025
race = 'Bahrain'

print(f"Loading {year} {race} GP - Driver Behavior Analysis\n")

session = fastf1.get_session(year, race, 'R')
session.load()

print("=" * 60)
print("DRIVER CLASSIFICATION - FEATURE EXPLORATION")
print("=" * 60)

# 1. RACE CONTROL MESSAGES - Penalties & Incidents
print("\n📋 RACE CONTROL MESSAGES (Penalties/Warnings):")
race_control = session.race_control_messages
print(f"Total messages: {len(race_control)}")
print(f"Columns: {list(race_control.columns)}\n")

# Filter for penalties/investigations
if len(race_control) > 0:
    penalties = race_control[race_control['Message'].str.contains('PENALTY|INVESTIGATION|WARNING', case=False, na=False)]
    print(f"Penalty-related messages: {len(penalties)}")
    if len(penalties) > 0:
        print("\nSample penalties:")
        print(penalties[['Time', 'Category', 'Message']].head(3))

# 2. DRIVER CONSISTENCY - Lap time variation
print("\n\n⏱️ DRIVER CONSISTENCY (Lap Time Std Dev):")
laps = session.laps

# Calculate lap time consistency per driver
driver_consistency = laps.groupby('Driver')['LapTime'].agg(['mean', 'std', 'count'])
driver_consistency = driver_consistency.dropna()
driver_consistency['std_seconds'] = driver_consistency['std'].dt.total_seconds()
print(driver_consistency.head())

# 3. TIRE MANAGEMENT
print("\n\n🏎️ TIRE MANAGEMENT:")
# Average lap time degradation on each tire stint
tire_data = laps[['Driver', 'Compound', 'TyreLife', 'LapTime']].dropna()
print(f"Sample tire data:")
print(tire_data.head())

# 4. TRACK POSITION CHANGES (Overtaking/Defending)
print("\n\n🔄 POSITION CHANGES (Aggression Indicator):")
# Calculate how many positions each driver gained/lost
results = session.results
results['PositionGained'] = results['GridPosition'] - results['Position']
print(results[['Abbreviation', 'GridPosition', 'Position', 'PositionGained']].sort_values('PositionGained', ascending=False))

print("\n" + "=" * 60)
print("FEATURES WE CAN USE FOR CLASSIFICATION:")
print("=" * 60)
print("""
✅ Lap time consistency (std deviation)
✅ Position changes (overtakes/aggression) 
✅ Tire degradation rate
✅ Penalty count (from race control)
✅ Incident involvement (from messages)
✅ Sector time patterns
""")