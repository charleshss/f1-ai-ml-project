"""
Teammate Performance Calculator
================================

Calculates driver performance metrics relative to teammate to normalise for
car performance differences.

This enables fair comparisons across teams by measuring drivers against their
teammates in identical machinery. For example:
- TSU vs VER (both in Red Bull)
- HAM vs LEC (both in Ferrari)
- GAS vs COL (both in Alpine)

Metrics:
- Points Delta: Championship points difference vs teammate
- Qualifying Delta: Average qualifying time gap (seconds)
- Race Pace Delta: Average race pace gap (seconds)
- Position Delta: Average finishing position difference

All deltas are signed:
- Positive points delta = ahead of teammate (better)
- Negative qualifying/pace delta = faster than teammate (better)
- Negative position delta = better finishing positions (better)
"""

import logging
from datetime import datetime

import fastf1
import pandas as pd
import numpy as np

logging.getLogger('fastf1').setLevel(logging.ERROR)
fastf1.Cache.enable_cache('data/')

# 2025 teammate pairs
TEAMMATE_PAIRS_2025 = {
    'VER': 'TSU',
    'TSU': 'VER',
    'NOR': 'PIA',
    'PIA': 'NOR',
    'LEC': 'HAM',
    'HAM': 'LEC',
    'ALO': 'STR',
    'STR': 'ALO',
    'RUS': 'ANT',
    'ANT': 'RUS',
    'GAS': 'COL',
    'COL': 'GAS',
    'DOO': 'GAS',
    'ALB': 'SAI',
    'SAI': 'ALB',
    'HUL': 'BEA',
    'BEA': 'HUL',
    'LAW': 'HAD',
    'HAD': 'LAW',
    'BOR': 'OCO',
    'OCO': 'BOR',
}


def calculate_teammate_performance(year: int = 2025) -> pd.DataFrame:
    """
    Calculate performance metrics relative to teammate.

    Processes all completed races to extract:
    - Championship points per driver
    - Qualifying times (best of Q3/Q2/Q1)
    - Race pace (average lap time excluding outliers)
    - Finishing positions

    Then calculates deltas vs teammate for each metric.

    Args:
        year: F1 season year

    Returns:
        DataFrame with columns:
        - Driver: Driver abbreviation
        - Teammate: Their teammate abbreviation
        - PointsDelta: Points difference vs teammate (positive = ahead)
        - QualifyingDelta: Average qualifying gap in seconds (negative = faster)
        - RacePaceDelta: Average race pace gap in seconds (negative = faster)
        - PositionDelta: Average position difference (negative = better finishes)
    """
    print(f"Calculating Teammate Performance for {year}")
    print("=" * 60)

    schedule = fastf1.get_event_schedule(year)
    session_dates = pd.to_datetime(schedule['Session5DateUtc'], utc=True).dt.tz_localize(None)
    completed = schedule[session_dates < datetime.now()]

    driver_points = {}
    driver_quali_times = {}
    driver_race_paces = {}
    driver_positions = {}

    for _, race in completed.iterrows():
        race_name = race['EventName']
        print(f"\n{race_name}...")

        # Race session
        try:
            race_session = fastf1.get_session(year, race_name, 'R')
            race_session.load()

            results = race_session.results
            laps = race_session.laps

            for _, row in results.iterrows():
                driver = row['Abbreviation']
                points = row['Points']
                position = row['Position']

                # Accumulate points
                if driver not in driver_points:
                    driver_points[driver] = 0
                driver_points[driver] += points

                # Record finishing positions
                if driver not in driver_positions:
                    driver_positions[driver] = []
                if pd.notna(position):
                    driver_positions[driver].append(int(position))

                # Calculate race pace (exclude pit laps and traffic)
                driver_laps = laps[laps['Driver'] == driver]
                lap_times = driver_laps['LapTime'].dt.total_seconds().dropna()

                if len(lap_times) > 5:
                    # Remove slowest 20% (pit laps, traffic)
                    clean_times = lap_times[lap_times < lap_times.quantile(0.8)]
                    if len(clean_times) > 0:
                        avg_pace = clean_times.mean()
                        if driver not in driver_race_paces:
                            driver_race_paces[driver] = []
                        driver_race_paces[driver].append(avg_pace)

        except Exception as e:
            print(f"  R: Skipped ({e})")

        # Qualifying session
        try:
            quali_session = fastf1.get_session(year, race_name, 'Q')
            quali_session.load()

            results = quali_session.results

            for _, row in results.iterrows():
                driver = row['Abbreviation']

                # Best qualifying time (Q3 > Q2 > Q1)
                best_time = None
                for q in ['Q3', 'Q2', 'Q1']:
                    if pd.notna(row.get(q)):
                        best_time = row[q].total_seconds()
                        break

                if best_time is not None:
                    if driver not in driver_quali_times:
                        driver_quali_times[driver] = []
                    driver_quali_times[driver].append(best_time)

        except Exception as e:
            print(f"  Q: Skipped ({e})")

    # Calculate deltas vs teammate
    results = []

    for driver, teammate in TEAMMATE_PAIRS_2025.items():
        if driver not in driver_points:
            continue

        driver_pts = driver_points.get(driver, 0)
        teammate_pts = driver_points.get(teammate, 0)
        points_delta = driver_pts - teammate_pts

        # Qualifying delta
        quali_delta = 0.0
        if driver in driver_quali_times and teammate in driver_quali_times:
            driver_quali = driver_quali_times[driver]
            teammate_quali = driver_quali_times[teammate]

            min_len = min(len(driver_quali), len(teammate_quali))
            if min_len > 0:
                gaps = [driver_quali[i] - teammate_quali[i] for i in range(min_len)]
                quali_delta = np.mean(gaps)

        # Race pace delta
        pace_delta = 0.0
        if driver in driver_race_paces and teammate in driver_race_paces:
            driver_paces = driver_race_paces[driver]
            teammate_paces = driver_race_paces[teammate]

            min_len = min(len(driver_paces), len(teammate_paces))
            if min_len > 0:
                gaps = [driver_paces[i] - teammate_paces[i] for i in range(min_len)]
                pace_delta = np.mean(gaps)

        # Position delta
        pos_delta = 0.0
        if driver in driver_positions and teammate in driver_positions:
            driver_pos = driver_positions[driver]
            teammate_pos = driver_positions[teammate]

            if len(driver_pos) > 0 and len(teammate_pos) > 0:
                pos_delta = np.mean(driver_pos) - np.mean(teammate_pos)

        results.append({
            'Driver': driver,
            'Teammate': teammate,
            'PointsDelta': points_delta,
            'QualifyingDelta': quali_delta,
            'RacePaceDelta': pace_delta,
            'PositionDelta': pos_delta,
        })

    df = pd.DataFrame(results).sort_values('PointsDelta', ascending=False)

    print("\n" + "=" * 60)
    print("TEAMMATE PERFORMANCE COMPARISON")
    print("=" * 60)
    print(df.to_string(index=False))

    return df


if __name__ == '__main__':
    teammate_df = calculate_teammate_performance(2025)

    teammate_df.to_csv('teammate_performance.csv', index=False)
    print("\n✓ Saved to teammate_performance.csv")
