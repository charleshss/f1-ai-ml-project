"""
F1 Risk Score Calculator
========================

Calculates risk scores for F1 drivers based on penalties, crashes, and incidents.
Scoring aligned with FIA penalty severity to capture dangerous/aggressive driving.

Key Principles:
- Only count confirmed penalties (not investigations)
- Focus on driver actions, not procedural violations
- Use structured data from FastF1 when available
- Filter noise early (blue flags, "no further action", etc.)

What we count:
- Time penalties (5s/10s/drive-through)
- Grid penalties for causing collisions
- Crashes (barriers, spins, stops on track)
- False starts (when penalised)
- Persistent track limit violations (3+)

What we don't count:
- "Noted" or "under investigation" with no penalty
- "No further action" outcomes
- Blue flags (routine traffic management)
- Impeding (procedural, not dangerous driving)
- Single track limit violations (common for all drivers)
"""

import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import Counter

import fastf1
import pandas as pd

logging.getLogger('fastf1').setLevel(logging.ERROR)
fastf1.Cache.enable_cache('data/')

# Scoring weights aligned with F1 penalty severity
WEIGHTS = {
    'penalty_5s': 5,
    'penalty_10s': 8,
    'penalty_drive_through': 10,
    'penalty_grid': 5,
    'caused_collision': 10,
    'crash_barrier': 8,
    'crash_stopped': 6,
    'caused_red_flag': 12,
    'dnf_crash': 7,
    'false_start': 5,
    'track_limits_persistent': 3,
}

# Regex patterns for incident detection
PENALTY_5S = re.compile(r'5\s*SECOND\s*(?:TIME\s*)?PENALTY', re.I)
PENALTY_10S = re.compile(r'10\s*SECOND\s*(?:TIME\s*)?PENALTY', re.I)
PENALTY_DRIVE_THROUGH = re.compile(r'DRIVE\s*THROUGH', re.I)
PENALTY_GRID = re.compile(r'GRID\s*(?:PLACE\s*)?PENALTY', re.I)
CAUSED_COLLISION = re.compile(r'CAUSING\s+(?:A\s+)?COLLISION', re.I)
FALSE_START = re.compile(r'FALSE\s*START', re.I)
HIT_BARRIER = re.compile(r'(?:IN\s+(?:THE\s+)?WALL|BARRIER|CRASH(?:ED)?)', re.I)
SPUN_OFF = re.compile(r'(?:SPUN|SPINNING|SPIN\b)', re.I)
STOPPED_TRACK = re.compile(r'(?:STOPPED|BEACHED|GRAVEL|ESCAPE\s+ROAD)', re.I)
RECOVERY = re.compile(r'RECOVERY\s+VEHICLE', re.I)
NO_ACTION = re.compile(r'NO\s+FURTHER\s+(?:ACTION|INVESTIGATION)', re.I)
NOTED_ONLY = re.compile(r'\bNOTED\b(?!.*PENALTY)', re.I)
UNDER_INVESTIGATION = re.compile(r'UNDER\s+INVESTIGATION|WILL\s+BE\s+INVESTIGATED', re.I)


def is_noise_message(msg: str) -> bool:
    """
    Filter out noise messages that don't indicate risky driving.

    Args:
        msg: Race control message text

    Returns:
        True if message should be ignored, False otherwise
    """
    if not msg:
        return True

    if NO_ACTION.search(msg):
        return True
    if NOTED_ONLY.search(msg) and 'PENALTY' not in msg.upper():
        return True
    if 'BLUE FLAG' in msg.upper():
        return True
    if 'PIT LANE INFRINGEMENT' in msg.upper():
        return True
    if 'IMPEDING' in msg.upper():
        return True

    return False


def extract_driver_from_message(msg: str, racing_number: any) -> str | None:
    """
    Extract driver abbreviation from race control message.

    Prefers FastF1's structured RacingNumber field, falls back to parsing
    "CAR X (ABC)" pattern from message text.

    Args:
        msg: Race control message text
        racing_number: FastF1 RacingNumber field value

    Returns:
        Driver abbreviation or None if not found
    """
    if pd.notna(racing_number):
        return str(int(racing_number))

    match = re.search(r'CAR\s+\d+\s*\(([A-Z]{3})\)', msg)
    if match:
        return match.group(1)

    return None


def classify_incident(msg: str) -> tuple[str | None, int]:
    """
    Classify a race control message and return incident type and severity.

    Args:
        msg: Race control message text

    Returns:
        Tuple of (incident_type, points) or (None, 0) if not relevant
    """
    if is_noise_message(msg):
        return None, 0

    # Time penalties (confirmed incidents)
    if PENALTY_10S.search(msg):
        return 'penalty_10s', WEIGHTS['penalty_10s']
    if PENALTY_5S.search(msg):
        return 'penalty_5s', WEIGHTS['penalty_5s']
    if PENALTY_DRIVE_THROUGH.search(msg):
        return 'penalty_drive_through', WEIGHTS['penalty_drive_through']
    if PENALTY_GRID.search(msg):
        return 'penalty_grid', WEIGHTS['penalty_grid']

    # Collisions
    if CAUSED_COLLISION.search(msg):
        if 'PENALTY' in msg.upper() or not UNDER_INVESTIGATION.search(msg):
            return 'caused_collision', WEIGHTS['caused_collision']

    # False starts
    if FALSE_START.search(msg) and 'PENALTY' in msg.upper():
        return 'false_start', WEIGHTS['false_start']

    # Crashes and driver errors
    if 'RED FLAG' in msg.upper() and 'CAR' in msg.upper():
        if re.search(r'CAR\s+\d+', msg, re.I):
            return 'caused_red_flag', WEIGHTS['caused_red_flag']

    if HIT_BARRIER.search(msg):
        return 'crash_barrier', WEIGHTS['crash_barrier']

    if STOPPED_TRACK.search(msg) or SPUN_OFF.search(msg):
        if re.search(r'CAR\s+\d+', msg, re.I):
            return 'crash_stopped', WEIGHTS['crash_stopped']

    if RECOVERY.search(msg):
        if 'TURN' in msg.upper() or 'AT' in msg.upper():
            return 'crash_stopped', WEIGHTS['crash_stopped']

    return None, 0


def calculate_track_limits_score(driver_deletions: int) -> int:
    """
    Calculate score for track limit violations.

    Only persistent offenders (3+ violations) are scored, as occasional
    violations are common for all drivers.

    Args:
        driver_deletions: Number of lap time deletions

    Returns:
        Risk score for track limit violations
    """
    if driver_deletions >= 3:
        excess = driver_deletions - 2
        return excess * WEIGHTS['track_limits_persistent']
    return 0


def calculate_risk_scores(year: int = 2025) -> pd.DataFrame:
    """
    Calculate risk scores for all drivers in a season.

    Processes qualifying and race sessions from completed events, extracting
    penalties, crashes, and incidents from race control messages.

    Args:
        year: F1 season year

    Returns:
        DataFrame with columns:
        - Driver: Driver abbreviation
        - RiskScore: Total risk score
        - IncidentScore: Score from penalties/crashes
        - TrackLimitsScore: Score from track limit violations
        - TrackLimitViolations: Number of lap deletions
        - Penalties5s, Penalties10s, etc.: Breakdown by incident type
    """
    print(f"Calculating F1 Risk Scores for {year} Season")
    print("=" * 60)

    schedule = fastf1.get_event_schedule(year)
    session_dates = pd.to_datetime(schedule['Session5DateUtc'], utc=True).dt.tz_localize(None)
    completed = schedule[session_dates < datetime.now()]

    driver_incidents: Dict[str, List[str]] = {}
    driver_track_limits: Dict[str, int] = Counter()

    for _, race in completed.iterrows():
        race_name = race['EventName']
        print(f"\n{race_name}...")

        for session_type in ['Q', 'R']:
            try:
                session = fastf1.get_session(year, race_name, session_type)
                session.load(messages=True, laps=False, telemetry=False, weather=False)

                messages = session.race_control_messages
                if messages is None or messages.empty:
                    continue

                results = session.results

                # Build driver number to abbreviation mapping
                driver_map = {}
                if results is not None and not results.empty:
                    driver_map = results.set_index('DriverNumber')['Abbreviation'].to_dict()

                # Process each race control message
                for _, row in messages.iterrows():
                    msg = str(row.get('Message', ''))

                    # Track limit violations
                    if 'DELETED' in msg.upper() and 'TRACK LIMITS' in msg.upper():
                        racing_num = row.get('RacingNumber')
                        if pd.notna(racing_num):
                            driver_abbr = driver_map.get(int(racing_num))
                            if driver_abbr:
                                driver_track_limits[driver_abbr] += 1
                        continue

                    # Other incidents
                    incident_type, _ = classify_incident(msg)
                    if incident_type:
                        racing_num = row.get('RacingNumber')
                        driver_abbr = None

                        if pd.notna(racing_num):
                            driver_abbr = driver_map.get(int(racing_num))
                        else:
                            driver_abbr = extract_driver_from_message(msg, None)

                        if driver_abbr:
                            if driver_abbr not in driver_incidents:
                                driver_incidents[driver_abbr] = []
                            driver_incidents[driver_abbr].append(incident_type)

            except Exception as e:
                print(f"  {session_type}: Skipped ({e})")

    # Calculate final scores
    results = []
    all_drivers = set(driver_incidents.keys()) | set(driver_track_limits.keys())

    for driver in all_drivers:
        incidents = driver_incidents.get(driver, [])
        track_limits = driver_track_limits.get(driver, 0)

        incident_score = sum(WEIGHTS.get(inc, 0) for inc in incidents)
        tl_score = calculate_track_limits_score(track_limits)
        total_score = incident_score + tl_score

        incident_counts = Counter(incidents)

        results.append({
            'Driver': driver,
            'RiskScore': total_score,
            'IncidentScore': incident_score,
            'TrackLimitsScore': tl_score,
            'TrackLimitViolations': track_limits,
            'Penalties5s': incident_counts.get('penalty_5s', 0),
            'Penalties10s': incident_counts.get('penalty_10s', 0),
            'CausedCollisions': incident_counts.get('caused_collision', 0),
            'CrashBarrier': incident_counts.get('crash_barrier', 0),
            'CrashStopped': incident_counts.get('crash_stopped', 0),
            'CausedRedFlag': incident_counts.get('caused_red_flag', 0),
            'FalseStarts': incident_counts.get('false_start', 0),
        })

    df = pd.DataFrame(results).sort_values('RiskScore', ascending=False)

    print("\n" + "=" * 60)
    print("FINAL RISK SCORES")
    print("=" * 60)
    print(df.to_string(index=False))

    return df


if __name__ == '__main__':
    risk_df = calculate_risk_scores(2025)

    output_path = Path('risk_scores_clean.csv')
    risk_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
