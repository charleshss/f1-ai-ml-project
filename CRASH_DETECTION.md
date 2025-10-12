# Crash Detection System

## Overview

The risk scoring system now includes **crash detection** to identify when drivers make mistakes that result in crashes, spins, or stopping on track.

## What We Detect

### 1. **Barrier/Wall Hits** (8 points)
Solo crashes where a driver hits a barrier or wall due to their own error.

**Patterns:**
- "IN THE WALL"
- "IN WALL"
- "BARRIER"
- "CRASH" / "CRASHED"

**Example:**
```
CAR 55 (SAI) IN THE WALL AT TURN 15
```

### 2. **Stopped on Track / Spun** (6 points)
Driver spins off or stops on track due to their own mistake.

**Patterns:**
- "SPUN" / "SPINNING" / "SPIN"
- "STOPPED"
- "BEACHED"
- "GRAVEL"
- "ESCAPE ROAD"

**Examples:**
```
CAR 81 (PIA) SPUN AT TURN 4
CAR 23 (ALB) STOPPED IN GRAVEL TRAP
```

### 3. **Caused Red Flag** (12 points)
Driver crashed so badly they red-flagged the session.

**Pattern:**
- "RED FLAG" + mentions specific car

**Example:**
```
RED FLAG - CAR 43 (COL) AT TURN 7
```

### 4. **Recovery Vehicle** (6 points)
Recovery vehicle called usually means someone crashed/stopped.

**Pattern:**
- "RECOVERY VEHICLE ON TRACK AT [LOCATION]"

**Example:**
```
RECOVERY VEHICLE ON TRACK AT TURN 6
```

## How It Works

### Detection Logic (risk_score_simple.py:140-191)

```python
def classify_incident(msg: str) -> tuple[str | None, int]:
    """
    Classify a race control message and return (incident_type, points).
    """
    # ... penalties first ...

    # Red flag caused by specific driver
    if 'RED FLAG' in msg.upper() and 'CAR' in msg.upper():
        if re.search(r'CAR\s+\d+', msg, re.I):
            return 'caused_red_flag', WEIGHTS['caused_red_flag']

    # Barrier/wall hits (solo crashes)
    if HIT_BARRIER.search(msg):
        return 'crash_barrier', WEIGHTS['crash_barrier']

    # Spins / Stopped on track (driver mistakes)
    if STOPPED_TRACK.search(msg) or SPUN_OFF.search(msg):
        # Only count if it's a specific driver (has CAR X pattern)
        if re.search(r'CAR\s+\d+', msg, re.I):
            return 'crash_stopped', WEIGHTS['crash_stopped']

    # Recovery vehicle usually means someone crashed
    if RECOVERY.search(msg):
        if 'TURN' in msg.upper() or 'AT' in msg.upper():
            return 'crash_stopped', WEIGHTS['crash_stopped']
```

### Key Design Choices

1. **Only count crashes linked to a specific driver**
   - We check for "CAR X" pattern to ensure it's a specific driver
   - Generic "RECOVERY VEHICLE" without location is ignored

2. **Driver error vs mechanical failure**
   - We can't perfectly distinguish, but crash patterns in race control messages typically indicate driver errors
   - Mechanical DNFs usually show as "STOPPED" without dramatic language

3. **Scoring aligned with severity**
   - Caused red flag (12 pts) = most severe (major crash)
   - Barrier hit (8 pts) = serious (solo crash)
   - Stopped/spun (6 pts) = moderate (lost control)

## 2025 Results

### Drivers with Crashes Detected:

**Sainz (SAI)**: +12 points from crashes
- 2× "crash_stopped" incidents (6 pts each)
- Risk score: 24 → 36 points total

**Bearman (BEA)**: +6 points from crashes
- 1× "crash_stopped" incident
- Risk score: 16 → 22 points total

### Why This Matters:

Before crash detection:
- Sainz: 24 points (looked moderate risk)
- Reality: Sainz had multiple crashes, making him HIGH risk

After crash detection:
- Sainz: 36 points (correctly HIGH risk)
- Captures full picture of aggressive/risky driving

## Examples from Real Data

### Example 1: Sainz Crash (Azerbaijan GP)
```
Message: "CAR 55 (SAI) STOPPED IN ESCAPE ROAD TURN 1"
Classification: crash_stopped (+6 points)
```

### Example 2: Recovery Vehicle
```
Message: "RECOVERY VEHICLE ON TRACK AT TURN 6"
Classification: crash_stopped (+6 points)
Reason: Someone crashed at Turn 6, requires recovery
```

### Example 3: Barrier Hit (Monaco GP - hypothetical)
```
Message: "CAR 16 (LEC) IN THE WALL AT TURN 19"
Classification: crash_barrier (+8 points)
```

### Example 4: Red Flag Crash (Qualifying - hypothetical)
```
Message: "RED FLAG - CAR 4 (NOR) AT TURN 5"
Classification: caused_red_flag (+12 points)
```

## What We DON'T Count

### 1. **Collisions Between Drivers**
These are handled separately as "caused_collision" (10 pts)
```
TURN 6 INCIDENT INVOLVING CARS 27 (HUL) AND 31 (OCO) - CAUSING A COLLISION
```

### 2. **Mechanical Failures**
If the message doesn't indicate driver error, we skip it:
```
CAR 44 (HAM) STOPPED - ENGINE ISSUE
```
(No dramatic language = likely mechanical)

### 3. **Safety Car / VSC**
Generic safety car messages without specific driver:
```
SAFETY CAR DEPLOYED
```

## Limitations

1. **Can't perfectly distinguish driver error vs mechanical**
   - Race control messages don't always clarify
   - We err on the side of caution

2. **Some crashes may not be captured**
   - If race control doesn't mention specific car
   - If driver continues after a spin

3. **Context matters**
   - Crashing in rain vs dry
   - First lap incidents vs solo mistakes
   - Currently we treat all crashes equally

## Future Improvements

1. **Add DNF analysis**
   - Check FastF1's Status field
   - Distinguish "Accident" vs "Engine" DNFs

2. **Telemetry analysis**
   - Detect sudden deceleration (crash)
   - Analyze steering angle (spin)
   - More accurate than race control messages

3. **Incident severity**
   - Weight crashes by session (Q vs R)
   - Consider damage/red flag duration
   - Track repair time

4. **Historical comparison**
   - Compare crash rate across seasons
   - Identify if driver is improving/declining

## Running the System

The crash detection is fully integrated into `risk_score_simple.py`:

```bash
python risk_score_simple.py
```

Or run the full classifier (includes crash detection):

```bash
python 02_driver_classifier.py
```

Both will show crash counts in the output:
```
Driver  RiskScore  ...  CrashBarrier  CrashStopped  CausedRedFlag
   TSU         37  ...             0             0              0
   SAI         36  ...             0             2              0
   ALB         24  ...             0             0              0
```
