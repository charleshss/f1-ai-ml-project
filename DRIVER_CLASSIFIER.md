# F1 Driver Style Classifier - Technical Documentation

## Overview

The F1 Driver Style Classifier is a supervised machine learning system that categorizes Formula 1 drivers into **3 performance categories** based on their 2025 season performance. It combines multiple data sources to provide a holistic view of driver behavior and performance.

## Driver Categories

### 1. AGGRESSIVE
High-risk drivers who push the limits, resulting in penalties and crashes, but often with strong underlying pace.

**Characteristics:**
- High risk scores (18-24+ points)
- Frequent penalties and incidents
- Pushing car to the limit
- Can deliver strong results despite risky driving

**Examples:** Verstappen (18 risk, +245 pts vs teammate), Albon (24 risk, +41 pts vs teammate)

### 2. CONSISTENT
Reliable drivers with clean racing, solid points scoring, and good racecraft.

**Characteristics:**
- Low to moderate risk scores (5-10 points)
- Few penalties or incidents
- Stable lap times
- Positive or neutral teammate comparison
- Reliable points finishes

**Examples:** Leclerc (5 risk, +53 pts vs teammate), Norris (10 risk, -16 pts vs teammate), Alonso (10 risk, +8 pts vs teammate)

### 3. STRUGGLING
Drivers performing poorly relative to their teammate OR showing high incidents with poor results.

**Characteristics:**
- High risk scores with poor results OR
- Significantly behind teammate in performance metrics
- Poor qualifying/race pace relative to teammate
- Off-pace or frequent mistakes

**Examples:** Tsunoda (37 risk, -245 pts vs teammate - worst on grid), Antonelli (23 risk, -143 pts vs teammate - rookie struggles), Sainz (36 risk, -41 pts vs teammate)

## Feature Engineering

The classifier uses **7 features** across 3 data sources:

### 1. Risk Score (from `risk_score_simple.py`)

**What it measures:** Driver mistakes, penalties, and dangerous driving

**Components:**
- **Penalties:**
  - 5-second penalties: 5 points
  - 10-second penalties: 8 points
  - Drive-through penalties: 10 points
  - Grid penalties: 5 points
- **Collisions:**
  - Causing collision: 10 points
- **Crashes (driver error):**
  - Barrier/wall hits: 8 points
  - Spins/stops on track: 6 points
  - Caused red flag: 12 points
- **Other:**
  - False starts: 5 points
  - Persistent track limits: 3 points

**Data source:** FastF1 race control messages from all qualifying and race sessions

**Alignment:** Scoring weights aligned with FIA penalty point system severity

### 2. Teammate Performance (from `teammate_performance.py`)

**Why this matters:** Normalizes for car performance - comparing drivers in the same machinery.

**Metrics:**

**a) Points Delta**
- Total championship points vs teammate
- Positive = ahead of teammate
- Example: VER +245 vs TSU (dominant season)

**b) Qualifying Delta (seconds)**
- Average gap to teammate across all qualifying sessions
- Negative = faster than teammate
- Compares best Q3/Q2/Q1 times per session

**c) Position Delta**
- Average finishing position difference
- Negative = better finishing positions
- Example: TSU averages worse finishes than VER

**Feature Importance:** PointsDelta is the **most important feature** (27.8% importance) - beating your teammate matters more than raw risk score.

### 3. Race Performance Features (from lap data)

**a) Consistency (lap time variance)**
- Standard deviation of lap times
- Lower = more consistent
- Filters out outliers (pit laps, traffic)

**b) Position Change**
- Grid position → finishing position delta
- Positive = gained positions (good overtaker)
- Example: ALB +1.5 average (strong racer)

**c) Tyre Degradation**
- Lap time delta: first 3 laps vs last 3 laps on same compound
- Lower = better tyre management
- Indicates racecraft and car management

## Machine Learning Approach

### Supervised Learning with Minimal Seeds

**Training data:** 8 seed labels out of 21 drivers (38% pre-labeled)

**Why so few seeds?**
- Prevents overfitting
- Allows model to learn patterns from data
- Creates meaningful predictions with confidence scores

**Seed drivers:**
- Aggressive: VER, ALB (clear high-risk + results)
- Consistent: LEC, NOR, ALO (clean racing + solid performance)
- Struggling: TSU, ANT, SAI (poor teammate comparison OR high risk + poor results)

### Model: Random Forest Classifier

**Configuration:**
```python
RandomForestClassifier(
    n_estimators=30,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)
```

**Why Random Forest?**
- Handles non-linear relationships
- Provides feature importance
- Resistant to overfitting with proper hyperparameters
- Works well with small datasets

### Feature Importance (2025 Season)

1. **PointsDelta** (27.8%) - Most important: teammate comparison
2. **PositionDelta** (16.8%) - Finishing position vs teammate
3. **RiskScore** (16.7%) - Penalties and crashes
4. **PositionChange** (10.9%) - Overtaking ability
5. **TyreDegradation** (10.9%) - Racecraft
6. **QualifyingDelta** (8.8%) - Pace vs teammate
7. **Consistency** (8.2%) - Lap time variance

**Key insight:** Teammate performance (PointsDelta + PositionDelta + QualifyingDelta = 53.4%) is more important than risk score alone.

### Model Output

**For each driver:**
- Predicted category (Aggressive/Consistent/Struggling)
- Confidence score (0.0-1.0)
- Whether they were a seed label or predicted
- All feature values

**Example predictions:**
- RUS: Aggressive (60% confidence) - model prediction
- PIA: Consistent (63% confidence) - model prediction
- HUL: Consistent (63% confidence) - model prediction

## Data Sources

### FastF1 API

**What we use:**
- Race control messages (penalties, incidents, crashes)
- Lap times and telemetry
- Session results (positions, points, tyre compounds)
- Qualifying times (Q1/Q2/Q3)

**What FastF1 captures:**
- ✅ Real-time race control messages during sessions
- ✅ All lap times and sector times
- ✅ Tyre compounds and pit stops
- ✅ Session results and classifications

**What FastF1 does NOT capture:**
- ❌ Post-session steward decisions
- ❌ Grid penalties applied to future races
- ❌ FIA steward document details
- ❌ Driver intent or mechanical failures (explicitly)

### 2025 Teammate Pairs

```python
TEAMMATE_PAIRS_2025 = {
    'VER': 'TSU',  # Red Bull
    'NOR': 'PIA',  # McLaren
    'LEC': 'HAM',  # Ferrari
    'ALO': 'STR',  # Aston Martin
    'RUS': 'ANT',  # Mercedes (ANT replaced BOT mid-season)
    'GAS': 'COL',  # Alpine (COL replaced DOO)
    'ALB': 'SAI',  # Williams
    'HUL': 'BEA',  # Haas
    'LAW': 'HAD',  # RB (formerly Alpha Tauri)
    'BOR': 'OCO',  # Sauber/Kick
}
```

## Limitations and Design Decisions

### 1. Crash Detection Limitations

**Challenge:** Cannot perfectly distinguish driver error vs mechanical failure

**Examples:**
- Barrier hit = likely driver error
- "STOPPED" without dramatic language = likely mechanical
- Spin in wet vs dry conditions = treated equally

**Design decision:** Use race control message language as proxy for driver error
- Messages with "IN THE WALL", "SPUN", "BARRIER" = driver error
- Generic "STOPPED" messages = filtered out

**Impact:** Conservative approach - we may undercount some crashes

### 2. Impeding Penalties

**Challenge:** Impeding during qualifying is procedural, not dangerous driving

**Example:** Hadjar penalized for impeding Sainz (Canadian GP)
- FastF1 captures: "IMPEDING" message
- Our system: Correctly filters as noise
- Grid penalty applied to next race
- Not counted in risk score

**Why filtered?**
- Not indicative of dangerous/risky driving style
- Purely procedural (didn't see other car)
- Different from racing incidents

### 3. Grid Penalties for Future Races

**Challenge:** Grid penalties applied to future races don't appear in FastF1

**Example:** Canadian GP impeding → penalty for next race
- Announced via FIA steward documents
- Not captured in race control messages
- Not available through FastF1 API

**Impact:** System only captures penalties applied in the session they occurred

### 4. Rookie vs Veteran Context

**Challenge:** Model treats all drivers equally

**Examples:**
- Antonelli (ANT): Rookie struggles are normal
- Verstappen (VER): 18 risk score but world champion

**Design decision:** Let the model learn these nuances from features
- ANT has high risk (23) + poor teammate comparison (-143 pts) = Struggling
- VER has moderate risk (18) + dominant teammate comparison (+245 pts) = Aggressive

**Future improvement:** Could add "years of experience" feature

### 5. Session Context

**Challenge:** Not all incidents are equal

**Examples:**
- First lap incidents (chaotic, many cars)
- Rain vs dry crashes
- Sprint vs race penalties
- Qualifying vs race incidents

**Current approach:** All incidents weighted equally

**Future improvement:** Weight by session type and conditions

### 6. Small Dataset

**Challenge:** Only 21 drivers across 2025 season

**Impact:**
- Limited training data
- Model may overfit to specific drivers
- Confidence scores vary (46-83%)

**Mitigation:**
- Use only 38% seed labels
- Random Forest with max_depth=3
- Balanced class weights

### 7. Mechanical DNFs vs Crash DNFs

**Challenge:** FastF1 doesn't explicitly label DNF reasons

**Current approach:**
- Use race control message language
- Crash patterns = driver error
- Generic "STOPPED" = filtered out

**Future improvement:** Analyze FastF1's "Status" field in results

## Why These Design Decisions?

### 1. Minimal Seed Labels (38%)

**Goal:** Let the model learn patterns, not memorize

**Trade-off:**
- More seeds = higher accuracy, less generalization
- Fewer seeds = lower accuracy, better generalization
- 38% is sweet spot for 21 drivers

### 2. Teammate Performance as Primary Feature

**Goal:** Fair comparison across different cars

**Why it matters:**
- TSU's 37 risk score looks terrible
- But -245 pts vs VER tells the real story
- HAM's struggles vs LEC show Ferrari transition

**Impact:** PointsDelta became most important feature (27.8%)

### 3. Aligned with FIA Penalty System

**Goal:** Risk scores that make F1 sense

**Approach:**
- 10s penalty more severe than 5s = 8 pts vs 5 pts
- Collision = 10 pts (matches FIA penalty points)
- Barrier hit = 8 pts (severe solo crash)

**Validation:** Tsunoda (37 pts), Sainz (36 pts), Leclerc (5 pts) all make sense to F1 fans

### 4. Filter Noise Early

**Goal:** Only count meaningful incidents

**Filtered out:**
- Blue flags (procedural)
- "NO FURTHER ACTION" (no penalty)
- Impeding (procedural)
- Generic safety messages

**Impact:** Clean signal, no inflated scores

### 5. Use Structured Data When Available

**Goal:** Reliability over regex fragility

**Example:** FastF1's `RacingNumber` field instead of parsing "CAR 44" text

**Trade-off:** Miss some incidents in text-only messages, but avoid false positives

## System Validation

### Does it make F1 sense?

**High risk drivers:**
- TSU (37): Known for incidents and struggles vs VER ✅
- SAI (36): Multiple crashes this season ✅
- ALB (24): Scrappy racer, Williams struggles ✅

**Clean drivers:**
- HAM (0): Veteran, clean driving ✅
- LEC (5): Ferrari's consistent performer ✅
- NOR (10): McLaren's reliable point scorer ✅

**Struggling drivers:**
- TSU: -245 pts vs VER (worst on grid) ✅
- ANT: -143 pts vs RUS (rookie adaptation) ✅
- SAI: 36 risk + -41 pts vs ALB ✅

### Feature Importance Makes Sense

1. **PointsDelta** (27.8%): Championship is about points ✅
2. **PositionDelta** (16.8%): Finishing ahead of teammate ✅
3. **RiskScore** (16.7%): Mistakes cost points ✅

## Running the System

### Full Classification

```bash
# Runs risk scoring + teammate comparison + classification
python 02_driver_classifier.py
```

**Output:**
- Console: Detailed classification with confidence scores
- `driver_classifications.json`: Full results with features

### Individual Components

```bash
# Risk scores only
python risk_score_simple.py

# Teammate performance only
python teammate_performance.py
```

## Future Improvements

### Data Quality
1. **Add Sprint sessions** - currently only Q + R
2. **DNF analysis** - use FastF1's Status field
3. **Telemetry-based crash detection** - sudden deceleration, steering angle
4. **Post-session steward decisions** - external data source

### Features
1. **Overtake count** - from telemetry
2. **Years of experience** - context for rookies
3. **Incident severity** - weight by session type
4. **Weather conditions** - crash context

### Model
1. **More seasons** - 2024 vs 2025 comparison
2. **Cross-validation** - better confidence estimates
3. **Ensemble methods** - combine multiple models
4. **Temporal features** - improving/declining trends

### Visualization
1. **Interactive dashboard** - Plotly/Streamlit
2. **Driver profiles** - deep dive per driver
3. **Season evolution** - how drivers change over time
4. **Feature correlation matrix** - understand relationships

## Technical Stack

- **FastF1**: F1 telemetry and timing data
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning (RandomForestClassifier, StandardScaler)
- **Python 3.12+**: Core language

## References

- [FastF1 Documentation](https://docs.fastf1.dev/)
- [FIA Penalty Points System](https://www.fia.com/)
- [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## License

MIT License - see LICENSE file
