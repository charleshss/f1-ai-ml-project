# F1 Driver Style Classifier

Machine learning system to classify F1 drivers into 3 categories based on their 2025 season performance.

**📖 For detailed technical documentation, see [DRIVER_CLASSIFIER.md](DRIVER_CLASSIFIER.md)**

## Driver Categories

1. **AGGRESSIVE** - High risk scores, pushing limits, penalties & crashes
   - Examples: Verstappen, Tsunoda, Antonelli

2. **CONSISTENT** - Clean racing, solid points, reliable performance
   - Examples: Leclerc, Norris, Hamilton, Piastri

3. **STRUGGLING** - Poor results, high incidents, or off-pace
   - Examples: Sainz, Stroll

## Features Used

The classifier uses **7 features** across 3 data sources:

### Risk Score
- Penalties (5s/10s/drive-through/grid)
- Crashes (barrier hits, stops on track, red flags)
- Collisions, false starts, persistent track limits

### Teammate Performance (normalizes for car performance)
- **Points Delta**: Championship points vs teammate
- **Qualifying Delta**: Average gap in qualifying sessions
- **Position Delta**: Average finishing position difference

### Race Performance
- **Consistency**: Lap time variance (std dev)
- **Position Change**: Overtaking ability (grid vs finish)
- **Tyre Degradation**: Racecraft & tyre management

**Feature Importance:** PointsDelta (27.8%) > PositionDelta (16.8%) > RiskScore (16.7%)

## Files

### Core Files:
- `02_driver_classifier.py` - Main driver classification system
- `risk_score_simple.py` - Risk scoring with crash detection
- `teammate_performance.py` - Teammate performance comparison
- `01_data_exploration.py` - Initial data exploration & learning

### Output Files:
- `driver_classifications.json` - Classification results with confidence scores
- `teammate_performance.csv` - Teammate comparison metrics

### Documentation:
- `DRIVER_CLASSIFIER.md` - **Comprehensive technical documentation**
- `CRASH_DETECTION.md` - Crash detection system details
- `CLEANUP_SUMMARY.md` - Project cleanup history

## Running the System

```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the classifier (calculates risk scores + classifies drivers)
python 02_driver_classifier.py
```

## How It Works

1. **Risk Scoring** (`risk_score_simple.py`):
   - Parses race control messages from FastF1
   - Detects penalties (5s/10s/drive-through/grid)
   - Detects crashes (barrier hits, spins, stops on track)
   - Detects collisions (caused by driver)
   - Filters noise (blue flags, "NO FURTHER ACTION", procedural stuff)
   - Scores aligned with FIA penalty point system

2. **Teammate Performance** (`teammate_performance.py`):
   - Compares drivers within the same car
   - Calculates points, qualifying, and race pace deltas
   - Normalizes performance for fair cross-team comparison
   - Example: TSU -245 pts vs VER reveals struggles

3. **Race Performance Features** (`02_driver_classifier.py`):
   - Extracts lap times, positions, tyre data from races
   - Aggregates across season per driver
   - Calculates consistency, position change, tyre degradation

4. **Supervised Learning**:
   - Uses 8 seed labels (38% of 21 drivers)
   - Trains Random Forest classifier
   - Predicts remaining 13 drivers
   - PointsDelta is most important feature (27.8%)

**See [DRIVER_CLASSIFIER.md](DRIVER_CLASSIFIER.md) for complete technical details, limitations, and design decisions.**

## Key Design Principles

1. **Simple beats complex** - Clean, readable code over fragile regex
2. **Domain knowledge matters** - Aligned with F1/FIA penalty system
3. **Filter noise early** - Race control messages are 90% noise
4. **Fair comparisons** - Teammate performance normalizes for car differences
5. **Minimal pre-training** - Let the model learn (38% seed labels)
6. **Validate against reality** - Results should make F1 sense
7. **Use structured data** - FastF1's RacingNumber field over text parsing

## 2025 Season Results

### Top Risk Scores:
- TSU (Tsunoda): 37 points (1×5s, 4×10s penalties)
- SAI (Sainz): 36 points (3×10s penalties + 2 crashes)
- ALB (Albon): 24 points (3×10s penalties)
- ANT (Antonelli): 23 points (3×5s, 1×10s - rookie aggression)

### Clean Drivers:
- HAM (Hamilton): 0 points
- HAD (Hadjar): 0 points
- LEC (Leclerc): 5 points (1×5s)
- COL (Colapinto): 5 points (1×5s)

## Limitations

**Data Limitations:**
- Post-session steward decisions not captured in FastF1
- Grid penalties for future races not in race control messages
- Cannot perfectly distinguish driver error vs mechanical failure

**Design Trade-offs:**
- Impeding penalties filtered as procedural (not dangerous driving)
- All incidents weighted equally (no session/weather context)
- Small dataset (21 drivers) limits model generalization

**See [DRIVER_CLASSIFIER.md](DRIVER_CLASSIFIER.md) for detailed explanations and rationale.**

## Future Improvements

1. Include Sprint sessions (currently only Q + R)
2. Add overtake count from telemetry
3. Compare 2024 vs 2025 driver evolution
4. Weight incidents by session type and conditions
5. Interactive dashboard (Plotly/Streamlit)

## Data Source

Uses [FastF1](https://github.com/theOehrly/Fast-F1) - F1 telemetry and timing data API.

## License

MIT License - see LICENSE file
