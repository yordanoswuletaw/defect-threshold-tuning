# Defect Detection Threshold Tuning Analysis

This project analyzes defect detection model performance and determines optimal confidence thresholds using various metrics. It includes comprehensive analysis of prediction patterns, spatial distribution, and polygon characteristics.

## Project Structure

```
.
├── data/                     # Data directory
│   ├── anno_df.csv           # Ground truth annotations
│   ├── pred_df.csv           # Model predictions
│   ├── polygon_matches.csv    # Matched predictions and annotations
│   ├── image_metrics.csv      # Per-image performance metrics
│   └── threshold_metrics_10.csv  # Performance metrics at 10 thresholds
│   └── threshold_metrics_100.csv  # Performance metrics at 100 thresholds
│   └── threshold_metrics_1000.csv  # Performance metrics at 1000 thresholds
├── src/                      # Source code directory
│   ├── scripts/               # Analysis scripts package
│   │   ├── __init__.py
│   │   ├── analyze_confidence.py
│   │   ├── analyze_pr_curve.py
│   │   ├── analyze_polygon_areas.py
│   │   └── polygons_comparator.py
│   └── eda.ipynb              # Main analysis notebook
├── pyproject.toml            # Poetry dependency configuration
└── README.md                 # Project documentation
```

## Setup

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install project dependencies:
   ```bash
   poetry install
   ```

3. Activate the Poetry shell:
   ```bash
   poetry shell
   ```

4. Data Files in the `data/` directory:
   - Input Data:
     * `anno_df.csv`: Ground truth annotations with polygon coordinates
     * `pred_df.csv`: Model predictions with confidence scores
   - Generated Analysis Files:
     * `polygon_matches.csv`: Results of matching predictions to ground truth
     * `image_metrics.csv`: Performance metrics calculated per image
     * `threshold_metrics_10.csv`: Model performance at 10 confidence thresholds
     * `threshold_metrics_100.csv`: Model performance at 100 confidence thresholds
     * `threshold_metrics_1000.csv`: Model performance at 1000 confidence thresholds

## Analysis Overview

The main analysis is contained in `eda.ipynb`, which includes:

1. Data Exploration
   - Dataset statistics and characteristics
   - Class distribution analysis
   - Confidence score distribution

2. Threshold Analysis
   - Precision-Recall curve analysis
   - F1 score optimization
   - Optimal threshold determination

3. Spatial Analysis
   - Polygon size distribution
   - Spatial clustering patterns
   - Over-prediction analysis

4. Performance Metrics
   - IoU (Intersection over Union) analysis
   - Class-wise performance
   - Image-level statistics

## Key Scripts

- `analyze_confidence.py`: Analyzes confidence score distributions and their relationship with prediction accuracy
- `analyze_pr_curve.py`: Generates precision-recall curves and finds optimal thresholds
- `analyze_polygon_areas.py`: Compares polygon areas between predictions and annotations
- `polygons_comparator.py`: Matches predictions to ground truth annotations

## Results

The analysis provides:
- Optimal confidence threshold for defect detection
- Detailed performance metrics at various thresholds
- Insights into model behavior and potential improvements
- Recommendations for model retraining

## Requirements

- Python 3.10+
- pandas >= 2.2.3
- numpy >= 2.2.5
- matplotlib >= 3.10.1
- seaborn >= 0.13.2
- scikit-learn >= 1.6.1
- shapely >= 2.1.0
- scipy >= 1.15.2
- tqdm >= 4.67.1

## Usage

1. Start with the Jupyter notebook (make sure you're in the Poetry shell):
   ```bash
   poetry run jupyter notebook eda.ipynb
   ```

2. Run individual analysis scripts:
   ```bash
   poetry run python src/scripts/analyze_confidence.py
   poetry run python src/scripts/analyze_pr_curve.py
   poetry run python src/scripts/analyze_polygon_areas.py
   ```

## License

MIT
