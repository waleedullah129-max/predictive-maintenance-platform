# Technical Presentation - Predictive Maintenance Platform

## 1. Problem Statement
Predict Remaining Useful Life (RUL) of turbofan jet engines from sensor
telemetry, enabling proactive maintenance instead of fixed-schedule or
reactive repairs. Based on NASA C-MAPSS FD001 (100 train + 100 test engines).

## 2. Approach
- Engine-grouped train/validation splits (GroupKFold) to prevent leakage
- 72 engineered features: rolling mean/std/trend per sensor + domain features
- XGBoost regressor, RUL clipped at 125 cycles (standard C-MAPSS practice)
- Hyperparameters selected via grid search, validated against CV baseline

## 3. Key Results
| Metric | Value |
|---|---|
| CV RMSE | 16.25 +/- 0.69 |
| Test RMSE (last cycle per engine) | 16.92 |
| Test NASA Score | 522.9 |

## 4. Two Leakage Bugs Found and Fixed
1. RUL_clipped (the target) was accidentally included as a feature -- RMSE
   dropped to an impossible 0.6. Fixed by locking feature columns before
   creating any target-derived column.
2. utilization_rate was engineered directly from RUL, algebraically
   encoding the answer. Fixed by removing the feature.

## 5. Enterprise Platform Modules Built
- Explainability: SHAP global + per-engine explanations
- Health Score: RUL rescaled to 0-100 with 4 risk tiers
- Failure Classification: rule-based (documented limitation)
- Maintenance Recommendation Engine: Health Score + Failure Mode -> action
- Alert Center: automated SEVERE/WARNING alerts for at-risk engines
- FastAPI: REST API for real-time prediction
- Streamlit Dashboard: fleet overview + live CSV upload -> prediction
- MLflow: experiment tracking (params, metrics, model artifacts)
- Drift Detection: KS-test based distribution monitoring

## 6. Known Limitations
- Single dataset (FD001) -- one operating condition, one fault mode
- Failure classification is heuristic, not a trained classifier
- SQLite MLflow backend (single-machine, not multi-user)

## 7. Future Work
- Extend to FD002-FD004 for real multi-class failure classification
- LSTM/CNN sequence models as an alternative to windowed features
- Automated retraining triggered by drift detection thresholds
- Production-grade deployment (auth, hosted MLflow, containerization)
