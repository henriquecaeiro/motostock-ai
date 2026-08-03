# Forecasting Model Evaluation

## Evaluation Purpose

MotoStock AI compared Random Forest and XGBoost as candidate demand forecasting models.

The evaluation used aggregate error metrics to compare how closely each model's predictions matched observed demand.

## Aggregate Results

### Random Forest

| Metric | Result |
|---|---:|
| MAE | 1.321885 |
| RMSE | 1.849646 |
| MAPE | 63.501327% |
| WAPE | 81.850435% |

### XGBoost

| Metric | Result |
|---|---:|
| MAE | 1.296611 |
| RMSE | 1.869340 |
| MAPE | 62.575466% |
| WAPE | 80.285498% |

## Model Selection

XGBoost was selected as the primary forecasting model because it achieved:

- the lower MAE;
- the lower MAPE.
- the lower WAPE.

Random Forest achieved a slightly lower RMSE:

```text
Random Forest RMSE: 1.849646
XGBoost RMSE:       1.869340
```

The difference between the models was small.

Selecting XGBoost does not mean that it is perfect or that it will be the best model for every product and every future period.

## Metric Interpretation

### MAE

Mean Absolute Error measures the average absolute difference between predicted and actual values.

A lower MAE indicates that predictions are closer to actual values on average.

### RMSE

Root Mean Squared Error gives greater weight to larger errors.

A lower RMSE generally indicates fewer or smaller large forecasting errors.

### MAPE

Mean Absolute Percentage Error expresses error as a percentage of the actual value.

MAPE can become unstable or difficult to interpret when actual demand is zero or very small.

## Aggregate Metrics Are Not Sufficient

Aggregate metrics summarize performance across many observations.

They can hide:

- weak performance for specific products;
- systematic underprediction;
- systematic overprediction;
- different error patterns across high-volume and low-volume products;
- periods with unusually large errors.

For this reason, evaluation should also be performed by product and by forecast horizon.

## Bag Delivery 45L Limitation

A product-level analysis identified underprediction for `Bag Delivery 45L`.

```text
Actual total:    402.00 units
Predicted total: 338.206726 units
Underprediction:  63.793274 units
```

The model predicted 63.793274 fewer units than the observed total for the evaluated period.

This example shows that acceptable aggregate metrics do not guarantee equally reliable predictions for every product.

## Operational Implications

Product-level underprediction can increase stockout risk.

This limitation supports the use of:

- safety stock;
- product-level evaluation;
- continuous model monitoring;
- business review of important recommendations;
- future model improvement.

Safety stock reduces operational exposure to forecast error, but it does not eliminate uncertainty.

Model predictions and recommendation outputs must remain decision-support inputs rather than guaranteed outcomes.
