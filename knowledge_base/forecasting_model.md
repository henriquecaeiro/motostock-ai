# Demand Forecasting Model

## Prediction Target

The forecasting target is:

```text
quantity_sold
```

It represents the number of units sold for a specific product on a specific day.

## Data Granularity

The forecasting dataset uses the following granularity:

```text
one product per day
```

Each row represents the daily sales behavior of one product.

## Forecast Horizon

The primary forecast horizon is 14 days.

The system generates daily forecasts for the next 14 days and aggregates them to estimate total demand across the horizon.

## Lag Features

Lag features provide the model with previous sales values.

MotoStock AI uses:

- 1-day lag;
- 7-day lag;
- 14-day lag.

Conceptually:

```text
lag_1  = quantity sold one day earlier
lag_7  = quantity sold seven days earlier
lag_14 = quantity sold fourteen days earlier
```

These features help the model represent recent behavior and weekly patterns.

## Rolling Mean Features

Rolling means summarize recent historical demand.

MotoStock AI uses rolling windows of:

- 1 day;
- 7 days;
- 14 days.

Rolling means help smooth daily variation and represent the recent demand level.

Only historical information available before the predicted date should be used to calculate forecasting features.

## Calendar Features

Calendar features help the model represent time-related patterns.

They include deterministic calendar information used by the current pipeline, such as:

- day of the week;
- day of the month;
- month;
- weekend indicators.

Calendar features do not guarantee that the model will capture every seasonal or business event.

## Product Name as a Categorical Feature

`product_name` is used as a categorical variable.

This allows one forecasting model to learn differences between products while also using shared patterns from the complete dataset.

The model still depends on sufficient and representative historical data for each product.

## Recursive Forecasting

MotoStock AI uses recursive forecasting for multi-day predictions.

The process is:

1. generate the prediction for the first future day;
2. add that prediction to the temporary forecasting history;
3. recalculate the features required for the following day;
4. generate the next prediction;
5. repeat until the full forecast horizon is complete.

Because later predictions may depend on earlier predicted values, forecast error can accumulate across the horizon.

## Continuous Model Output

The model produces continuous numerical values.

For example, a daily prediction may be:

```text
2.47 units
```

This value represents an estimated level of demand. It is not immediately rounded at each recursive step.

## Non-Negative Forecasts

Negative sales demand is not meaningful for stock planning.

Negative daily predictions are therefore limited to zero:

```text
non_negative_forecast = max(0, raw_forecast)
```

The raw prediction may still be preserved for model analysis, while the non-negative value is used operationally.

## Horizon Aggregation and Rounding

Non-negative daily forecasts are summed across the complete horizon.

Rounding is applied only after the horizon has been aggregated. Daily recursive values should not be independently rounded before they are used to generate later features.

The recommendation process converts the aggregated forecast into whole units using its documented rule.

## Interpretation

Forecasts are estimates based on historical patterns.

They are not guarantees of future sales. Actual demand may differ because of promotions, supplier conditions, market changes, unusual events, data quality, or behavior not represented in the training history.
