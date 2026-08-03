# Limitations and Data Access Rules

## Forecasts are not guarantees

Demand forecasts estimate future sales from historical patterns. Actual demand
may be higher or lower. Forecast output must not be presented as a guaranteed
future value.

## Recursive error accumulation

MotoStock AI uses recursive multi-day forecasting. Earlier future predictions
can be used to build features for later days, so uncertainty can increase over a
longer horizon.

## Cold start and data quality

New products may not have enough history for reliable lags, rolling means or
product-specific behavior. Missing sales, duplicate records, incorrect
quantities, inconsistent product names, unrecorded stockouts and incomplete
promotion data can reduce model quality. The model cannot recover information
that was never captured.

## Promotions and unusual events

Promotions, price changes, local events, weather changes, supplier shortages,
competitor activity, regulatory changes and sudden customer behavior may not be
represented in the current features.

## Product-level underprediction

Aggregate metrics can hide systematic errors for individual products. The
evaluated `Bag Delivery 45L` example had a meaningful XGBoost underprediction.
Product-level evaluation, safety stock and human review remain necessary.

## Static knowledge versus current data

The RAG knowledge base contains curated documentation: definitions,
architecture, forecasting methodology, model evaluation, recommendation rules,
limitations and glossary terms. It must not be treated as a source of current
operational values.

Current forecasts, inventory, purchase quantities and recommendation statuses
must come from the allowlisted live-data tools and application services. The
assistant must not use a static document snapshot to answer a current-data
question.

## Assistant behavior when data is unavailable

If the repository, forecasting model, tool service, embedding model or Ollama
service is unavailable, the API returns a controlled error or states that it
cannot verify the requested value. It must not invent products, quantities,
forecasts, statuses or recommendations.

## Decision-support limitation

MotoStock AI supports decisions. Final purchasing and inventory decisions must
also consider supplier conditions, budget, promotions, operational context and
human judgment.
