# Limitations and Data Access Rules

## Forecasts Are Not Guarantees

Demand forecasts estimate future sales from historical patterns.

Actual demand may be higher or lower than predicted. Forecast outputs must not be presented as guaranteed future values.

## Recursive Error Accumulation

MotoStock AI uses recursive multi-day forecasting.

Predictions for earlier future days may be used to construct features for later future days. As a result, forecasting error can accumulate across the horizon.

Longer recursive horizons may contain greater uncertainty than the first predicted days.

## Cold Start

New products may have little or no sales history.

This is known as the cold-start problem.

Without sufficient historical data, lag features, rolling means, seasonality, and product-specific behavior cannot be estimated reliably.

Forecasts for new products require additional business judgment or alternative strategies.

## Historical Data Quality

Model quality depends on the quality of the historical data.

Problems such as the following may reduce forecasting reliability:

- missing sales;
- duplicated records;
- incorrect quantities;
- inconsistent product names;
- unrecorded stockouts;
- changes in sales processes;
- incomplete promotion information.

The model cannot recover information that was never captured correctly.

## Promotions and Unexpected Events

Demand may change because of events that are not represented in historical features.

Examples include:

- promotions;
- price changes;
- local events;
- weather changes;
- supplier shortages;
- competitor activity;
- regulatory changes;
- sudden changes in customer behavior.

The current model may not anticipate these events unless relevant information is included in the forecasting data.

## Product-Specific Underprediction

Aggregate metrics can hide errors for individual products.

The evaluated `Bag Delivery 45L` example showed meaningful underprediction. Similar behavior may occur for other products.

This limitation supports:

- product-level evaluation;
- safety stock;
- monitoring;
- business review of recommendations.

## Static Knowledge and Current Data

The RAG knowledge base contains curated static documentation.

It may contain:

- business definitions;
- architecture descriptions;
- forecasting methodology;
- historical model evaluation;
- stock recommendation rules;
- known limitations;
- glossary terms.

It must not be treated as a source of current operational values.

## Mandatory Current-Data Rule

Current forecasts, inventory values, purchase quantities and current recommendations must come from live-data tools, not from the knowledge base.

The knowledge base must not contain snapshots that the language model could mistake for current operational data.

## RAG Responsibilities

RAG should answer questions such as:

- What is safety stock?
- How does the recommendation formula work?
- What is the purpose of XGBoost in MotoStock AI?
- What are the known forecasting limitations?
- Why was XGBoost selected?

These are questions about documented and relatively stable knowledge.

## Tool Responsibilities

Live-data tools should answer questions such as:

- Which products are currently critical?
- What is the current stock of a product?
- What is the latest 14-day forecast?
- How many units should be purchased today?
- Which product currently has the highest priority?
- What recommendations were generated most recently?

Until live-data tools are implemented and successfully called, the assistant must explicitly state that it cannot access those current values.

It must not invent plausible products, quantities, forecasts, statuses, or recommendations.

## Decision-Support Limitation

MotoStock AI supports decisions.

Final purchasing and inventory decisions should also consider business context, supplier conditions, available budget, promotions, and operational judgment.
