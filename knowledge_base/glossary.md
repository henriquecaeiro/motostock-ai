# MotoStock AI Glossary

## Demand Forecasting

Demand forecasting is the process of estimating how many units of a product may be sold during a future period.

Forecasts are estimates based on available data, not guarantees.

## Lag Feature

A lag feature represents a previous value of a time series.

For example, a 7-day lag for `quantity_sold` contains the sales value recorded seven days earlier.

Lag features help a model learn recent and repeating historical patterns.

## Rolling Mean

A rolling mean is the average of recent historical observations within a moving window.

For example, a 7-day rolling mean summarizes the recent average demand across seven days.

## Safety Stock

Safety stock is additional inventory maintained as a buffer against uncertainty, forecast error, unexpected demand, or replenishment delays.

In MotoStock AI, safety stock is currently calculated as 20% of forecasted demand, rounded upward.

## Required Stock

Required stock is the stock level calculated to cover forecasted demand plus safety stock.

```text
required_stock
= forecasted_demand_units + safety_stock
```

## Lead Time

Lead time is the expected amount of time between placing a replenishment order and receiving the product.

Lead time is an estimate and should not be treated as a guaranteed delivery date.

## Stockout

A stockout occurs when demand exists for a product but the product is not available in inventory.

Stockouts may cause lost sales and customer dissatisfaction.

## Overstock

Overstock occurs when inventory is significantly higher than the calculated or expected need.

Overstock may tie up capital, increase storage costs, and reduce cash flow.

## Cold Start

Cold start is the difficulty of producing reliable predictions for a new product with little or no historical data.

Without sufficient history, the model cannot build reliable lag, rolling, or product-specific patterns.

## MAE

Mean Absolute Error measures the average absolute difference between predicted and actual values.

Lower MAE generally indicates smaller average prediction errors.

## RMSE

Root Mean Squared Error measures prediction error while giving greater weight to larger errors.

Lower RMSE generally indicates fewer or smaller large errors.

## MAPE

Mean Absolute Percentage Error expresses prediction error as a percentage of actual values.

MAPE may be unstable when actual values are zero or very small.

## RAG

Retrieval-Augmented Generation is an approach that retrieves relevant documents before asking a language model to generate an answer.

RAG helps ground answers in curated project knowledge.

## Embedding

An embedding is a numerical vector that represents the semantic meaning of a piece of text.

Texts with similar meanings should generally have embeddings that are close in vector space.

## Vector Similarity

Vector similarity measures how close two embeddings are.

A vector search uses this similarity to retrieve document chunks that are semantically related to a user question.
