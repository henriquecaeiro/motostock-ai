# MotoStock AI Business Problem

## Business Context

Motorcycle and delivery gear retailers need to maintain enough inventory to serve customers without purchasing more products than necessary.

Demand varies by product and over time. Items such as helmets, delivery bags, rain gear, and motorcycle accessories may have different sales patterns, seasonality, and replenishment needs.

Without analytical support, purchasing decisions may depend too heavily on manual judgment, recent sales, or incomplete information.

## Stockout Risk

A stockout occurs when customer demand exists but the product is unavailable.

Stockouts can cause:

- lost sales;
- dissatisfied customers;
- delayed deliveries;
- reduced customer trust;
- customers purchasing from competitors.

Products with unstable demand or long supplier lead times may require additional attention because they cannot always be replenished quickly.

## Overstock Risk

Overstock occurs when the store keeps significantly more inventory than it is expected to need.

Excess inventory can cause:

- capital to remain tied up in unsold products;
- higher storage costs;
- reduced cash flow;
- product deterioration or obsolescence;
- unnecessary purchasing expenses.

The objective is not to maximize inventory. It is to balance product availability with inventory cost and operational risk.

## Forecasting Objective

The forecasting component estimates how many units of each product may be sold during a future period.

MotoStock AI currently uses a primary forecasting horizon of 14 days.

Forecasts provide estimates based on historical sales patterns. They are inputs for business decisions, not guarantees of future sales.

## Recommendation Objective

The stock recommendation component combines:

- forecasted demand;
- current stock;
- safety stock;
- supplier lead time;
- business priority rules.

Its objective is to estimate whether a product requires replenishment and how many units may need to be purchased.

The recommendation engine does not replace business judgment. Managers should also consider supplier availability, budget, promotions, operational knowledge, and unexpected market events.

## Target Users

MotoStock AI is designed to support:

- store owners;
- inventory managers;
- purchasing teams;
- operations analysts;
- employees responsible for stock replenishment.

## Decision-Support Role

MotoStock AI is a decision-support system.

It provides forecasts, stock indicators, and replenishment recommendations to help users make more informed decisions. It does not make autonomous purchasing decisions and does not guarantee business outcomes.
