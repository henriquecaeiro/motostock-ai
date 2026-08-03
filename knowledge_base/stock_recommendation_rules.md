# Stock Recommendation Rules

## Overview

The MotoStock AI recommendation engine converts demand forecasts into structured stock replenishment recommendations.

The rules are deterministic and are applied after the daily demand forecast has been generated.

## Forecasted Demand Units

Daily negative forecasts are first limited to zero.

The non-negative daily forecasts are then summed across the complete forecast horizon.

```text
forecasted_demand_units
= ceil(sum of non-negative daily forecasts)
```

The ceiling operation converts the aggregated continuous forecast into whole units and avoids rounding expected demand downward.

## Safety Stock

Safety stock provides a buffer against forecast error and demand uncertainty.

MotoStock AI currently uses 20% of forecasted demand:

```text
safety_stock
= ceil(forecasted_demand_units × 20%)
```

Safety stock reduces stockout exposure, but it does not guarantee that all future demand will be covered.

## Required Stock

Required stock combines forecasted demand and safety stock:

```text
required_stock
= forecasted_demand_units + safety_stock
```

This represents the stock level the recommendation engine considers appropriate for the forecast horizon under the current business rules.

## Recommended Purchase Quantity

The recommended purchase quantity compares required stock with current stock:

```text
recommended_purchase_quantity
= max(0, required_stock - current_stock)
```

The result cannot be negative.

When current stock already covers required stock, the purchase recommendation is zero.

## Stock Status

### Critical

A product is classified as `critical` when current stock is below forecasted demand.

```text
current_stock < forecasted_demand_units
```

The current inventory may not cover expected demand for the forecast horizon.

### Warning

A product is classified as `warning` when current stock covers forecasted demand but does not cover forecasted demand plus safety stock.

```text
forecasted_demand_units
≤ current_stock
< required_stock
```

Expected demand may be covered, but the risk buffer is incomplete.

### Healthy

A product is classified as `healthy` when current stock covers the required stock level.

```text
current_stock ≥ required_stock
```

A healthy classification does not guarantee that future demand will be covered. It means that current stock satisfies the documented forecast and safety-stock rule.

### Overstock

A product is classified as `overstock` when current stock exceeds 1.5 times the required stock.

```text
current_stock > required_stock × 1.5
```

This status indicates that inventory may be significantly higher than the current calculated need.

## Supplier Lead Time

Supplier lead time represents how long replenishment is expected to take.

Lead time does not replace the required-stock or purchase-quantity formulas. It is used as an urgency factor when recommendations are prioritized.

A product with a longer replenishment time may need earlier attention because a new order will take longer to become available.

Supplier lead time is an operational estimate, not a guaranteed delivery date.

## Priority Score

The recommendation engine assigns a `priority_score` to help rank products for review.

The priority score represents urgency. Higher scores indicate that a product should generally receive earlier attention.

The score uses recommendation context, including supplier lead time and stock urgency. It supports ordering and review, but it does not replace the recommended purchase quantity.

Users should evaluate the priority score together with:

- stock status;
- recommended purchase quantity;
- supplier availability;
- budget;
- operational knowledge.

## Decision-Support Limitation

Stock recommendations are calculated from forecasts and documented business rules.

They are decision-support outputs, not automatic purchase orders or guarantees.
