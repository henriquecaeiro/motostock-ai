import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(y_true, y_pred):
    """Return the same forecast metrics used by all model notebooks."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Zero targets are excluded because percentage error is undefined for them.
    non_zero_mask = y_true > 0
    if non_zero_mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(
            np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask])
                / y_true[non_zero_mask]
            )
        ) * 100

    if np.sum(np.abs(y_true)) == 0:
        wape = np.nan
    else:
        wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape}


def error_by_product(product_df, prediction_col, model_name):
    """Summarize forecast errors for one product and one model."""

    y_true = product_df["quantity_sold"].to_numpy()
    y_pred = product_df[prediction_col].to_numpy()
    metrics = evaluate_forecast(y_true, y_pred)

    total_actual_quantity = y_true.sum()
    total_predicted_quantity = y_pred.sum()

    # Positive signed error means overprediction; negative means underprediction.
    signed_total_error = total_predicted_quantity - total_actual_quantity
    if signed_total_error > 0:
        error_direction = "Overprediction"
    elif signed_total_error < 0:
        error_direction = "Underprediction"
    else:
        error_direction = "Balanced"

    return {
        "product_name": product_df["product_name"].iloc[0],
        "model_name": model_name,
        **metrics,
        "total_actual_quantity": total_actual_quantity,
        "total_predicted_quantity": total_predicted_quantity,
        "signed_total_error": signed_total_error,
        "error_direction": error_direction,
    }
