import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE should not be calculated when real demand is zero
    non_zero_mask = y_true > 0

    if non_zero_mask.sum() == 0:
        mape = np.nan
    else:
        mape = (
            np.mean(
                np.abs(
                    (y_true[non_zero_mask] - y_pred[non_zero_mask])
                    / y_true[non_zero_mask]
                )
            )
            * 100
        )

    # WAPE is usually better for low-demand forecasting
    if np.sum(np.abs(y_true)) == 0:
        wape = np.nan
    else:
        wape = (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape}


def error_by_product(product_df, prediction_col, model_name):
    y_true = product_df["quantity_sold"].to_numpy()
    y_pred = product_df[prediction_col].to_numpy()

    absolute_error = np.abs(y_true - y_pred)
    mae = np.mean(absolute_error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    non_zero_mask = y_true != 0

    if non_zero_mask.sum() == 0:
        mape = np.nan
    else:
        mape = (
            np.mean(
                np.abs(
                    (y_true[non_zero_mask] - y_pred[non_zero_mask])
                    / y_true[non_zero_mask]
                )
            )
            * 100
        )

    total_actual_quantity = y_true.sum()
    total_predicted_quantity = y_pred.sum()

    difference = total_predicted_quantity - total_actual_quantity

    if difference > 0:
        error_direction = "Overprediction"
    elif difference < 0:
        error_direction = "Underprediction"
    else:
        error_direction = "Balanced"

    return {
        "product_name": product_df["product_name"].iloc[0],
        "model_name": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "total_actual_quantity": total_actual_quantity,
        "total_predicted_quantity": total_predicted_quantity,
        "error_direction": error_direction,
    }
