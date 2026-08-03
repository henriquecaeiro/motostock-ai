import pandas as pd


def get_feature_importance(model_pipe):
    feature_names = model_pipe.named_steps["preprocess"].get_feature_names_out()
    importances = model_pipe.named_steps["model"].feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df
