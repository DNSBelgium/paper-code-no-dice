from gamblingfinder.visualize import make_plots_binary
import pandas as pd


def model(dbt, _session):
    predictions = dbt.ref("lr_g_train_test").df()

    make_plots_binary(predictions, file_prefix="lr_binary")

    return pd.DataFrame({"ok": [True]})
