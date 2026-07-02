from gamblingfinder.visualize import make_plots_multi
import pandas as pd


def model(dbt, _session):
    predictions = dbt.ref("lr_multi_train_test").df()

    make_plots_multi(predictions, file_prefix="lr_multi", title_appendix="- Multiclass Model")

    return pd.DataFrame({"ok": [True]})
