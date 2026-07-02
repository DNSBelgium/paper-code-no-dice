from gamblingfinder.visualize import make_plots_multi
import pandas as pd


def model(dbt, _session):
    predictions = dbt.ref("lr_chained_test").df()

    make_plots_multi(predictions, file_prefix="lr_chained", title_appendix="- Chained Models")

    return pd.DataFrame({"ok": [True]})
