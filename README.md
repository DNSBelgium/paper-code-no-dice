# Code and data for paper "No Dice: Detecting Illegal Gambling Websites in a Top-Level Domain"

This repository contains code and data alongside the paper "No Dice: Detecting Illegal Gambling Websites in a Top-Level Domain", currently under submission at KDD 2027.

The code is structured as a [dbt](https://docs.getdbt.com/docs/introduction?version=2.0&name=Fusion) pipeline. To run it, create a new Python virtual environment and execute the following steps:

```
pip install -e ./python
dbt deps
dbt build
```

The resulting figures can be found in the `plots/` directory.

## Data

The file `data/ground_truth.parquet` contains the features of the train and test websites, specifically the text embeddings and numerical crawler features. The label is `0` for non-gambling websites, `1` for gambling operators, and `2` for gambling link sites. Similarly, `data/illegal_outside_be.parquet` contains the features of known illegal gambling sites outside the .be zone.