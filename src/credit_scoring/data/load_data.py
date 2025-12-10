from __future__ import annotations
import pandas as pd
import openml

def load_openml_german_credit() -> tuple[pd.DataFrame, pd.Series]:
    ds = openml.datasets.get_dataset(31)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    y_bin = (y.astype(str).str.lower().str.contains("bad") | (y == 1)).astype(int)
    X = pd.DataFrame(X)
    return X, y_bin
