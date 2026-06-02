from __future__ import annotations

from typing import Any, Dict, List, Mapping

import pandas as pd


def regression_feature_frame(values: Mapping[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    row = {c: values[c] for c in feature_columns}
    return pd.DataFrame([row])[feature_columns]
