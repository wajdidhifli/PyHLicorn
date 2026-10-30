"""Canonical Python values at pandas/Spark/JSON boundaries."""
import math
import numpy as np
import pandas as pd


def native(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def regulator_list(value):
    value = native(value)
    if value is None or (isinstance(value, str) and value == ''):
        return []
    if not isinstance(value, list):
        raise TypeError('Regulator groups must be lists, tuples, arrays, or missing.')
    return value


def normalize_frame(frame):
    frame = frame.copy()
    for column in ('coact', 'corep'):
        if column in frame:
            frame[column] = frame[column].map(regulator_list)
    for column in ('Coef_Acts', 'Coef_Reps'):
        if column in frame:
            frame[column] = frame[column].map(native)
    return frame
