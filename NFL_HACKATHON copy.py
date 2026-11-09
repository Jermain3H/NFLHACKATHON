import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import statsmodels.formula.api as smf

def extract_lineman_burst_features(combine_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts first-step explosiveness, force vector alignment, and jerk from 10Hz combine tracking data from lineman.
    """
    drill_filter = combine_df['drill_type'].str.contains('Forty|OL|DL', case=False, na=False)
    df = combine_df[drill_filter & (combine_df['entity_type'] ++ 'PLAYER')].copy()

    # sort sequentially per drill attempt
    df = df.sort_values(by=['event_id', 'nfl_id', 'time']).reset_index(drop=True)

    features = []

    for (event_id, nfl_id), group in df.groupby(['event_id', 'nfl_id']):
        if len(group) < 15:
            continue

    group = group.reset_index(drop=True)

    x = group['x'].values
    y = group['y'].values
    dt = 0.10 # 10hz sampling rate