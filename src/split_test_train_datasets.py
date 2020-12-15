#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split Test/Train Anonymized Datasets
==========================================
:File: split_test_train_datasets.py
:Description: Splits up anonymized datasets into a pre-defined lenght of train/
    test sizes
:Version: 0.0.1
:Created: 2019-09-22
:Authors: Jason Meno (jam)
:Last Modified: 2019-09-26 (jam)
:Dependencies:
    - A Tidepool dataset to qualify
:License: BSD-2-Clause
"""
# %% Import Dependencies
import pandas as pd
import numpy as np
import os
import time
from datetime import timedelta
import traceback
import sys

# %% Functions

def find_split_date(df, test_size):
    df['date'] = df['time'].str.split('T', expand=True)[0]
    typesPerDay = pd.DataFrame(
            df.groupby('date').apply(lambda x: np.unique(list(x['type']))),
                    columns=['type_list']
                    ).reset_index()

    data_types = ['cbg', 'bolus', 'basal']

    for data_type in data_types:
        typesPerDay[data_type] = \
            typesPerDay['type_list'].apply(lambda x: data_type in x)

    typesPerDay['cgm+pump'] = typesPerDay['cbg'] & typesPerDay['bolus'] & typesPerDay['basal']

    cgmPumpDayStart = typesPerDay.loc[typesPerDay['cgm+pump'] == True, 'date'].min()
    cgmPumpDayEnd = typesPerDay.loc[typesPerDay['cgm+pump'] == True, 'date'].max()

    cgm_pump_date_range = pd.DataFrame(
                            pd.date_range(
                                cgmPumpDayStart,
                                cgmPumpDayEnd),
                                columns=['date']
                                ).astype(str)

    cgm_pump_date_range = \
        pd.merge(cgm_pump_date_range, typesPerDay, on='date', how='left')

    cgm_pump_date_range['cgm_pump_rolling_sum'] = \
        cgm_pump_date_range['cgm+pump'].rolling(test_size, min_periods=1).apply(lambda x: x.sum(), raw=False)

    # The best split date is the last date in an N-day window
    # which contains >= 80% of those days worth of cgm+pump data

    min_days_with_data = 0.8 * test_size

    best_end_date = \
        cgm_pump_date_range.loc[cgm_pump_date_range['cgm_pump_rolling_sum'] >= min_days_with_data, 'date'].max()

    # The subtracted timedelta gives a midnight split date,
    # so subtracting one from test_days gives us the next day's midnight
    # as the proper split

    best_split_date = (pd.to_datetime(best_end_date) - timedelta(days=test_size-1)).strftime("%Y-%m-%d")

    df.drop(columns='date', inplace=True)

    return best_split_date


def split_dataset(data_df,
                  test_days):
    """Split dataset into train/test"""


    # Keep all cgm & pump settings for both datasets
    cgmPumpSettings = data_df[((data_df.type == "cgmSettings") |
                           (data_df.type == "pumpSettings"))]

    remaining_df = data_df[~((data_df.type == "cgmSettings") |
                           (data_df.type == "pumpSettings"))]

    split_date = find_split_date(data_df, test_days)

    train_df = remaining_df[remaining_df["time"] < split_date]
    test_df = remaining_df[remaining_df["time"] >= split_date]

    train_df = pd.concat([cgmPumpSettings, train_df])
    test_df = pd.concat([cgmPumpSettings, test_df])

    return train_df, test_df, split_date

