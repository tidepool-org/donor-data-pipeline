#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:14:38 2019

@author: jameno
"""
import pandas as pd
import numpy as np
from multiprocessing.pool import ThreadPool
import time
import datetime as dt
import os
import time
from multiprocessing import Pool, cpu_count, get_context
import traceback
import sys

#%%
data_location = 'PHI-2019-09-05-csvData/'
file_list = os.listdir(data_location)

# Filter only .csv files
file_list = [filename for filename in file_list if '.csv' in filename]
#%%

def get_unique_length_values(data):
    """Get the unique lengths of items in each column and their values

    Useful for finding changes in dataframes after processing

    """

    unique_len_values = pd.DataFrame([], index=[np.arange(1000)])
    for col in list(data):
        # print(col)

        unique_lengths = \
            data[col].\
            astype(str).\
            str.\
            len().\
            drop_duplicates().\
            sort_values(ascending=True)

        col_vals_by_len_and_type = \
            data.loc[unique_lengths.index, col].astype(str).values + \
            ' <<' + \
            data.loc[unique_lengths.index, 'type'].astype(str).values + \
            '>>'

        if(len(unique_lengths) > 1000):
            print("UNIQUE LENGTHS OVER 1000!!!!")
            unique_lengths = unique_lengths[:1000]
            col_vals_by_len_and_type = col_vals_by_len_and_type[:1000]

        unique_len_values.loc[range(len(col_vals_by_len_and_type)),
                              col +
                              " - " +
                              "LEN"
                              ] = unique_lengths.values

        unique_len_values.loc[range(len(col_vals_by_len_and_type)),
                              col] = col_vals_by_len_and_type

    return unique_len_values


def batch_get_unique_lengths(file_name, data_location, user_loc):

    file_path = data_location + file_name

    if(user_loc % 100 == 0):
        print(user_loc)

    try:
        data = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        data = pd.DataFrame()
        print("Failed to import: " + file_path)

    unique_dataframe_length_values = get_unique_length_values(data)

    return unique_dataframe_length_values

# %%
if __name__ == "__main__":
    # Start Pipeline

    start_time = time.time()

    # Startup CPU multiprocessing pool
    pool = Pool(int(cpu_count()))

    pool_array = [pool.apply_async(
            batch_get_unique_lengths,
            args=[file_list[user_loc],
                  data_location,
                  user_loc
                  ]
            ) for user_loc in range(len(file_list))]

    pool.close()
    pool.join()

    end_time = time.time()
    elapsed_minutes = (end_time - start_time)/60
    elapsed_time_message = "Unique dataframe collections completed in: " + \
        str(elapsed_minutes) + " minutes\n"
    print(elapsed_time_message)

    # %% Append results of each pool into an array

    results_array = []

    for result_loc in range(len(pool_array)):
        try:
            results_array.append(pool_array[result_loc].get())
        except Exception as e:
            print('Failed to get results! ' + str(e))
            exception_text = traceback.format_exception(*sys.exc_info())
            print('\nException Text:\n')
            for text_string in exception_text:
                print(text_string)

    print("Finished appending results")
# %%
    # Convert results into dataframe
    all_dataframes_df = pd.concat(results_array, sort=False)

# %%
    print("Dropping duplicates...", end="")
    # Drop duplicate df_changes and reorder and sort
    all_dataframes_df = \
        all_dataframes_df.\
        reindex(sorted(all_dataframes_df.columns), axis=1).\
        reset_index(drop=True)

    for col_idx in range(0, len(list(all_dataframes_df)), 2):
        value_col = list(all_dataframes_df)[col_idx]
        len_col = list(all_dataframes_df)[col_idx+1]

        non_null_length_loc = all_dataframes_df[len_col].notnull()
        dropped_and_sorted = \
            all_dataframes_df.loc[non_null_length_loc, [value_col, len_col]].\
            drop_duplicates(len_col).\
            sort_values(by=len_col).\
            reset_index(drop=True)

        all_dataframes_df[[value_col, len_col]] = np.nan
        all_dataframes_df.loc[dropped_and_sorted.index, [value_col, len_col]] = \
            dropped_and_sorted

    # Drop all rows that are the same (only nans)
    all_dataframes_df.drop_duplicates(keep=False, inplace=True)
    # Drop all length columns
    all_dataframes_df.drop(list(all_dataframes_df)[1::2], axis=1, inplace=True)

    print("done")
# %%
    today_timestamp = dt.datetime.now().strftime("%Y-%m-%d")
    all_dataframes_df.to_csv('PHI-batch_all_dataframe_value_lengths' + today_timestamp + '.csv', index=False)
    print("Export Complete!")