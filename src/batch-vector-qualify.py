#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:14:38 2019

@author: jameno
"""
import pandas as pd
import vector_qualify
from multiprocessing.pool import ThreadPool
import time
import datetime as dt
import os
import time
from multiprocessing import Pool, cpu_count, get_context
import traceback
import sys
#%%

data_location = 'PHI-2019-11-13-csvData/'
file_list = os.listdir(data_location)

# Filter only .csv files
file_list = [filename for filename in file_list if '.csv' in filename]

chosen_donors_file = "PHI-all-qualify-stats-and-metadata-2019-11-21.csv"
donor_list = pd.read_csv(chosen_donors_file, low_memory=False)

#%%

def get_vector_summary(file_name, data_location, user_loc):

    file_path = data_location + file_name
    print(str(user_loc) + " STARTING")
    if((user_loc % 100 == 0) & (user_loc > 99)):
        print(user_loc)
        log_file = open('batch-vector-qualify-log.txt', 'a')
        log_file.write(str(user_loc)+"\n")
        log_file.close()

    import_status = ""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        import_status = "Succesfully Imported"

    except Exception as e:
        df = pd.DataFrame()
        print("Failed to import: " + file_path)
        import_status = "Failed to Import - " + str(e)

    vector_stats, vector_summary = vector_qualify.get_vector_summary(df, file_name)
    vector_summary['pandas_import_status'] = import_status
    print(str(user_loc) + " QUALIFICATION COMPLETE!")
    return vector_summary


# %%
if __name__ == "__main__":
    # Start Pipeline
    start_time = time.time()

    # Startup CPU multiprocessing pool
    pool = Pool(int(cpu_count()))

    pool_array = [pool.apply_async(
            get_vector_summary,
            args=[file_list[user_loc],
                  data_location,
                  user_loc
                  ]
            ) for user_loc in range(len(file_list))]

    pool.close()
    pool.join()

    end_time = time.time()
    elapsed_minutes = (end_time - start_time)/60
    elapsed_time_message = "Vector Qualify completed in: " + \
        str(elapsed_minutes) + " minutes\n"
    print(elapsed_time_message)
    log_file = open('batch-vector-qualify-log.txt', 'a')
    log_file.write(str(elapsed_time_message)+"\n")
    log_file.close()

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

    # %%
    # Convert results into dataframe
    vector_summary_df = pd.concat(results_array, sort=False)
    missing_columns = list(set(list(donor_list)) - set(list(vector_summary_df)))
    vector_summary_df = pd.merge(vector_summary_df, donor_list[['file_name'] + missing_columns], how='left', on='file_name')
    today_timestamp = dt.datetime.now().strftime("%Y-%m-%d")
    vector_export_filename = 'PHI-batch-vector-qualify-' + today_timestamp + '.csv'
    vector_summary_df.to_csv(vector_export_filename, index=False)
