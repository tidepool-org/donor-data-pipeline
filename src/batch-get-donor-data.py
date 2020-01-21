#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:14:38 2019

@author: jameno
"""
import pandas as pd
from multiprocessing.pool import ThreadPool
import get_single_tidepool_dataset
import time
import datetime as dt
import os
import time
from multiprocessing import Pool, cpu_count, get_context
import traceback
import sys
#%%

def download_data(userid, donor_group, user_loc, csv_dir):
    if(os.path.exists(csv_dir + "PHI-" + userid + ".csv.gz")):
        print(str(user_loc) + " ALREADY DOWNLOADED")
        return [userid, "ALREADY DOWNLOADED"]

    print(str(user_loc) + " STARTING DOWNLOAD")
    if((user_loc % 100 == 0) & (user_loc > 99)):
        print(user_loc)
        log_file = open('batch_get_donor_data_log.txt', 'a')
        log_file.write(str(user_loc)+"\n")
        log_file.close()

    status = ""

    try:
        data = get_single_tidepool_dataset.get_and_return_dataset(
                donor_group=donor_group,
                userid_of_shared_user=userid
            )

        if len(data) > 0:
            filename = csv_dir + "PHI-" + userid + ".csv.gz"
            data.to_csv(filename, index=False, compression='gzip')
            status = "Successfully Downloaded"

        else:
            status = "No Data"

    except Exception as e:
        status = "DOWNLOAD FAILED!" + str(e)
        print(status)
    print(str(user_loc) + " DOWNLOAD SUCCESSFUL!")
    return [userid, status]


# %%
if __name__ == "__main__":
    chosen_donors_file = "PHI-all-qualify-stats-and-metadata-2019-11-21.csv"
    donor_list = pd.read_csv(chosen_donors_file, low_memory=False)
    phi_donor_list=donor_list.copy()
    today_timestamp = dt.datetime.now().strftime("%Y-%m-%d")
    csv_dir = "PHI-" + today_timestamp + "-csvData/"

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # %% Start Pipeline

    start_time = time.time()

    # Startup CPU multiprocessing pool
    pool = Pool(int(cpu_count()))

    pool_array = [pool.apply_async(
            download_data,
            args=[phi_donor_list.loc[user_loc, 'userid'],
                  phi_donor_list.loc[user_loc, 'donorGroup'],
                  user_loc,
                  csv_dir
                  ]
            ) for user_loc in range(len(phi_donor_list))]

    pool.close()
    pool.join()

    end_time = time.time()
    elapsed_minutes = (end_time - start_time)/60
    elapsed_time_message = "Downloaded " + str(len(phi_donor_list)) + "  datasets in: " + \
        str(elapsed_minutes) + " minutes\n"
    print(elapsed_time_message)
    log_file = open('batch_get_donor_data_log.txt', 'a')
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
    output_results = pd.DataFrame(results_array)
    output_results.columns = ["userid", "download_status"]
    output_filename = \
        'PHI-batch-get-donor-data-results-' + \
        today_timestamp + \
        '.csv'
    output_results.to_csv(output_filename, index=False)
