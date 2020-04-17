#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:14:38 2019

@author: jameno
"""
import pandas as pd
from multiprocessing.pool import ThreadPool
import time
import datetime as dt
import os
import time
from multiprocessing import Pool, cpu_count, get_context
import traceback
import sys
import subprocess as sub
#%%

def get_metadata(userid, donor_group, user_loc, export_directory):

    if(user_loc % 100 == 0):
        print(user_loc)

    # Set the python unbuffered state to 1 to allow stdout buffer access
    # This allows continuous reading of subprocess output
    os.environ["PYTHONUNBUFFERED"] = "1"
    p = sub.Popen(
        [
             "python", "./get_single_donor_metadata.py",
             "-userid", userid,
             "-donor_group", donor_group,
             "-export_directory", export_directory
         ],
        stdout=sub.PIPE,
        stderr=sub.PIPE
    )

    # Continuous write out stdout output
    #for line in iter(p.stdout.readline, b''):
    #    sys.stdout.write(line.decode(sys.stdout.encoding))
    for line in iter(p.stdout.readline, b''):
        sys.stdout.write(line.decode("utf-8"))

    output, errors = p.communicate()
    output = output.decode("utf-8")
    errors = errors.decode("utf-8")

    if errors != '':
        print(errors)

    return

#%%
if __name__ == "__main__":
    phi_donor_list = pd.read_csv("PHI-unique-donor-list-2019-11-12.csv")
    today_timestamp = dt.datetime.now().strftime("%Y-%m-%d")
    export_directory = "PHI-" + today_timestamp + "-tempMetadata/"

    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
    # %% Start Pipeline

    start_time = time.time()

    # Startup CPU multiprocessing pool
    pool = Pool(int(cpu_count()))

    pool_array = [pool.apply_async(
            get_metadata,
            args=[phi_donor_list.loc[user_loc, 'userID'],
                  phi_donor_list.loc[user_loc, 'donorGroup'],
                  user_loc,
                  export_directory
                  ]
            ) for user_loc in range(len(phi_donor_list))]

    pool.close()
    pool.join()


    end_time = time.time()
    elapsed_minutes = (end_time - start_time)/60
    elapsed_time_message = "Metadata runs completed in: " + \
        str(elapsed_minutes) + " minutes\n"
    print(elapsed_time_message)

    # %% Append results of each pool into an array
    """
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
    """

    print("Appending datasets to results array...", end="")
    results_array = []

    result_file_list = os.listdir(export_directory)

    for results_file in result_file_list:
        results_file_path = os.path.join(export_directory, results_file)
        results_array.append(pd.read_csv(results_file_path))
    print("done!")
# %%
    # Convert results into dataframe
    metadata_df = pd.concat(results_array, sort=False)
    metadata_filename = "PHI-batch-metadata-" + today_timestamp + ".csv"
    metadata_df.to_csv(metadata_filename, index=False)
