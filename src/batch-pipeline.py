#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Pipeline
===============
:File: batch-pipeline.py
:Description: A batch script for running the donor data pipeline
:Version: 0.0.1
:Created: 2019-09-28
:Authors: Jason Meno (jam)
:Dependencies:
    - A Tidepool .env file
    -
:License: BSD-2-Clause
"""

# %% Import all modules
import pandas as pd
import numpy as np
import datetime as dt
import time
import os
import subprocess as sub
from multiprocessing import Pool, cpu_count, current_process, get_context
from multiprocessing.pool import ThreadPool
import traceback
import sys
import donor_data_pipeline
import pickle

# %% Global Pipeline Defaults (eventually become config args)

# Download and save the unique donor list
unique_donors_file = 'PHI-unique-donor-list-2019-11-12.csv'
saveUniqueDonorList = False

import_data_path = 'PHI-2019-11-13-csvData/'
import_data = True
save_new_data = False

chosen_donors_file = "PHI-batch-vector-qualify-10k-with-metadata-2019-12-05.csv"
subset_selection = "sample"
dataset_type = "all"
test_set_days = 0


# %%
def run_pipeline(userid,
                 donor_group,
                 export_directory,
                 user_loc,
                 import_data,
                 save_new_data,
                 data_path,
                 dataset_type,
                 test_set_days):
    # Set the python unbuffered state to 1 to allow stdout buffer access
    # This allows continuous reading of subprocess output
    os.environ["PYTHONUNBUFFERED"] = "1"
    p = sub.Popen(
        [
             "python", "./donor_data_pipeline.py",
             "-userid", userid,
             "-donor_group", donor_group,
             "-export_directory", export_directory,
             "-user_loc", str(user_loc),
             "-import_data", str(import_data),
             "-save_new_data", str(save_new_data),
             "-data_path", data_path,
             "-dataset_type", dataset_type,
             "-test_set_days", str(test_set_days),
             "-custom_start_date", "2019-05-01",
             "-custom_end_date", "2019-08-01",
             "-api_sleep_buffer", "True"
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

    if errors == '':
        print(output)
    else:
        print(errors)

    done_statement = str(user_loc) + " is complete and returned successfully"

    return done_statement

# %% Multithreading Protection
# Each multiprocess fork will import this entire script as a module!
# Keep the insides safe from entering an infinite loop with a __main__ check

if __name__ == '__main__':

    today_timestamp = dt.datetime.now().strftime("%Y-%m-%d")
    export_directory = "PHI-" + subset_selection + "-export-" + today_timestamp

    pipeline_results_dir = export_directory + '/PHI-pipeline-results/'
    train_data_dir = export_directory + "/train/train-data/"
    test_data_dir = export_directory + "/test/test-data/"

    csv_dir = export_directory + "/PHI-csvData/"
    qa_dir = export_directory + "/QA/"
    qa_LTE_cDays_dir = qa_dir + "PHI-LTE-cDays/"
    qa_qualifed_cDays_dir = qa_dir + "PHI-qualified-days/"
    qa_vector_qualified_cDays_dir = qa_dir + "PHI-vector-qualified-days/"
    qa_train_summary_viz_dir = qa_dir + "vizQA/train-dataset-summary-vizQA/"
    qa_test_summary_viz_dir = qa_dir + "vizQA/test-dataset-summary-vizQA/"
    qa_train_local_time_viz_dir = qa_dir + "vizQA/train-local-time-vizQA/"
    qa_test_local_time_viz_dir = qa_dir + "vizQA/test-local-time-vizQA/"
    qa_train_vector_qualify_viz_dir = qa_dir + \
        "vizQA/train-vector-qualify-vizQA/"
    qa_test_vector_qualify_viz_dir = qa_dir + \
        "vizQA/test-vector-qualify-vizQA/"
    qa_plotly_dropped_data_viz_dir = qa_dir + \
     "vizQA/plotly-dropped-data-vizQA/"

    directories = [pipeline_results_dir,
                   train_data_dir,
                   test_data_dir,
                   csv_dir,
                   qa_dir,
                   qa_LTE_cDays_dir,
                   qa_qualifed_cDays_dir,
                   qa_vector_qualified_cDays_dir,
                   qa_train_summary_viz_dir,
                   qa_test_summary_viz_dir,
                   qa_train_local_time_viz_dir,
                   qa_test_local_time_viz_dir,
                   qa_train_vector_qualify_viz_dir,
                   qa_test_vector_qualify_viz_dir,
                   qa_plotly_dropped_data_viz_dir
                   ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    phi_donor_list = \
        donor_data_pipeline.get_unique_donor_list(today_timestamp,
                                                  saveUniqueDonorList,
                                                  unique_donors_file)

    donor_list = pd.read_csv(chosen_donors_file, low_memory=False)
    donor_list = donor_list[donor_list['new_selection'] == subset_selection]
    donor_list.reset_index(drop=True, inplace=True)

    if('import_data_path' in locals()):
        donor_list["data_path"] = \
            import_data_path + \
            "PHI-" + donor_list['userid'] + ".csv.gz"
    else:
        donor_list["data_path"] = ""

    dictionary_template = pd.read_csv('data-dictionary-template.csv',
                                      low_memory=False)

    # %% Start Pipeline

    start_time = time.time()

    # Startup CPU multiprocessing pool
    n_cpus = int(os.cpu_count())
    pool = Pool(n_cpus)

    # List comprehension initiates asynchronous processing
    pool_array = [
            pool.apply_async(
                    run_pipeline,
                    args=[donor_list.loc[user_loc, 'userid'],
                          donor_list.loc[user_loc, 'donorGroup'],
                          export_directory,
                          user_loc,
                          import_data,
                          save_new_data,
                          donor_list.loc[user_loc, 'data_path'],
                          dataset_type,
                          test_set_days
                          ],
                    ) for user_loc in range(len(donor_list))
                 ]
    # Close pool for letting any other subprocesses from being added
    pool.close()

    # Close each worker in pool before joining
    for result_loc in range(len(pool_array)):
        try:
            print(str(pool_array[result_loc].get()) + " --- at --- " + dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        except Exception as e:
            print('Failed to get results! ' + str(e))
            exception_text = traceback.format_exception(*sys.exc_info())
            print('\nException Text:\n')
            for text_string in exception_text:
                print(text_string)

    # Join all child subprocesses to the parent process once finished
    pool.join()

    end_time = time.time()
    elapsed_minutes = (end_time - start_time)/60
    elapsed_time_message = "Pipeline runs completed in: " + \
        str(elapsed_minutes) + " minutes\n"
    print(elapsed_time_message)

    # %% Append results of each pool into an array

    results_array = []
    """
    for result_loc in range(len(pool_array)):
        try:
            results_array.append(pool_array[result_loc].get())
        except Exception as e:
            print('Failed to get results! ' + str(e))
            exception_text = traceback.format_exception(*sys.exc_info())
            print('\nException Text:\n')
            for text_string in exception_text:
                print(text_string)

    # Convert results into dataframe
    """

    result_file_list = os.listdir(pipeline_results_dir)

    for results_file in result_file_list:
        results_file_path = os.path.join(pipeline_results_dir, results_file)

        with open(results_file_path, 'rb') as pickle_file:
            results_array.append(pickle.load(pickle_file))

    results_df = pd.DataFrame(results_array)

    # %% Separate results into their respective variables

    data_dictionary = pd.DataFrame(set(results_df[0].sum()),
                                   columns=['FIELD']).sort_values(
                                           by='FIELD', ascending=True)

    data_dictionary = pd.merge(data_dictionary,
                               dictionary_template,
                               on='FIELD',
                               how='left')

    all_removed_columns = pd.DataFrame(set(results_df[1].sum()),
                                       columns=['REMOVED_FIELD']).sort_values(
                                               by='REMOVED_FIELD',
                                               ascending=True)

    all_qualification_metadata = pd.concat(results_df[2].values, sort=False)
    all_vector_qualify_metadata = pd.concat(results_df[3].values, sort=False)
    all_anonymized_metadata = pd.concat(results_df[4].values, sort=False)
    all_train_summary_metadata = pd.concat(results_df[5].values, sort=False)
    all_test_summary_metadata = pd.concat(results_df[6].values, sort=False)
    all_train_anonymized_stats_qa = pd.concat(results_df[7].values, sort=False)
    all_test_anonymized_stats_qa = pd.concat(results_df[8].values, sort=False)
    all_pipeline_metadata = pd.concat(results_df[9].values, sort=False)
    all_df_changes = pd.concat(results_df[10].values, sort=False)

    # Drop duplicate df_changes and reorder and sort
    all_df_changes = \
        all_df_changes.\
        reindex(sorted(all_df_changes.columns), axis=1).\
        reset_index(drop=True)

    for col_idx in range(0, len(list(all_df_changes)), 2):
        value_col = list(all_df_changes)[col_idx]
        len_col = list(all_df_changes)[col_idx+1]

        non_null_length_loc = all_df_changes[len_col].notnull()
        dropped_and_sorted = \
            all_df_changes.loc[non_null_length_loc, [value_col, len_col]].\
            drop_duplicates(len_col).\
            sort_values(by=len_col).\
            reset_index(drop=True)

        all_df_changes[[value_col, len_col]] = np.nan
        all_df_changes.loc[dropped_and_sorted.index, [value_col, len_col]] = \
            dropped_and_sorted

    # Drop all rows that are the same (only nans)
    all_df_changes.drop_duplicates(keep=False, inplace=True)
    # Drop all length columns
    all_df_changes.drop(list(all_df_changes)[1::2], axis=1, inplace=True)
    # %% Export all data to their appropriate folders

    data_dictionary.to_csv(export_directory + "/train/" +
                           subset_selection + "-data-dictionary.csv",
                           index=False)

    data_dictionary.to_csv(export_directory + "/test/" +
                           subset_selection + "-data-dictionary.csv",
                           index=False)

    all_train_summary_metadata.to_csv(export_directory + "/train/" +
                                      subset_selection + "-train-metadata-summary.csv",
                                      index=False)

    all_test_summary_metadata.to_csv(export_directory + "/test/" +
                                     subset_selection + "-test-metadata-summary.csv",
                                     index=False)

    all_removed_columns.to_csv(qa_dir +
                               "removed-columns-during-anonymization-" +
                               today_timestamp +
                               ".csv",
                               index=False)

    all_anonymized_metadata.to_csv(qa_dir +
                                   "PHI-anonymized-metadata-" +
                                   today_timestamp +
                                   ".csv",
                                   index=False)

    all_qualification_metadata.to_csv(qa_dir +
                                      "PHI-qualification-metadata-" +
                                      today_timestamp +
                                      ".csv",
                                      index=False)

    all_vector_qualify_metadata.to_csv(qa_dir +
                                       "PHI-vector-qualify-metadata-" +
                                       today_timestamp +
                                       ".csv",
                                       index=False)

    all_train_anonymized_stats_qa.to_csv(qa_dir +
                                         "train-anonymized-stats-QA-" +
                                         today_timestamp +
                                         ".csv",
                                         index=False
                                         )

    all_test_anonymized_stats_qa.to_csv(qa_dir +
                                        "test-anonymized-stats-QA-" +
                                        today_timestamp +
                                        ".csv",
                                        index=False
                                        )

    all_pipeline_metadata.to_csv(qa_dir +
                                 "PHI-pipeline-metadata-" +
                                 today_timestamp +
                                 ".csv",
                                 index=False
                                 )

    all_df_changes.to_csv(qa_dir +
                          "PHI-dataframe-changes-" +
                          today_timestamp +
                          ".csv",
                          index=False
                          )

    print("All pipeline data exports complete.")

# %% Test
