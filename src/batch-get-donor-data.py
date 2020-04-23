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
import subprocess as sub


# %%
def download_data(userid, donor_group, user_loc, export_dir, xtoken_dict):

    # Add a 0.1s sleep buffer
    time.sleep(0.1)

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
                userid_of_shared_user=userid,
                session_token=xtoken_dict.get(donor_group)
            )

        if len(data) > 0:
            filename = export_dir + "PHI-" + userid + ".csv.gz"
            data.to_csv(filename, index=False, compression='gzip')
            status = "Successfully Downloaded"

        else:
            status = "No Data"

    except Exception as e:
        status = "DOWNLOAD FAILED! " + str(e)

    print(status)

    return [userid, status]


def download_data_subprocess(userid,
                             donor_group,
                             user_loc,
                             export_dir,
                             xtoken_dict):

    # Add a 0.1s sleep buffer
    time.sleep(0.1)

    # print(str(user_loc) + " STARTING DOWNLOAD")
    if((user_loc % 100 == 0) & (user_loc > 99)):
        print(user_loc)
        log_file = open('batch_get_donor_data_log.txt', 'a')
        log_file.write(str(user_loc)+"\n")
        log_file.close()

    # Set the python unbuffered state to 1 to allow stdout buffer access
    # This allows continuous reading of subprocess output
    os.environ["PYTHONUNBUFFERED"] = "1"
    p = sub.Popen(
        [
             "python", "./get_single_tidepool_dataset.py",
             "-donor_group", donor_group,
             "-userid_of_shared_user", userid,
             "-session_token", xtoken_dict.get(donor_group),
             "-export_dir", export_dir
         ],
        stdout=sub.PIPE,
        stderr=sub.PIPE
    )

    # Continuous write out stdout output
    # for line in iter(p.stdout.readline, b''):
    #    sys.stdout.write(line.decode(sys.stdout.encoding))
    for line in iter(p.stdout.readline, b''):
        sys.stdout.write(line.decode("utf-8"))

    output, errors = p.communicate()
    output = output.decode("utf-8")
    errors = errors.decode("utf-8")

    if errors != '':
        print(errors)

    return


def create_xtokens_dict(phi_donor_list):

    unique_donor_groups = list(phi_donor_list.donorGroup.unique())

    xtoken_dict = {}

    for donor_group in unique_donor_groups:
        auth = get_single_tidepool_dataset.get_donor_group_auth(donor_group)
        xtoken, _ = get_single_tidepool_dataset.login_and_get_xtoken(auth)
        xtoken_dict.update({donor_group: xtoken})

    return xtoken_dict


def session_logout(phi_donor_list):
    unique_donor_groups = list(phi_donor_list.donorGroup.unique())

    for donor_group in unique_donor_groups:
        auth = get_single_tidepool_dataset.get_donor_group_auth(donor_group)
        get_single_tidepool_dataset.logout(auth)

    return


# %%
if __name__ == "__main__":
    chosen_donors_file = \
        "../data/PHI-2020-04-17-donor-data/PHI-2020-04-17-uniqueDonorList.csv"
    donor_list = pd.read_csv(chosen_donors_file, low_memory=False)
    phi_donor_list = donor_list.copy()
    today_timestamp = dt.datetime.now().strftime("%Y-%m-%d")
    export_dir = "PHI-" + today_timestamp + "-csvData/"

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # Skip already downloaded data and accounts with missing data
    file_list = os.listdir(export_dir)
    downloaded_files = pd.Series(file_list).apply(lambda x: x[4:-7])
    not_downloaded_accounts = ~phi_donor_list.userID.isin(downloaded_files)

    # Get accounts known to be missing data
    if os.path.exists('PHI-empty-accounts.txt'):
        empty_account_list = open('PHI-empty-accounts.txt', 'r')
        empty_accounts = empty_account_list.read()
        empty_accounts = pd.Series(empty_accounts.split('\n')[:-1])
        not_empty_accounts = ~phi_donor_list.userID.isin(empty_accounts)
    else:
        not_empty_accounts = True

    keep_file_bool = not_downloaded_accounts & not_empty_accounts
    phi_donor_list = phi_donor_list[keep_file_bool].reset_index(drop=True)
    phi_donor_list = phi_donor_list.loc[:3]

    # Create API xtokens for each donor group
    xtoken_dict = create_xtokens_dict(phi_donor_list)

    # %% Start Pipeline

    print("Retrieving " + str(len(phi_donor_list)) + " data files.")
    start_time = time.time()

    # Startup CPU multiprocessing pool
    pool = Pool(int(cpu_count()))

    pool_array = [pool.apply_async(
            download_data_subprocess,
            args=[phi_donor_list.loc[user_loc, 'userID'],
                  phi_donor_list.loc[user_loc, 'donorGroup'],
                  user_loc,
                  export_dir,
                  xtoken_dict
                  ]
            ) for user_loc in range(len(phi_donor_list))]

    pool.close()
    pool.join()

    end_time = time.time()
    elapsed_minutes = (end_time - start_time)/60

    new_file_list = os.listdir(export_dir)
    new_file_list = [file for file in new_file_list if 'csv' in file]
    new_downloaded_files = pd.Series(new_file_list).apply(lambda x: x[4:-7])
    successful_download_count = (
            sum(~new_downloaded_files.isin(downloaded_files))
    )

    elapsed_time_message = (
            "Finished downloading "
            + str(successful_download_count)
            + " / "
            + str(len(phi_donor_list))
            + " datasets in: "
            + str(elapsed_minutes)
            + " minutes\n"
    )
    print(elapsed_time_message)
    log_file = open('batch_get_donor_data_log.txt', 'a')
    log_file.write(str(elapsed_time_message)+"\n")
    log_file.close()

    # %% Logout of each session and delete tokens
    session_logout(phi_donor_list)

#    # %% Append results of each pool into an array
#
#    results_array = []
#
#    for result_loc in range(len(pool_array)):
#        try:
#            results_array.append(pool_array[result_loc].get())
#        except Exception as e:
#            print('Failed to get results! ' + str(e))
#            exception_text = traceback.format_exception(*sys.exc_info())
#            print('\nException Text:\n')
#            for text_string in exception_text:
#                print(text_string)
#
#    # %% Convert results into dataframe and save
#    output_results = pd.DataFrame(results_array)
#    output_results.columns = ["userid", "download_status"]
#    output_filename = \
#        'PHI-batch-get-donor-data-results-' + \
#        today_timestamp + \
#        '.csv'
#    output_results.to_csv(output_filename, index=False)
