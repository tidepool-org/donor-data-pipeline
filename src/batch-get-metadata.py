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

#%%


def get_metadata_subprocess(userid, donor_group, user_loc, export_dir, xtoken_dict):

    # Add a 0.1s sleep buffer
    time.sleep(0.1)

    if (user_loc % 100 == 0) & (user_loc > 99):
        print(user_loc)

    # Set the python unbuffered state to 1 to allow stdout buffer access
    # This allows continuous reading of subprocess output
    os.environ["PYTHONUNBUFFERED"] = "1"
    p = sub.Popen(
        [
            "python",
            "./get_single_donor_metadata.py",
            "-userid",
            userid,
            "-donor_group",
            donor_group,
            "-export_directory",
            export_dir,
            "-session_token",
            xtoken_dict.get(donor_group),
        ],
        stdout=sub.PIPE,
        stderr=sub.PIPE,
    )

    # Continuous write out stdout output
    # for line in iter(p.stdout.readline, b''):
    #    sys.stdout.write(line.decode(sys.stdout.encoding))
    for line in iter(p.stdout.readline, b""):
        sys.stdout.write(line.decode("utf-8"))

    output, errors = p.communicate()
    output = output.decode("utf-8")
    errors = errors.decode("utf-8")

    if errors != "":
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
    today_timestamp = dt.datetime.now().strftime("%Y-%m-%d")
    data_path = "../data/PHI-{}-donor-data/".format(today_timestamp)
    chosen_donors_file = data_path + "PHI-{}-uniqueDonorList.csv".format(today_timestamp)
    donor_list = pd.read_csv(chosen_donors_file, low_memory=False)
    phi_donor_list = donor_list.copy()
    export_dir = data_path + "PHI-{}-tempMetadata/".format(today_timestamp)

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    file_list = os.listdir(export_dir)
    userids = pd.Series(file_list).apply(lambda x: x[4:-4])
    # Skip already downloaded metadata files
    keep_file_bool = ~phi_donor_list.userID.isin(userids)
    phi_donor_list = phi_donor_list[keep_file_bool].reset_index(drop=True)[:150]

    # Create API xtokens for each donor group
    xtoken_dict = create_xtokens_dict(phi_donor_list)

    # %% Start Multiprocessing Pool

    start_time = time.time()

    # Startup CPU multiprocessing pool
    pool = Pool(int(cpu_count()))

    pool_array = [
        pool.apply_async(
            get_metadata_subprocess,
            args=[
                phi_donor_list.loc[user_loc, "userID"],
                phi_donor_list.loc[user_loc, "donorGroup"],
                user_loc,
                export_dir,
                xtoken_dict,
            ],
        )
        for user_loc in range(len(phi_donor_list))
    ]

    pool.close()
    pool.join()

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    elapsed_time_message = "Metadata runs completed in: " + str(elapsed_minutes) + " minutes\n"
    print(elapsed_time_message)

    # %% Logout of each session and delete tokens
    session_logout(phi_donor_list)

    # %% Append results of each pool into an array
    print("Appending metadata files together...", end="")
    metadata_list = []

    metadata_file_list = os.listdir(export_dir)

    for metadata_file in metadata_file_list:
        metadata_file_path = os.path.join(export_dir, metadata_file)
        metadata_list.append(pd.read_csv(metadata_file_path))
    print("done!")
    # %%
    # Convert results into dataframe
    metadata_df = pd.concat(metadata_list, sort=False)
    metadata_filename = data_path + "PHI-batch-metadata-" + today_timestamp + ".csv"
    metadata_df.to_csv(metadata_filename, index=False)
