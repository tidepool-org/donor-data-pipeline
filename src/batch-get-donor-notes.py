#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:14:38 2019

@author: jameno
"""
import pandas as pd
import datetime as dt
import os
import time
from multiprocessing import Pool, cpu_count
import get_single_donor_notes
import traceback
import sys
import subprocess as sub

#%%


def get_donor_notes_subprocess(userid, donor_group, user_loc, export_dir, xtoken_dict):

    # Add a 0.1s sleep buffer
    # time.sleep(0.1)

    if (user_loc % 100 == 0) & (user_loc > 99):
        print(user_loc)

    # Set the python unbuffered state to 1 to allow stdout buffer access
    # This allows continuous reading of subprocess output
    os.environ["PYTHONUNBUFFERED"] = "1"
    p = sub.Popen(
        [
            "python",
            "./get_single_donor_notes.py",
            "-userid",
            userid,
            "-donor_group",
            donor_group,
            "-export_dir",
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
        auth = get_single_donor_notes.get_donor_group_auth(donor_group)
        xtoken, _ = get_single_donor_notes.login_and_get_xtoken(auth)
        xtoken_dict.update({donor_group: xtoken})

    return xtoken_dict


def session_logout(phi_donor_list):
    unique_donor_groups = list(phi_donor_list.donorGroup.unique())

    for donor_group in unique_donor_groups:
        auth = get_single_donor_notes.get_donor_group_auth(donor_group)
        get_single_donor_notes.logout(auth)

    return


# %%
if __name__ == "__main__":
    today_timestamp = dt.datetime.now().strftime("%Y-%m-%d")
    data_path = "../data/PHI-{}-donor-data/".format(today_timestamp)
    chosen_donors_file = data_path + "PHI-{}-uniqueDonorList.csv".format(today_timestamp)
    donor_list = pd.read_csv(chosen_donors_file, low_memory=False)
    phi_donor_list = donor_list.copy()
    export_dir = data_path + "PHI-{}-temp-donor-notes/".format(today_timestamp)

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    file_list = os.listdir(export_dir)
    userids = pd.Series(file_list).apply(lambda x: x[4:-4])
    # Skip already downloaded metadata files
    keep_file_bool = ~phi_donor_list.userID.isin(userids)
    phi_donor_list = phi_donor_list[keep_file_bool].reset_index(drop=True)

    # Create API xtokens for each donor group
    xtoken_dict = create_xtokens_dict(phi_donor_list)

    # %% Start Multiprocessing Pool

    start_time = time.time()

    # Startup CPU multiprocessing pool
    pool = Pool(int(cpu_count()))

    pool_array = [
        pool.apply_async(
            get_donor_notes_subprocess,
            args=[
                phi_donor_list.loc[user_loc, "userID"],
                phi_donor_list.loc[user_loc, "donorGroup"],
                user_loc,
                export_dir,
                xtoken_dict,
            ],
        ) for user_loc in range(len(phi_donor_list))
    ]

    pool.close()
    pool.join()

    # %% Logout of each session and delete tokens
    session_logout(phi_donor_list)

    # %% Append results of each pool into an array
    all_donor_notes = []

    donor_notes_file_list = os.listdir(export_dir)

    if len(donor_notes_file_list) > 0:
        print("Appending all donor notes together...", end="")
        for donor_notes_file in donor_notes_file_list:
            donor_notes_file_path = os.path.join(export_dir, donor_notes_file)
            all_donor_notes.append(pd.read_csv(donor_notes_file_path))
        print("done!")

        # Convert results into dataframe
        all_donor_notes_df = pd.concat(all_donor_notes, sort=False).reset_index(drop=True)
        all_donor_notes_filename = data_path + "PHI-all-donor-notes-" + today_timestamp + ".csv"
        all_donor_notes_df.to_csv(all_donor_notes_filename, index=False)
        num_notes = len(all_donor_notes_df)
        num_donors_with_notes = len(all_donor_notes_df['userid'].unique())
    else:
        num_notes = 0
        num_donors_with_notes = 0

    end_time = time.time()
    elapsed_minutes = round((end_time - start_time) / 60, 4)
    elapsed_time_message = "{} donor notes from {} donors, downloaded in {} minutes.".format(num_notes, num_donors_with_notes, str(elapsed_minutes))
    print(elapsed_time_message)
