# -*- coding: utf-8 -*-
"""get_donor_data_and_metadata.py
In the context of the big data donation
project, this code grabs donor data and metadata.

This code calls accept_new_donors_and_get_donor_list.py
to get the most recent donor list
"""

# %% REQUIRED LIBRARIES
import pandas as pd
import datetime as dt
import numpy as np
import os
import sys
import getpass
import requests
import json
import pdb
import argparse
import environmentalVariables
import time

# %% FUNCTIONS

# %% Parse Argument Function
def get_args():
    codeDescription = "Download a single Tidepool Donor Metadata File"

    parser = argparse.ArgumentParser(description=codeDescription)

    parser.add_argument("-userid",
                        dest="userid",
                        default="",
                        help="Tidepool userid to download")

    parser.add_argument("-donor_group",
                        dest="donor_group",
                        default="",
                        help="Tidepool donor_group to download data from")

    parser.add_argument("-export_directory",
                        dest="export_directory",
                        default="",
                        help="Directory for metadata results to be exported")

    args = parser.parse_args()

    return args

def make_folder_if_doesnt_exist(folder_paths):
    ''' function requires a single path or a list of paths'''
    if not isinstance(folder_paths, list):
        folder_paths = [folder_paths]
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    return


def get_shared_metadata(
        donor_group=np.nan,
        userid_of_shared_user=np.nan,
        auth=np.nan,
        email=np.nan,
        password=np.nan):

    # login
    if pd.notnull(donor_group):
        #print("pd.notnull(donor_group)")
        if donor_group == "bigdata":
            dg = ""
        else:
            dg = donor_group

        auth = environmentalVariables.get_environmental_variables(dg)

    if pd.isnull(auth):
        #print("pd.notnull(auth)")
        if pd.isnull(email):
            email = input("Enter Tidepool email address:\n")

        if pd.isnull(password):
            password = getpass.getpass("Enter password:\n")

        auth = (email, password)

    api_call = "https://api.tidepool.org/auth/login"
    #print("before api_response")
    api_response = requests.post(api_call, auth=auth)

    if(api_response.status_code == 504):
        print("login gateway timeout... sleeping 5 sec")
        time.sleep(5)
        api_response = requests.post(api_call, auth=auth)
    #print("after api_response")
    if(api_response.ok):
        xtoken = api_response.headers["x-tidepool-session-token"]
        userid_master = json.loads(api_response.content.decode())["userid"]
        headers = {
            "x-tidepool-session-token": xtoken,
            "Content-Type": "application/json"
        }
    else:
        sys.exit("Error with " + auth[0] + ":" + str(api_response.status_code))

    if pd.isnull(userid_of_shared_user):
        userid_of_shared_user = userid_master

    api_call = (
        "https://api.tidepool.org/metadata/%s/profile"
        % userid_of_shared_user
    )
    api_response = requests.get(api_call, headers=headers)
    if(api_response.status_code == 504):
        print("profile gateway timeout... sleeping 5 sec")
        time.sleep(5)
        api_response = requests.get(api_call, headers=headers)

    df = pd.DataFrame(
        dtype=object,
        columns=[
            "fullName",
            "diagnosisType",
            "diagnosisDate",
            "biologicalSex",
            "birthday",
            "targetTimezone",
            "targetDevices",
            "isOtherPerson",
            "about"
        ]
    )

    if(api_response.ok):
        user_profile = json.loads(api_response.content.decode())
        if "patient" in user_profile.keys():
            for k, d in zip(
                user_profile["patient"].keys(),
                user_profile["patient"].values()
            ):
                df.at[userid_of_shared_user, k] = d
        if "fullName" in user_profile.keys():
            df.at[userid_of_shared_user, "fullName"] = user_profile["fullName"]
        else:
            df.at[userid_of_shared_user, "fullName"] = np.nan
    else:
        sys.exit(
            "Error getting metadata API " +
            str(api_response.status_code)
        )

    # logout
    api_call = "https://api.tidepool.org/auth/logout"
    api_response = requests.post(api_call, auth=auth)

    if(api_response.status_code == 504):
        print("logout gateway timeout... sleeping 5 sec")
        time.sleep(5)
        api_response = requests.post(api_call, auth=auth)

    if(api_response.ok):
        pass
        #print("successfully logged out of", auth[0])

    else:
        sys.exit(
            "Error with logging out for " +
            auth[0] + ":" + str(api_response.status_code)
        )
    df.index.rename("userid", inplace=True)
    #print("returning...")
    return df, userid_of_shared_user

# %% START OF CODE
def get_and_return_metadata(
    date_stamp=dt.datetime.now().strftime("%Y-%m-%d"),
    data_path=np.nan,
    donor_group=np.nan,
    userid_of_shared_user=np.nan,
    auth=np.nan,
    email=np.nan,
    password=np.nan
):

    # get metadata
    meta_df, userid = get_shared_metadata(
        donor_group=donor_group,
        userid_of_shared_user=userid_of_shared_user,
        auth=auth,
        email=email,
        password=password
    )

    return meta_df


if __name__ == "__main__":

    metadata_args = get_args()

    metadata = get_and_return_metadata(
            donor_group = metadata_args.donor_group,
            userid_of_shared_user = metadata_args.userid
        )

    metadata.reset_index(inplace=True, drop=False)

    filepath = metadata_args.export_directory + "PHI-" + metadata_args.userid + ".csv"
    metadata.to_csv(filepath, index=False)


