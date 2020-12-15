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
import orjson
import pdb
import argparse
import environmentalVariables
import time

# %% FUNCTIONS

# %% Parse Argument Function
def get_args():
    codeDescription = "Download a single Tidepool Donor Metadata File"

    parser = argparse.ArgumentParser(description=codeDescription)

    parser.add_argument("-userid", dest="userid", default="", help="Tidepool userid to download")

    parser.add_argument(
        "-donor_group", dest="donor_group", default="", help="Tidepool donor_group to download data from"
    )

    parser.add_argument(
        "-export_directory", dest="export_directory", default="", help="Directory for metadata results to be exported"
    )

    parser.add_argument(
        "-session_token",
        dest="session_token",
        default=np.nan,
        help="Optional reusable session xtoken used for downloading data",
    )

    args = parser.parse_args()

    return args


def login_and_get_xtoken(auth):
    api_call = "https://api.tidepool.org/auth/login"
    api_response = requests.post(api_call, auth=auth)
    if api_response.ok:
        xtoken = api_response.headers["x-tidepool-session-token"]
        userid_master = orjson.loads(api_response.content.decode())["userid"]
        print("successfully established session for", auth[0])
    else:
        sys.exit("Error with " + auth[0] + ":" + str(api_response.status_code))

    return xtoken, userid_master


def logout(auth):
    api_call = "https://api.tidepool.org/auth/logout"
    api_response = requests.post(api_call, auth=auth)

    if api_response.ok:
        print("successfully logged out of", auth[0])
        pass

    else:
        sys.exit("Error with logging out for " + auth[0] + ":" + str(api_response.status_code))

    return


def get_metadata(
    donor_group=np.nan, userid_of_shared_user=np.nan, auth=np.nan, email=np.nan, password=np.nan, session_token=np.nan,
):

    """

    Parameters
    ----------
    donor_group : str
        Optional donor group name if getting Tidepool donor data
    userid_of_shared_user : str
        Optional userid if downloading data shared with master account
    auth : Tuple
        Optional (email, password) to be passed into login/logout functions
    email : str
        Optional email of account to login (if none then a request prompt will appear)
    password : str
        Optional password of account to login (if none then a request prompt will appear)
    session_token : str
        Optional xtoken if a single session is reused to download multiple datasets

    Returns
    -------
    metadata : list or pandas.DataFrame
        The complete data from the API
    userid_of_shared_user : str
        The userid of the account downloaded from (will be same as master if none specified)
    """

    # login
    if pd.notnull(donor_group):
        if donor_group == "bigdata":
            dg = ""
        else:
            dg = donor_group

        auth = environmentalVariables.get_environmental_variables(dg)

    if pd.isnull(auth):
        if pd.isnull(email):
            email = input("Enter Tidepool email address:\n")

        if pd.isnull(password):
            password = getpass.getpass("Enter password:\n")

        auth = (email, password)

    if pd.isnull(session_token):
        # login to get xtoken
        xtoken, userid_master = login_and_get_xtoken(auth)

        if pd.isnull(userid_of_shared_user):
            userid_of_shared_user = userid_master

    else:
        xtoken = session_token

    headers = {"x-tidepool-session-token": xtoken, "Content-Type": "application/json"}

    if pd.isnull(userid_of_shared_user):
        userid_of_shared_user = userid_master

    api_call = "https://api.tidepool.org/metadata/%s/profile" % userid_of_shared_user
    api_response = requests.get(api_call, headers=headers)
    if api_response.status_code == 504:
        print("profile gateway timeout... sleeping 5 sec")
        time.sleep(5)
        api_response = requests.get(api_call, headers=headers)

    metadata_df = pd.DataFrame(
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
            "about",
        ],
    )

    if api_response.ok:
        user_profile = orjson.loads(api_response.content.decode())
        if "patient" in user_profile.keys():
            for k, d in zip(user_profile["patient"].keys(), user_profile["patient"].values()):
                metadata_df.at[userid_of_shared_user, k] = d
        if "fullName" in user_profile.keys():
            metadata_df.at[userid_of_shared_user, "fullName"] = user_profile["fullName"]
        else:
            metadata_df.at[userid_of_shared_user, "fullName"] = np.nan
    else:
        sys.exit("Error getting metadata API " + str(api_response.status_code))

    metadata_df.index.rename("userid", inplace=True)

    if pd.isnull(session_token):
        logout(auth)

    return metadata_df, userid_of_shared_user


# %%
if __name__ == "__main__":

    metadata_args = get_args()

    metadata_df, _ = get_metadata(
        donor_group=metadata_args.donor_group,
        userid_of_shared_user=metadata_args.userid,
        session_token=metadata_args.session_token,
    )

    metadata_df.reset_index(inplace=True, drop=False)
    metadata_df['donorGroup'] = metadata_args.donor_group

    filepath = metadata_args.export_directory + "PHI-" + metadata_args.userid + ".csv"
    metadata_df.to_csv(filepath, index=False)
