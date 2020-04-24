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
import environmentalVariables
import argparse


# %% FUNCTIONS
def get_args():
    codeDescription = "Download a single Tidepool Dataset"

    parser = argparse.ArgumentParser(description=codeDescription)

    parser.add_argument("-userid_of_shared_user",
                        dest="userid_of_shared_user",
                        default=np.nan,
                        help="Tidepool userid to download")

    parser.add_argument("-donor_group",
                        dest="donor_group",
                        default=np.nan,
                        help="Tidepool donor_group to download data from")

    parser.add_argument("-session_token",
                        dest="session_token",
                        default=np.nan,
                        help="The session xtoken used for downloading data")

    parser.add_argument("-export_dir",
                        dest="export_dir",
                        default="",
                        help="The session xtoken used for downloading data")

    args = parser.parse_args()

    return args


def data_api_call(userid, startDate, endDate, headers):

    startDate = startDate.strftime("%Y-%m-%d") + "T00:00:00.000Z"
    endDate = endDate.strftime("%Y-%m-%d") + "T23:59:59.999Z"

    api_call = (
        "https://api.tidepool.org/data/" + userid + "?" +
        "endDate=" + endDate + "&" +
        "startDate=" + startDate + "&" +
        "dexcom=true" + "&" +
        "medtronic=true" + "&" +
        "carelink=true"
    )

    api_response = requests.get(api_call, headers=headers)
    if(api_response.ok):
        json_data = json.loads(api_response.content.decode())
        df = pd.DataFrame(json_data)
        # print("getting data between %s and %s" % (startDate, endDate))

    else:
        sys.exit(
            "ERROR in getting data for "
            + str(userid)
            + " Between "
            + str(startDate)
            + " & "
            + str(endDate)
            + " API RESPONSE: "
            + str(api_response.status_code)
        )

    endDate = pd.to_datetime(startDate) - pd.Timedelta(1, unit="d")

    return df, endDate


def get_donor_group_auth(donor_group):
    if donor_group == "bigdata":
        dg = ""
    else:
        dg = donor_group

    auth = environmentalVariables.get_environmental_variables(dg)

    return auth


def login_and_get_xtoken(auth):
    api_call = "https://api.tidepool.org/auth/login"
    api_response = requests.post(api_call, auth=auth)
    if(api_response.ok):
        xtoken = api_response.headers["x-tidepool-session-token"]
        userid_master = json.loads(api_response.content.decode())["userid"]
        print("successfully established session for", auth[0])
    else:
        sys.exit("Error with "
                 + auth[0]
                 + ":"
                 + str(api_response.status_code))

    return xtoken, userid_master


def logout(auth):
    api_call = "https://api.tidepool.org/auth/logout"
    api_response = requests.post(api_call, auth=auth)

    if(api_response.ok):
        print("successfully logged out of", auth[0])
        pass

    else:
        sys.exit(
            "Error with logging out for " +
            auth[0] + ":" + str(api_response.status_code)
        )

    return


def get_data(
    weeks_of_data=10*52,
    donor_group=np.nan,
    userid_of_shared_user=np.nan,
    auth=np.nan,
    email=np.nan,
    password=np.nan,
    session_token=np.nan
):
    if pd.notnull(donor_group):
        auth = get_donor_group_auth(donor_group)

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

    headers = {
            "x-tidepool-session-token": xtoken,
            "Content-Type": "application/json"
        }

    df = pd.DataFrame()
    endDate = pd.datetime.now() + pd.Timedelta(1, unit="d")

    if weeks_of_data > 52:
        years_of_data = int(np.floor(weeks_of_data/52))

        for years in range(0, years_of_data + 1):
            startDate = pd.datetime(
                endDate.year - 1,
                endDate.month,
                endDate.day
            ) + pd.Timedelta(1, unit="d")
            year_df, endDate = data_api_call(
                userid_of_shared_user,
                startDate,
                endDate,
                headers
            )

            df = pd.concat(
                [df, year_df],
                ignore_index=True,
                sort=False
            )

    else:
        startDate = (
            pd.to_datetime(endDate) - pd.Timedelta(weeks_of_data*7, "d")
        )

        df, _ = data_api_call(
            userid_of_shared_user,
            startDate,
            endDate,
            headers
            )

    if pd.isnull(session_token):
        logout(auth)

    return df, userid_of_shared_user


# %% START OF CODE
def get_and_return_dataset(
    date_stamp=dt.datetime.now().strftime("%Y-%m-%d"),
    weeks_of_data=52*10,
    donor_group=np.nan,
    userid_of_shared_user=np.nan,
    auth=np.nan,
    email=np.nan,
    password=np.nan,
    session_token=np.nan
):

    # get dataset
    data, userid = get_data(
        weeks_of_data=weeks_of_data,
        donor_group=donor_group,
        userid_of_shared_user=userid_of_shared_user,
        auth=auth,
        email=email,
        password=password,
        session_token=session_token
    )

    return data


# %%
if __name__ == "__main__":

    data_args = get_args()

    try:
        data = get_and_return_dataset(
                    donor_group=data_args.donor_group,
                    userid_of_shared_user=data_args.userid_of_shared_user,
                    session_token=data_args.session_token
                )

        if len(data) > 0:
            filename = (
                    data_args.export_dir
                    + "PHI-"
                    + data_args.userid_of_shared_user
                    + ".csv.gz"
            )
            data.reset_index(inplace=True, drop=True)
            data.to_csv(filename, index=False, compression='gzip')
        else:
            # Append userid to list of empty datasets
            empty_dataset_list = open('PHI-empty-accounts.txt', 'a')
            empty_dataset_list.write(data_args.userid_of_shared_user + "\n")
            empty_dataset_list.close()

    except Exception as e:
        print("~~~~~~~~~~~Exception Captured Below~~~~~~~~~~~~")
        print("FAILED TO GET DATA FOR " + data_args.userid_of_shared_user)
        print(e)
        print("\n")

        failed_dataset_list = open('PHI-failed-accounts.txt', 'a')
        failed_dataset_list.write(data_args.userid_of_shared_user + "\n")
        failed_dataset_list.close()
