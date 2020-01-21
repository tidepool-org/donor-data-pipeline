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

# %% FUNCTIONS
def make_folder_if_doesnt_exist(folder_paths):
    ''' function requires a single path or a list of paths'''
    if not isinstance(folder_paths, list):
        folder_paths = [folder_paths]
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    return


def get_data_api(userid, startDate, endDate, headers):

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
        #print("getting data between %s and %s" % (startDate, endDate))

    else:
        sys.exit(
            "ERROR in getting data between %s and %s" % (startDate, endDate),
            api_response.status_code
        )

    endDate = pd.to_datetime(startDate) - pd.Timedelta(1, unit="d")

    return df, endDate


def get_data(
    weeks_of_data=10*52,
    donor_group=np.nan,
    userid_of_shared_user=np.nan,
    auth=np.nan,
    email=np.nan,
    password=np.nan,
):
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

    api_call = "https://api.tidepool.org/auth/login"
    api_response = requests.post(api_call, auth=auth)
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
            year_df, endDate = get_data_api(
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

        df, _ = get_data_api(
            userid_of_shared_user,
            startDate,
            endDate,
            headers
            )

    # logout
    api_call = "https://api.tidepool.org/auth/logout"
    api_response = requests.post(api_call, auth=auth)

    if(api_response.ok):
        pass
        #print("successfully logged out of", auth[0])

    else:
        sys.exit(
            "Error with logging out for " +
            auth[0] + ":" + str(api_response.status_code)
        )

    return df, userid_of_shared_user


# %% START OF CODE
def get_and_return_dataset(
    date_stamp=dt.datetime.now().strftime("%Y-%m-%d"),
    weeks_of_data=52*10,
    donor_group=np.nan,
    userid_of_shared_user=np.nan,
    auth=np.nan,
    email=np.nan,
    password=np.nan
):

    # get dataset
    data, userid = get_data(
        weeks_of_data=weeks_of_data,
        donor_group=donor_group,
        userid_of_shared_user=userid_of_shared_user,
        auth=auth,
        email=email,
        password=password
    )

    return data
