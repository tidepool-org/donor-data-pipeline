# -*- coding: utf-8 -*-
"""
Download Tidepool Account Device Data
=====================================
:File: get_single_tidepool_dataset.py
:Description: Downloads all a Tidepool account's device data
:Version: 0.0.1
:Created: 2019-05-26
:Authors: Ed Nykaza (etn), Jason Meno (jam)
:Last Modified: 2019-08-26 (jam)
:Dependencies:
    - .env
    - environmentalVariables.py
:License: BSD-2-Clause

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
envPath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if envPath not in sys.path:
    sys.path.insert(0, envPath)
import environmentalVariables


# %% Function Definitions
def parse_arguments():
    r"""Parses command line arguments

    Parameters
    ----------
    None

    Returns
    -------
    args : argparse.Namespace
    A namespace of arguments parsed by the argparse package.

    This namespace includes the following:
        **date_stamp** : str
            A YYYY-MM-DD formatted datestamp
        **weeks_of_data** : int
            The number of weeks of data you want to download
        **donor_group** : str
            name of the donor group in the tidepool .env file
        **userid_of_shared_user** : str
            userid of account shared with the donor group or master account
        **auth** : tuple
            An ('email', 'password') string tuple of a Tidepool account
        **email** : str
            Master account email
        **password** : str
            Master account password
        **data_path** : str
            The absolute output path where the data is stored

    Notes
    -----
    Called From:
        - main

    """
    codeDescription = "get donor metadata"
    parser = argparse.ArgumentParser(description=codeDescription)

    parser.add_argument(
        "-d",
        "--date-stamp",
        dest="date_stamp",
        default=dt.datetime.now().strftime("%Y-%m-%d"),
        help="date, in '%Y-%m-%d' format, of the date when " +
        "donors were accepted"
    )

    parser.add_argument(
        "-w",
        "--weeks-of-data",
        dest="weeks_of_data",
        default=52*10,
        help="enter the number of weeks of data you want to download"
    )

    parser.add_argument(
        "-dg",
        "--donor-group",
        dest="donor_group",
        default=np.nan,
        help="name of the donor group in the tidepool .env file"
    )

    parser.add_argument(
        "-u",
        "--userid",
        dest="userid_of_shared_user",
        default=np.nan,
        help="userid of account shared with the donor group or master account"
    )

    parser.add_argument(
        "-a",
        "--auth",
        dest="auth",
        default=np.nan,
        help="tuple that contains (email, password)"
    )

    parser.add_argument(
        "-e",
        "--email",
        dest="email",
        default=np.nan,
        help="email address of the master account"
    )

    parser.add_argument(
        "-p",
        "--password",
        dest="password",
        default=np.nan,
        help="password of the master account"
    )

    parser.add_argument(
        "-o",
        "--output-data-path",
        dest="data_path",
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "data"
            )
        ),
        help="the output path where the data is stored"
    )

    args = parser.parse_args()

    return args


def make_folder_if_doesnt_exist(folder_paths):
    r"""Makes folder directories from any folder path string or list of folder
    path strings.

    Parameters
    ----------
    folder_paths : str or list
        The folder paths to be made

    Returns
    -------
    None

    Notes
    -----
    Called From:
        - get_and_save_metadata

    """
    if not isinstance(folder_paths, list):
        folder_paths = [folder_paths]
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    return


def get_data_api(userid, startDate, endDate, headers):
    r"""Sends data request to Tidepool API

    Parameters
    ----------
    userid : str
        userid of account shared with the donor group or master account
    startDate : datetime.datetime
        The starting date for data to be gathered
    endDate : datetime.datetime or pandas._libs.tslibs.timestamps.Timestamp
        The ending date for data to be gathered (<= 1 year from startDate)


    Returns
    -------
    df : pandas.DataFrame
        A very large dataframe containg up to one year of Tidepool account
        device data
    endDate : pandas._libs.tslibs.timestamps.Timestamp
        The next end date to be gathered, which is the startDate - 1 day

    Notes
    -----
    Called From:
        - get_data

    """
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
        print("getting data between %s and %s" % (startDate, endDate))

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
    r"""Logs into Tidepool account, requests one year of data at time, concats
        this data together, and then logs out

    Parameters
    ----------
    weeks_of_data : int
        The number of weeks of data you want to download
    donor_group : str
        name of the donor group in the tidepool .env file
    userid_of_shared_user : str
        userid of account shared with the donor group or master account
    auth : tuple
        An ('email', 'password') string tuple of a Tidepool account
    email : str
        Master account email
    password : str
        Master account password


    Returns
    -------
    df : pandas.DataFrame
        A very large dataframe containg all concatenated Tidepool account
        device data
    userid_of_shared_user : str
        userid of account shared with the donor group or master account

    Notes
    -----
    Called From:
        - get_and_save_dataset
    Calls To:
        - environmentalVariables.get_environmental_variables
        - get_data_api

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
        print(
            "getting data for the master account since no shared " +
            "user account was given"
        )

    print("logging into", auth[0], "...")

    # download user data
    print("downloading data ...")
    df = pd.DataFrame()
    endDate = pd.datetime.now() + pd.Timedelta(1, unit="d")

    if weeks_of_data > 52:
        years_of_data = int(np.floor(weeks_of_data/52))

        for years in range(0, years_of_data + 1):
            startDate = pd.datetime(
                endDate.year - 1,
                endDate.month,
                endDate.day + 1
            )
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
        print("successfully logged out of", auth[0])

    else:
        sys.exit(
            "Error with logging out for " +
            auth[0] + ":" + str(api_response.status_code)
        )

    return df, userid_of_shared_user


def get_and_save_dataset(
    date_stamp=np.nan,
    data_path=np.nan,
    weeks_of_data=np.nan,
    donor_group=np.nan,
    userid_of_shared_user=np.nan,
    auth=np.nan,
    email=np.nan,
    password=np.nan
):
    r"""Calls other functions to create folders, gather data, and then saves it

    Parameters
    ----------
    date_stamp : str
        A YYYY-MM-DD formatted datestamp
    data_path : str
        The absolute output path where the data is stored
    weeks_of_data : int
        The number of weeks of data you want to download
    donor_group : str
        name of the donor group in the tidepool .env file
    userid_of_shared_user : str
        userid of account shared with the donor group or master account
    auth : tuple
        An ('email', 'password') string tuple of a Tidepool account
    email : str
        Master account email
    password : str
        Master account password


    Returns
    -------
    None

    Notes
    -----
    Called From:
        - main
    Calls To:
        - make_folder_if_doesnt_exist
        - get_data

    """
    # create output folders if they don't exist

    phi_date_stamp = "PHI-" + date_stamp
    donor_folder = os.path.join(data_path, phi_date_stamp + "-donor-data")

    dataset_path = os.path.join(
        donor_folder,
        phi_date_stamp + "-csvData"
    )
    make_folder_if_doesnt_exist(dataset_path)

    # get dataset
    data, userid = get_data(
        weeks_of_data=weeks_of_data,
        donor_group=donor_group,
        userid_of_shared_user=userid_of_shared_user,
        auth=auth,
        email=email,
        password=password
    )

    # save data
    dataset_output_path = os.path.join(
        dataset_path,
        'PHI-' + userid + ".csv"
    )

    data.to_csv(dataset_output_path)


def main():
    r"""Main function for get_single_tidepool_dataset.py

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Called From:
        - __main__

    Calls To:
        - parse_arguments
        - get_and_save_dataset

    Creates:
        data/PHI-YYYY-MM-DD-donor-data/PHI-YYYY-MM-DD-csvData
            PHI-<userid>.csv

        This file can be up to several GB in size

    """
    args = parse_arguments()
    get_and_save_dataset(
        date_stamp=args.date_stamp,
        data_path=args.data_path,
        weeks_of_data=args.weeks_of_data,
        donor_group=args.donor_group,
        userid_of_shared_user=args.userid_of_shared_user,
        auth=args.auth,
        email=args.email,
        password=args.password
    )

    return


# %% Main Call
if __name__ == "__main__":
    main()
