# -*- coding: utf-8 -*-
"""get_single_donor_notes.py
Downloads all then notes for a dataset from the Tidepool API.
"""

# %% REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import sys
import getpass
import requests
import json
import argparse
import datetime
import orjson
import environmentalVariables


# %% FUNCTIONS
def get_args():
    codeDescription = "Download notes from a single Tidepool donor account"

    parser = argparse.ArgumentParser(description=codeDescription)

    parser.add_argument(
        "-userid_of_shared_user",
        dest="userid_of_shared_user",
        default=np.nan,
        help="Optional Tidepool userid to download, default is master account",
    )

    parser.add_argument(
        "-weeks_of_data",
        dest="weeks_of_data",
        default=52 * 10,
        help="Number of weeks of data to collect (default 10 years)",
    )

    parser.add_argument(
        "-donor_group", dest="donor_group", default=np.nan, help="Optional Tidepool donor_group to download data from",
    )

    parser.add_argument(
        "-session_token",
        dest="session_token",
        default=np.nan,
        help="Optional reusable session xtoken used for downloading data",
    )

    parser.add_argument(
        "-export_dir", dest="export_dir", default="", help="The export directory to save data to (Default current dir)",
    )

    parser.add_argument(
        "--mode", dest="_", default="_", help="Temp PyCharm console arg",
    )

    parser.add_argument(
        "--port", dest="_", default="_", help="Temp PyCharm console arg",
    )

    args = parser.parse_args()

    return args


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
    if api_response.ok:
        xtoken = api_response.headers["x-tidepool-session-token"]
        userid_master = json.loads(api_response.content.decode())["userid"]
        print("successfully established session for", auth[0])
    else:
        sys.exit("Error with " + auth[0] + ":" + str(api_response.status_code))

    return xtoken, userid_master


def get_notes_api_call(userid, startDate, endDate, headers):

    api_call = (
        "https://app.tidepool.org/message/notes/"
        + userid
        + "?endDate="
        + endDate
        + "&startDate="
        + startDate
    )

    api_response = requests.get(api_call, headers=headers)

    if api_response.ok:
        json_data = json.loads(api_response.content.decode())

    elif api_response.status_code == 404:
        json_data = []
        pass

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

    return json_data


def logout(auth):
    api_call = "https://api.tidepool.org/auth/logout"
    api_response = requests.post(api_call, auth=auth)

    if api_response.ok:
        print("successfully logged out of", auth[0])
        pass

    else:
        sys.exit("Error with logging out for " + auth[0] + ":" + str(api_response.status_code))

    return


# %% Main Function Call
def get_donor_notes(
    donor_group=np.nan,
    userid_of_shared_user=np.nan,
    auth=np.nan,
    email=np.nan,
    password=np.nan,
    session_token=np.nan,
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
    notes_df : list or pandas.DataFrame
        The complete donor notes dataframe from the API
    userid_of_shared_user : str
        The userid of the account downloaded from (will be same as master if none specified)
    """

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

    headers = {"x-tidepool-session-token": xtoken, "Content-Type": "application/json"}

    endDate = datetime.datetime.utcnow()
    startDate = endDate - pd.Timedelta(365 * 10, unit="d")

    startDate = startDate.strftime("%Y-%m-%d") + "T00:00:00.000Z"
    endDate = endDate.strftime("%Y-%m-%d") + "T23:59:59.999Z"

    notes = get_notes_api_call(userid_of_shared_user, startDate, endDate, headers)
    notes_df = pd.DataFrame()

    if 'messages' in notes:
        notes_df = pd.DataFrame(notes['messages'])

    if pd.isnull(session_token):
        logout(auth)

    return notes_df, userid_of_shared_user


# %%
if __name__ == "__main__":

    data_args = get_args()

    try:
        notes_df, dataset_userid = get_donor_notes(
            donor_group=data_args.donor_group,
            userid_of_shared_user=data_args.userid_of_shared_user,
            session_token=data_args.session_token,
        )

        if len(notes_df) > 0:
            filename = data_args.export_dir + "PHI-" + dataset_userid + "-notes.csv"
            notes_df.to_csv(filename, index=False)

        else:
            # Append userid to list of empty datasets
            empty_dataset_list = open("PHI-empty-notes-accounts.txt", "a")
            empty_dataset_list.write(dataset_userid + "\n")
            empty_dataset_list.close()

    except Exception as e:
        print("~~~~~~~~~~~Exception Captured Below~~~~~~~~~~~~")
        print("FAILED TO GET DATA FOR " + str(data_args.userid_of_shared_user))
        print(e)
        print("\n")

        failed_dataset_list = open("PHI-failed-notes-accounts.txt", "a")
        failed_dataset_list.write(str(data_args.userid_of_shared_user) + "\n")
        failed_dataset_list.close()
