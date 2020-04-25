# -*- coding: utf-8 -*-
"""get_donor_data_and_metadata.py
In the context of the big data donation
project, this code grabs donor data and metadata.

This code calls accept_new_donors_and_get_donor_list.py
to get the most recent donor list
"""

# %% REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import sys
import getpass
import requests
import json
import argparse
import environmentalVariables


# %% FUNCTIONS
def get_args():
    codeDescription = "Download a single Tidepool Dataset"

    parser = argparse.ArgumentParser(description=codeDescription)

    parser.add_argument("-userid_of_shared_user",
                        dest="userid_of_shared_user",
                        default=np.nan,
                        help="Tidepool userid to download")

    parser.add_argument("-max_chunk_size",
                        dest="max_chunk_size",
                        default=182,
                        help="Maximum number of days in each API data request")

    parser.add_argument("-weeks_of_data",
                        dest="weeks_of_data",
                        default=52*10,
                        help="Number of weeks of data to collect")

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
                        help="The export directory to save data to")

    parser.add_argument("-return_raw_json",
                        dest="return_raw_json",
                        default=False,
                        help="Return raw JSON, otherwise returns a dataframe.")

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


def find_data_start_year(userid, headers, start_year=np.nan):

    dates = pd.date_range('2010-01-01', end='today', freq='AS-JAN')
    date_strings = list(dates.strftime('%Y-%m-%d'))
    today_timestamp = pd.datetime.utcnow().strftime('%Y-%m-%d')
    date_strings.append(today_timestamp)

    for date_loc in range(len(date_strings)):

        date = date_strings[date_loc]

        startDate = date + "T00:00:00.000Z"
        endDate = date + "T23:59:59.999Z"

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

            if(len(json_data) > 0):
                # print(date + " has data!")
                # Data exists somewhere between the current and previous dates
                # Set start date to previous date
                start_year = date_strings[max(0, date_loc-1)]
                break

        else:
            sys.exit(
                "ERROR in checking first record data for "
                + str(userid)
                + " API RESPONSE: "
                + str(api_response.status_code)
            )

    return start_year


def check_dataset_for_uploads(userid, headers):

    # Get maximum of 10 years of data
    max_endDate = pd.datetime.utcnow()
    max_startDate = max_endDate - pd.Timedelta(365*10, unit="d")

    max_startDate = max_startDate.strftime("%Y-%m-%d") + "T00:00:00.000Z"
    max_endDate = max_endDate.strftime("%Y-%m-%d") + "T23:59:59.999Z"

    api_call = (
        "https://api.tidepool.org/data/" + userid + "?" +
        "endDate=" + max_endDate + "&" +
        "startDate=" + max_startDate + "&" +
        "dexcom=true" + "&" +
        "medtronic=true" + "&" +
        "carelink=true" + "&" +
        "type=upload"
    )

    api_response = requests.get(api_call, headers=headers)

    if(api_response.ok):
        upload_data = json.loads(api_response.content.decode())

        if(len(upload_data) > 0):
            uploads_exist = True
        else:
            print("No uploads exist in account!")
            uploads_exist = False

    else:
        sys.exit(
            "ERROR in getting data for "
            + str(userid)
            + " Between "
            + str(max_startDate)
            + " & "
            + str(max_endDate)
            + " API RESPONSE: "
            + str(api_response.status_code)
        )

    return uploads_exist


def data_api_call(userid, startDate, endDate, headers):

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

    if(api_response.ok):
        print("successfully logged out of", auth[0])
        pass

    else:
        sys.exit(
            "Error with logging out for " +
            auth[0] + ":" + str(api_response.status_code)
        )

    return


# %% Main Function Call
def get_dataset(
    weeks_of_data=10*52,
    max_chunk_size=365,
    donor_group=np.nan,
    userid_of_shared_user=np.nan,
    auth=np.nan,
    email=np.nan,
    password=np.nan,
    session_token=np.nan,
    return_raw_json=False,
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

    data = []

    # Check if dataset is contains data or is empty
    uploads_exist = check_dataset_for_uploads(userid_of_shared_user, headers)

    if uploads_exist:

        data_start_year = find_data_start_year(userid_of_shared_user, headers)
        days_since_data_start = (
                pd.datetime.utcnow() - pd.to_datetime(data_start_year)
        ).days + 1
        days_to_download = weeks_of_data * 7

        if days_since_data_start < days_to_download:
            days_to_download = days_since_data_start

        if days_to_download < max_chunk_size:
            days_per_chunk = days_to_download
        else:
            days_per_chunk = max_chunk_size

        total_chunks = int(np.ceil(days_to_download/days_per_chunk))

        print("Downloading "
              + str(days_to_download)
              + " days of data in "
              + str(days_per_chunk)
              + "-day chunks...",
              end="")

        endDate = pd.datetime.utcnow()
        startDate = endDate - pd.Timedelta(days_per_chunk, unit="d")

        for chunk in range(total_chunks):

            startDate = startDate.strftime("%Y-%m-%d") + "T00:00:00.000Z"
            endDate = endDate.strftime("%Y-%m-%d") + "T23:59:59.999Z"

            json_chunk = data_api_call(
                    userid_of_shared_user,
                    startDate,
                    endDate,
                    headers
                )

            data += json_chunk

            endDate = pd.to_datetime(startDate) - pd.Timedelta(1, unit="d")
            startDate = endDate - pd.Timedelta(days_per_chunk, unit="d")

        print("done.")

    if not return_raw_json:
        data = pd.DataFrame(data)

    if pd.isnull(session_token):
        logout(auth)

    return data, userid_of_shared_user


# %%
if __name__ == "__main__":

    data_args = get_args()

    try:
        data, dataset_userid = get_dataset(
                weeks_of_data=data_args.weeks_of_data,
                donor_group=data_args.donor_group,
                max_chunk_size=data_args.max_chunk_size,
                userid_of_shared_user=data_args.userid_of_shared_user,
                session_token=data_args.session_token,
                return_raw_json=data_args.return_raw_json
        )

        if len(data) > 0:
            filename = (
                    data_args.export_dir
                    + "PHI-"
                    + dataset_userid

            )

            if data_args.return_raw_json:
                with open(filename + ".json", 'w') as json_writer:
                    json.dump(data, json_writer)

            else:
                data.to_csv(filename + ".csv.gz",
                            index=False,
                            compression='gzip')
        else:
            # Append userid to list of empty datasets
            empty_dataset_list = open('PHI-empty-accounts.txt', 'a')
            empty_dataset_list.write(dataset_userid + "\n")
            empty_dataset_list.close()

    except Exception as e:
        print("~~~~~~~~~~~Exception Captured Below~~~~~~~~~~~~")
        print("FAILED TO GET DATA FOR " + dataset_userid)
        print(e)
        print("\n")

        failed_dataset_list = open('PHI-failed-accounts.txt', 'a')
        failed_dataset_list.write(dataset_userid + "\n")
        failed_dataset_list.close()
