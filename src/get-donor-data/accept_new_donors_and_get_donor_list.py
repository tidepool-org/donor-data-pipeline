# -*- coding: utf-8 -*-
"""
Accept New Donors & Get Donor List
==================================
:File: accept-new-donors-and-get-donor-list.py
:Description: Accepts all pending data share invitations and gets a unique
    donor list
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
import os
import sys
import requests
import json
import argparse
envPath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if envPath not in sys.path:
    sys.path.insert(0, envPath)
import environmentalVariables


# %% FUNCTIONS
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
        **data_path** : str
            A path to the main 'data' storage folder
        **date_stamp** : str
            A YYYY-MM-DD formatted datestamp
        **save_donor_list** : bool
            A True/False for saving the donor list

    Notes
    -----
    Called From:
        - main

    """
    # USER INPUTS (choices to be made in order to run the code)
    codeDescription = "accepts new donors (shares) and return a list of\
        userids"
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

    parser.add_argument(
        "-s",
        "--save-donor-list",
        dest="save_donor_list",
        default=True,
        help="specify if you want to save the donor list (True/False)"
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
        - accept_and_get_list

    """
    if not isinstance(folder_paths, list):
        folder_paths = [folder_paths]
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    return


def login_api(auth):
    r"""Calls API to log into a Tidepool account

    Parameters
    ----------
    auth : tuple
        An ('email', 'password') string tuple of a donor group

    Returns
    -------
    headers : dict
        Contains a session token and application content type
    userid : str
        The userid of the main Tidepool account

    Notes
    -----
    Called From:
        - accept_new_donors_and_get_donor_list

    """
    api_call = "https://api.tidepool.org/auth/login"
    api_response = requests.post(api_call, auth=auth)
    if(api_response.ok):
        xtoken = api_response.headers["x-tidepool-session-token"]
        userid = json.loads(api_response.content.decode())["userid"]
        headers = {
            "x-tidepool-session-token": xtoken,
            "Content-Type": "application/json"
        }

    else:
        sys.exit("Error with " + auth[0] + ":" + str(api_response.status_code))

    print("logging into", auth[0], "...")

    return headers, userid


def logout_api(auth):
    r"""Calls API to log out of a Tidepool account

    Parameters
    ----------
    auth : tuple
          An ('email', 'password') string tuple of a donor group

    Returns
    -------
    None

    Notes
    -----
    Called From:
        - accept_new_donors_and_get_donor_list

    """
    api_call = "https://api.tidepool.org/auth/logout"
    api_response = requests.post(api_call, auth=auth)

    if(api_response.ok):
        print("successfully logged out of", auth[0])

    else:
        sys.exit(
            "Error with logging out for " +
            auth[0] + ":" + str(api_response.status_code)
        )

    return


def accept_invite_api(headers, userid):
    r"""Loops through all pending data-share invites and accepts them

    Parameters
    ----------
    headers : dict
        Contains a session token and application content type
    userid : str
        The userid of the main Tidepool account

    Returns
    -------
    nAccepted : int
        Number of donor invites successfully accepted

    Notes
    -----
    Called From:
        - accept_new_donors_and_get_donor_list

    """
    print("accepting new donors ...")
    nAccepted = 0
    api_call = "https://api.tidepool.org/confirm/invitations/" + userid
    api_response = requests.get(api_call, headers=headers)
    if(api_response.ok):

        usersData = json.loads(api_response.content.decode())

        for i in range(0, len(usersData)):
            shareKey = usersData[i]["key"]
            shareID = usersData[i]["creatorId"]
            payload = {
                "key": shareKey
            }

            api_call2 = "https://api.tidepool.org/confirm/accept/invite/" + \
                userid + "/" + shareID

            api_response2 = requests.put(
                api_call2,
                headers=headers,
                json=payload
            )

            if(api_response2.ok):
                nAccepted = nAccepted + 1
            else:
                sys.exit(
                    "Error with accepting invites",
                    api_response2.status_code
                )

    elif api_response.status_code == 404:
        # this is the case where there are no new invitations
        print("very likely that no new invitations exist")
    else:
        sys.exit(
            "Error with getting list of invitations",
            api_response.status_code
        )

    return nAccepted


def get_donor_list_api(headers, userid):
    r"""Loops through all pending data-share invites and accepts them

    Parameters
    ----------
    headers : dict
        Contains a session token and application content type
    userid : str
        The userid of the main Tidepool account

    Returns
    -------
    df : pandas.DataFrame
        A single column DataFrame containing each userID found in the account

    Notes
    -----
    Called From:
        - accept_new_donors_and_get_donor_list

    """
    print("getting donor list ...")
    api_call = "https://api.tidepool.org/access/groups/" + userid
    api_response = requests.get(api_call, headers=headers)
    if(api_response.ok):
        donors_list = json.loads(api_response.content.decode())
    else:
        sys.exit(
            "Error with donor list api",
            api_response.status_code
        )
    df = pd.DataFrame(list(donors_list.keys()), columns=["userID"])

    return df


def accept_new_donors_and_get_donor_list(auth):
    r"""Accept new donors in a donor group and gets a list of their userIDs

    This function is a wrapper which calls other functions to log into a
    donor group account, accept pending data-share invitations, get the list of
    all donor userIDs within that account, and then logout.

    Parameters
    ----------
    auth : tuple
        An ('email', 'password') string tuple of a donor group

    Returns
    -------
    nAccepted : int
        The number of accepted data-share invitations
    df : pandas.DataFrame
        A single column DataFrame containing each userID found in the account

    Notes
    -----
    Called From:
        - accept_and_get_list

    Calls To:
        - login_api
        - accept_invite_api
        - get_donor_list_api
        - logout_api

    """
    # login
    headers, userid = login_api(auth)
    # accept invitations to the master donor account
    nAccepted = accept_invite_api(headers, userid)
    # get a list of donors associated with the master account
    df = get_donor_list_api(headers, userid)
    # logout
    logout_api(auth)

    return nAccepted, df


# %% START OF CODE
def accept_and_get_list(args):
    r"""Accepts all pending data share invitations and gets a unique donor list

    When a Tidepool user opts into donating their data, a data share-invite
    is sent to the big data Tidepool account (a special type of clinician
    account). In addition, Tidepool data donors get to choose an organization
    to support (e.g. JDRF, BT1, etc). Each organization donor group is tracked
    using the same data-share invite mechanism. These invitation requests
    need to be accepted before the data becomes available to download.

    This is the main function which loops through each donor group account,
    accepts the pending invitations, removes duplicate and QA test accounts,
    and then saves the unique donor list to a file.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace of arguments parsed by the argparse package.

        This namespace includes the following:
            **data_path** : str
                A path to the main 'data' storage folder
            **date_stamp** : str
                A YYYY-MM-DD formatted datestamp
            **save_donor_list** : bool
                A True/False for saving the donor list

    Returns
    -------
    final_donor_list : pandas.DataFrame
        A 2 column DataFrame containing every unique userID and the donorGroup
        account they can be found in.

    Notes
    -----
    Called From:
        - main

    Calls To:
        - make_folder_if_doesnt_exist
        - environmentalVariables.get_environmental_variables
        - accept_new_donors_and_get_donor_list

    If save_donor_list == True, the final_donor_list is saved to a csv, and it
    is also returned within memory.

    """
    # create output folders
    date_stamp = args.date_stamp  # dt.datetime.now().strftime("%Y-%m-%d")
    phi_date_stamp = "PHI-" + date_stamp
    donor_folder = os.path.join(args.data_path, phi_date_stamp + "-donor-data")
    make_folder_if_doesnt_exist(donor_folder)

    uniqueDonorList_path = os.path.join(
        donor_folder,
        phi_date_stamp + "-uniqueDonorList.csv"
    )

    # define the donor groups
    donor_groups = [
        "bigdata", "AADE", "BT1", "carbdm", "CDN",
        "CWD", "DHF", "DIATRIBE", "diabetessisters",
        "DYF", "JDRF", "NSF", "T1DX",
    ]

    all_donors_df = pd.DataFrame(columns=["userID", "donorGroup"])

    # accounts to ignore (QA testing)
    accounts_to_ignore = [
        'f597f21dcd', '0ef51a0121', '38c3795fcb', '69c99b51f6', '84c2cdd947',
        '9cdebdc316', '9daaf4d4c1', 'bdf4724bed', 'c7415b5097', 'dccc3baf63',
        'ee145393b0', '00cd0ffada', '122a0bf6c5', '898c3d8056', '9e4f3fbc2a',
        '1ebe2a2790', '230650bb9c', '3f8fdabcd7', '636aad0f58', '70df39aa43',
        '92a3c903fe', '3043996405', '0239c1cfb2', '03852a5acc', '03b1953135',
        '0ca5e75e4a', '0d8bdb05eb', '19123d4d6a', '19c25d34b5', '1f6866bebc',
        '1f851c13a5', '275ffa345f', '275ffa345f', '3949134b4a', '410865ba56',
        '57e2b2ed3d', '59bd6891e9', '5acf17a80a', '627d0f4bf1', '65247f8257',
        '6e5287d4c4', '6fc3a4ad44', '78ea6c3cad', '7d8a80e8ce', '8265248ea3',
        '8a411facd2', '98f81fae18', '9d601a08a3', 'aa9fbc4ef5', 'aaac56022a',
        'adc00844c3', 'aea4b3d8ea', 'bc5ee641a3', 'c8328622d0', 'cfef0b91ac',
        'df54366b1c', 'e67aa71493', 'f2103a44d5', 'dccc3baf63'
    ]

    for donor_group in donor_groups:
        if donor_group == "bigdata":
            dg = ""
        else:
            dg = donor_group

        nNewDonors, donors_df = accept_new_donors_and_get_donor_list(
            environmentalVariables.get_environmental_variables(dg)
        )

        donors_df["donorGroup"] = donor_group
        print(donor_group, "complete, there are %d new donors\n" % nNewDonors)
        all_donors_df = pd.concat([all_donors_df, donors_df])

    all_donors_df.sort_values(by=['userID', 'donorGroup'], inplace=True)
    unique_donors = all_donors_df.loc[~all_donors_df["userID"].duplicated()]
    total_donors = len(set(unique_donors["userID"]) - set(accounts_to_ignore))

    final_donor_list = pd.DataFrame(
        list(set(unique_donors["userID"]) - set(accounts_to_ignore)),
        columns=["userID"]
    )

    final_donor_list = pd.merge(
        final_donor_list,
        unique_donors,
        how="left",
        on="userID"
    )

    # polish up the final donor list
    final_donor_list.sort_values(by="donorGroup", inplace=True)
    final_donor_list.reset_index(drop=True, inplace=True)

    if args.save_donor_list:
        print("saving donor list ...\n")
        final_donor_list.to_csv(uniqueDonorList_path)
    else:
        print("donor list is NOT being saved ...\n")

    print("There are %d total donors," % total_donors)
    print("after removing donors that donated to more than 1 group,")
    print("and after removing QA testing accounts.")

    return final_donor_list


def main():
    r"""Main function for accept_new_donors_and_get_donor_list.py

    Parameters
    ----------
    None

    Returns
    -------
    final_donor_list : pandas.DataFrame
        A 2 column DataFrame containing every unique userID and the donorGroup
        account they can be found in.

    Notes
    -----
    Called From:
        - __main__

    Calls To:
        - parse_arguments
        - accept_and_get_list

    """

    args = parse_arguments()
    final_donor_list = accept_and_get_list(args)

    return final_donor_list


if __name__ == "__main__":
    final_donor_list = main()
