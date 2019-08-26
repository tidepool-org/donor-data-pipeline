#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load Environment Variables
============================
:File: environmentalVariables.py
:Description: Loads environmental variables from local .env into python memory
:Version: 0.0.1
:Created: 2018-02-21
:Authors: Ed Nykaza (EN), Jason Meno (JM)
:Last Modified: 2019-08-15 (JM)
:Dependencies:
    - .env file in same folder as this script (Tidepool see bigdata 1PWD)
:License: BSD-2-Clause

"""

# %% load in required libraries
import os
from os.path import join, dirname, isfile
from dotenv import load_dotenv


# %% load environmental variables
dotenv_path = join(dirname(__file__), '.env')
if isfile(dotenv_path):
    load_dotenv(dotenv_path)


# %% define functions
def get_environmental_variables(donorGroup):
    r"""Retrieves an email and password of a Tidepool bigdata donor group
    account from the local environment.

    Parameters
    ----------
    donorGroup : str
        The name of the donor group

    Returns
    -------
    emailAddress : str
        The email address of the donor group account
    pswd : str
        The password of the Tidepool donor group account

    Raises
    ------
        KeyError
            If the variables are not found in the local environment

    Notes
    -----
    Called From:
        - get-donor-data/accept-new-donors-and-get-donor-list.py
        - get-donor-data/get_single_donor_metadata.py
        - get-donor-data/get_single_tidepool_dataset.py

    Calls To:
        None

    """
    try:
        envEmailVariableName = "BIGDATA_" + donorGroup + "_EMAIL"
        emailAddress = os.environ[envEmailVariableName]

        envPasswordVariableName = "BIGDATA_" + donorGroup + "_PASSWORD"
        pswd = os.environ[envPasswordVariableName]

        return emailAddress, pswd

    except KeyError:
        raise KeyError("Details for Donor Group '{0}' not found in .env".format(donorGroup))
