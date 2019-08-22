#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
description: load environment variables
version: 0.0.1
created: 2018-02-21
author: Ed Nykaza
modified: 2019-08-15 (Jason Meno)
dependencies:
    * .env file in same folder as this script (Tidepool see bigdata 1PWD)
license: BSD-2-Clause
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
    """
    Retrieves an email and password of a Tidepool bigdata donor group account
    from the local environment.

    Args:
        donorGroup (str): The name of the donor group

    Returns:
        emailAddress (str): The email address of the donor group account
        pswd (str): The password of the Tidepool donor group account

    Raises:
        KeyError if the variables are not found in the local environment

    **This function only works if a local .env file is loaded into python**

    This function is used by the following scripts:
        get-donor-data/accept-new-donors-and-get-donor-list.py
        get-donor-data/get_single_donor_metadata.py
        get-donor-data/get_single_tidepool_dataset.py

    """
    try:
        envEmailVariableName = "BIGDATA_" + donorGroup + "_EMAIL"
        emailAddress = os.environ[envEmailVariableName]

        envPasswordVariableName = "BIGDATA_" + donorGroup + "_PASSWORD"
        pswd = os.environ[envPasswordVariableName]

        return emailAddress, pswd

    except KeyError:
        raise KeyError("Details for Donor Group '{0}' not found in .env".format(donorGroup))
