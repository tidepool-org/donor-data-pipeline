#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
description: Anonymize and export Tidepool data
version: 0.0.2
created: 2018-05-22
author: Ed Nykaza
dependencies:
    * requires tidepool-analytics environment (see readme for instructions)
    * requires Tidepool data (e.g., PHI-jill-jellyfish.json in example-data folder)
license: BSD-2-Clause
"""

# %% REQUIRED LIBRARIES
import pandas as pd
import datetime as dt
import numpy as np
import os
import sys
import shutil
import glob
import argparse
import hashlib
import ast
import time

MGDL_PER_MMOLL = 18.01559

# %% LOAD DATA FUNCTIONS
def checkInputFile(inputFile):
    if os.path.isfile(inputFile):
        if os.stat(inputFile).st_size > 2:
            if inputFile[-4:] == "json":
                inputData = pd.read_json(inputFile, orient="records")
                fileName = os.path.split(inputFile)[-1][:-5]
            elif inputFile[-4:] == "xlsx":
                inputData = readXlsxData(inputFile)
                fileName = os.path.split(inputFile)[-1][:-5]
            elif inputFile[-3:] == "csv":
                inputData = pd.read_csv(inputFile, low_memory=False)
                fileName = os.path.split(inputFile)[-1][:-4]
            else:
                sys.exit("{0} is not a json, xlsx, or csv".format(inputFile))
        else:
            sys.exit("{0} contains too little data".format(inputFile))
    else:
        sys.exit("{0} does not exist".format(inputFile))

    # if fileName has PHI in it, remove PHI to get userID
    if "PHI" in fileName.upper():
        fileName = fileName[4:]

    return inputData, fileName


# %% FILTER DATA FUNCTIONS
def checkDataFieldList(dataFieldPath):
    if not os.path.isfile(dataFieldPath):
        sys.exit("{0} is not a valid file path".format(dataFieldPath))

    dataFieldExportList = pd.read_csv(dataFieldPath)
    approvedDataFields = \
        list(dataFieldExportList.loc[dataFieldExportList.include.fillna(False),
                                     "dataFieldList"])

    hashSaltFields = list(dataFieldExportList.loc[
            dataFieldExportList.hashNeeded.fillna(False), "dataFieldList"])

    return approvedDataFields, hashSaltFields


def filterByDates(df, startDate, endDate):

    # filter by qualified start & end date, and sort
    df = \
        df[(df.time >= startDate) &
           (df.time <= (endDate + "T23:59:59"))]

    return df


def filterByDatesExceptUploadsAndSettings(df, startDate, endDate):

    # filter by qualified start & end date, and sort
    uploadEventsSettings = df[((df.type == "upload") |
                               (df.type == "deviceEvent") |
                               (df.type == "cgmSettings") |
                               (df.type == "pumpSettings"))]

    theRest = df[~((df.type == "upload") |
                 (df.type == "deviceEvent") |
                 (df.type == "cgmSettings") |
                 (df.type == "pumpSettings"))]

    # Only filter by the time field.
    # Any row that does not have an est.localTime will be dropped.
    #if "est.localTime" in list(df):
    #
    #    theRest = theRest[(theRest["est.localTime"] >= startDate) &
    #                      (theRest["est.localTime"] <=
    #                       (endDate + "T23:59:59"))]
    #else:
    theRest = theRest[(theRest["time"] >= startDate) &
                      (theRest["time"] <= (endDate + "T23:59:59"))]

    df = pd.concat([uploadEventsSettings, theRest])

    return df


def sortColumns(df):
    allSettingsFields = ["basalSchedules",
                         "bgTarget",
                         "bgTargets",
                         "carbRatio",
                         "carbRatios",
                         "insulinSensitivity",
                         "insulinSensitivities"]

    existingSettingsFields = list(set(df) & set(allSettingsFields))
    columnsWithoutSettings = list(set(df) - set(existingSettingsFields))
    columsWithDots = []
    for col in columnsWithoutSettings:
        if "." in col:
            columsWithDots.append(col)
    columnsWithoutSettingsAndDots = list(set(columnsWithoutSettings) - set(columsWithDots))
    newColOrder = sorted(columnsWithoutSettingsAndDots) + sorted(columsWithDots) + \
                  sorted(existingSettingsFields)
    df = df[newColOrder]

    return df


def tempRemoveFields(df):
    removeFields = ["basalSchedules",
                    "bolusSchedules",
                    "bgTarget",
                    "bgTargets",
                    "carbRatio",
                    "carbRatios",
                    "defaultAlerts",
                    "insulinSensitivity",
                    "insulinSensitivities"]
                    #"payload",
                    #"origin"]

    tempRemoveFields = list(set(df) & set(removeFields))
    tempDf = df[tempRemoveFields]
    df = df.drop(columns=tempRemoveFields)

    return df, tempDf


def removeBrackets(df, fieldName):
    if fieldName in list(df):
        df.loc[df[fieldName].notnull(), fieldName] = \
            df.loc[df[fieldName].notnull(), fieldName].str[0]

    return df


def flattenJson(df, columns_to_flatten):

    # Filter columns_to_flatten by only those in the dataset
    columns_to_flatten = list(set(columns_to_flatten) & set(list(df)))

    # remove fields that we don't want to flatten
    # Only use if using the list of all columnHeadings
    # df, holdData = tempRemoveFields(df)

    # get a list of data types of column headings
    # WARNING: Use with caution. Fields like payload and suppressed can
    # end up flattening into hundreds of columns, which may cause a crash
    # columns_to_flatten = list(df)

    # remove [] from annotations field
    # This is fixed bracket removal in the the loop below
    # df = removeBrackets(df, "annotations")

    # loop through each column and append flattened version into newDataFrame
    newDataFrame = pd.DataFrame()

    for colHead in columns_to_flatten:

        # if the df field has embedded json
        is_dict_loc = df[colHead].astype(str).str.contains('{')

        if any(is_dict_loc):
            # print("Flattening " + colHead)
            # grab the data that is in brackets
            jsonBlob = df[colHead][is_dict_loc].astype(str).copy()

            # Remove brackets if first and last values are brackets
            # Good for single-dict values wrapped in a list
            # Bad for list of dicts which become a tuple type when evaluated
            has_brackets = (jsonBlob.str[0] == '[') & (jsonBlob.str[-1] == ']')
            jsonBlob[has_brackets] = jsonBlob[has_brackets].str[1:-1]

            # Evaluate data types
            eval_df = jsonBlob.apply(ast.literal_eval)

            # Only get dict types for flattening - all others are ignored
            is_type_dict = eval_df.apply(lambda x: isinstance(x, dict))
            eval_df = eval_df[is_type_dict]

            # replace only the dict types that will be flattened with nan
            # all non-dict types are left inplace
            df.loc[eval_df.index, colHead] = np.nan

            flattened_df = pd.DataFrame(pd.io.json.json_normalize(eval_df)).add_prefix(colHead + '.')
            flattened_df.set_index(eval_df.index, inplace=True)
            # turn jsonBlob to dataframe (tolist())
            #newDataFrame = pd.concat([newDataFrame, pd.DataFrame(eval_df.tolist(),
            #                            index=eval_df.index).add_prefix(colHead + '.')], axis=1)

            newDataFrame = pd.concat([newDataFrame, flattened_df], axis=1)

    #newColHeadings = list(newDataFrame)
    # put df back into the main dataframe
    # and add the fields that were removed back in
    #dropped_new_columns = set(newColHeadings) - set(dataFieldsForExport)
    #print("Dropping flattened columns: " + str(dropped_new_columns))
    #columnFilter = list(set(newColHeadings) & set(dataFieldsForExport))
    #tempDataFrame = newDataFrame.filter(items=columnFilter)
    df = pd.concat([df, newDataFrame], axis=1)

    return df


def filterByApprovedDataFields(df, dataFieldsForExport):

    colHeadings = list(df)
    columnFilter = list(set(colHeadings) & set(dataFieldsForExport))
    dfExport = df.filter(items=columnFilter)

    # Drop specific data types
    dfExport = dfExport[~(df['type'] == 'reportedState')]

    return dfExport


# %% CLEAN DATA FUNCTIONS
def cleanDurationColumn(df):
    r"""A cleaning function to move all non-float-type durations into their own
    column

    The physicalActivity datatype introduces json blobs into the duration field
    which throws an error when searching for negative durations in the function
    removeNegativeDurations

    Parameters
    ----------
    df : pandas.DataFrame
        A very large dataframe contaning all Tidepool account device data

    Returns
    -------
    df : pandas.DataFrame
        A very large dataframe contaning all Tidepool account device data but
        may contain a new "activityDuration" column

    Notes
    -----
    Called From:
        - removeNegativeDurations

    """
    physicalActivityRows = df['type'] == 'physicalActivity'

    if sum(physicalActivityRows) > 0:
        df["activityDuration"] = np.nan
        df.loc[physicalActivityRows, "activityDuration"] = df.loc[physicalActivityRows, 'duration'].astype(str)
        df.loc[physicalActivityRows, 'duration'] = np.nan

    return df


def removeNegativeDurations(df):
    if "duration" in list(df):
        df = cleanDurationColumn(df)
        nNegativeDurations = sum(df.duration.astype(float) < 0)
        if nNegativeDurations > 0:
            df = df[~(df.duration.astype(float) < 0)]
    else:
        nNegativeDurations = np.nan

    return df, nNegativeDurations


def removeInvalidCgmValues(df):

    nBefore = len(df)
    # remove values < 38 and > 402 mg/dL
    df = df.drop(df[((df.type == "cbg") &
                     (df.value < 2.109284236597303))].index)
    df = df.drop(df[((df.type == "cbg") &
                     (df.value > 22.314006924003046))].index)
    nRemoved = nBefore - len(df)

    return df, nRemoved


def createBolusSchedules(df):
    if "bolus" in list(df):
        bolus_schedule_locations = df["bolus"].astype(str).str.contains("{") == True
        df.loc[bolus_schedule_locations, "bolusSchedules"] = df.loc[bolus_schedule_locations, "bolus"].astype(str).copy()
        df.loc[bolus_schedule_locations, "bolus"] = np.nan

    return df

def expand_entire_dict(ts):
    if "Series" not in type(ts).__name__:
        raise TypeError('Expecting a pandas time series object')
    notnull_idx = ts.index[ts.notnull()]
    temp_df = pd.DataFrame(
        ts[notnull_idx].tolist(),
        index=notnull_idx
    )

    return temp_df


def expand_embedded_dict(ts, key_):
    '''Expanded a single field that has embedded json
    Args:
        ts: a pandas time series of the field that has embedded json
        key_: the key that you want to expand
    Raise:
        TypeError: if you don't pass in a pandas time series
    Returns:
        key_ts: a new time series of the key of interest
    NOTE:
        this is new function
    TODO:
        could be refactored to allow multiple keys or all keys to be returned
        could be refactored for speed as the current process
    '''

    if "Series" not in type(ts).__name__:
        raise TypeError('Expecting a pandas time series object')
    key_ts = pd.Series(name=ts.name + "." + key_, index=ts.index)
    notnull_idx = ts.notnull()
    # TODO: maybe sped up by only getting the one field of interest?
    # though, the current method is fairly quick and compact
    temp_df = expand_entire_dict(ts)
    if key_ in list(temp_df):
        key_ts[notnull_idx] = temp_df[key_].values

    return key_ts


def tslim_calibration_fix(df):
    '''
    taken from https://github.com/tidepool-org/data-analytics/blob/
    etn/get-settings-and-events/projects/get-donors-pump-settings/
    get-users-settings-and-events.py
    refactored name(s) to meet style guide
    refactored pandas field call to df["field"] instead of df.field
    refactored to only expand one field
    '''

    # expand payload field one level
    if "payload" in list(df):
        df["payload.calibration_reading"] = (
            expand_embedded_dict(df["payload"], "calibration_reading")
        )

        if df["payload.calibration_reading"].notnull().sum() > 0:

            search_for = ['tan']
            tandem_data_index = (
                (df["deviceId"].str.contains('|'.join(search_for)))
                & (df["type"] == "deviceEvent")
            )

            cal_index = df["payload.calibration_reading"].notnull()
            valid_index = tandem_data_index & cal_index

            n_cal_readings = sum(valid_index)

            if n_cal_readings > 0:
                # if reading is > 30 then it is in the wrong units
                if df["payload.calibration_reading"].min() > 30:
                    df.loc[cal_index, "value"] = (
                        df.loc[valid_index, "payload.calibration_reading"]
                        / MGDL_PER_MMOLL
                    )
                else:
                    df.loc[cal_index, "value"] = (
                        df.loc[valid_index, "payload.calibration_reading"]
                    )
        else:
            n_cal_readings = 0
    else:
        n_cal_readings = 0
    return df, n_cal_readings


#%% CGM Specific Cleaning
def add_uploadDateTime(df):
    r"""Adds an "uploadTime" column to the dataframe and corrects missing
    upload times to records from healthkit data

    Parameters
    ----------
    df : pandas.DataFrame
        A very large dataframe contaning all Tidepool account device data


    Returns
    -------
    df : pandas.DataFrame
        The same dataframe as the input but with an added "uploadTime" column


    Notes
    -----
    Called From:
        - main

    """
    if "upload" in df.type.unique():
        uploadTimes = pd.DataFrame(
            df[df.type == "upload"].groupby("uploadId").time.describe()["top"]
        )
    else:
        uploadTimes = pd.DataFrame(columns=["top"])
    # if an upload does not have an upload date, then add one
    # NOTE: this is a new fix introduced with healthkit data...we now have
    # data that does not have an upload record
    unique_uploadIds = set(df["uploadId"].unique())
    unique_uploadRecords = set(
        df.loc[df["type"] == "upload", "uploadId"].unique()
    )
    uploadIds_missing_uploadRecords = unique_uploadIds - unique_uploadRecords

    # Old method that will hang on datasets with many uploadIds
    #for upId in uploadIds_missing_uploadRecords:
    #    last_upload_time = df.loc[df["uploadId"] == upId, "time"].max()
    #    uploadTimes.loc[upId, "top"] = last_upload_time

    uploadTimes.reset_index(inplace=True)
    uploadTimes.rename(
        columns={
            "top": "uploadTime",
            "index": "uploadId"
        },
        inplace=True
    )
    # New method
    missing_uploads_df = df.loc[df['uploadId'].isin(uploadIds_missing_uploadRecords), ['uploadId', 'time']]
    last_upload_time = missing_uploads_df.groupby('uploadId').time.max()
    last_upload_time = pd.DataFrame(last_upload_time).reset_index()
    last_upload_time.columns = ["uploadId", "uploadTime"]
    uploadTimes = pd.concat([uploadTimes, last_upload_time]).reset_index(drop=True)

    df = pd.merge(df, uploadTimes, how='left', on='uploadId')
    df["uploadTime"] = pd.to_datetime(df["uploadTime"], utc=True)
    df["uploadTime"] = df["uploadTime"].dt.tz_localize(None)

    return df


def round_time(df, timeIntervalMinutes=5, timeField="time",
               roundedTimeFieldName="roundedTime", startWithFirstRecord=True,
               verbose=False):
    '''
    A general purpose round time function that rounds the "time"
    field to nearest <timeIntervalMinutes> minutes
    INPUTS:
        * a dataframe (df) that contains a time field that you want to round
        * timeIntervalMinutes (defaults to 5 minutes given that most cgms output every 5 minutes)
        * timeField to round (defaults to the UTC time "time" field)
        * roundedTimeFieldName is a user specified column name (defaults to roundedTime)
        * startWithFirstRecord starts the rounding with the first record if True, and the last record if False (defaults to True)
        * verbose specifies whether the extra columns used to make calculations are returned
    '''

    df.sort_values(by=timeField, ascending=startWithFirstRecord, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # make sure the time field is in the right form
    t = pd.to_datetime(df[timeField].astype('datetime64[ns]'))

    # calculate the time between consecutive records
    t_shift = pd.to_datetime(df[timeField].astype('datetime64[ns]').shift(1))
    df["timeBetweenRecords"] = \
        round((t - t_shift).dt.days*(86400/(60 * timeIntervalMinutes)) +
              (t - t_shift).dt.seconds/(60 * timeIntervalMinutes)) * timeIntervalMinutes

    # separate the data into chunks if timeBetweenRecords is greater than
    # 2 times the <timeIntervalMinutes> minutes so the rounding process starts over
    largeGaps = list(df.query("abs(timeBetweenRecords) > " + str(timeIntervalMinutes * 2)).index)
    largeGaps.insert(0, 0)
    largeGaps.append(len(df))

    for gIndex in range(0, len(largeGaps) - 1):
        chunk = t[largeGaps[gIndex]:largeGaps[gIndex+1]]
        firstRecordChunk = t[largeGaps[gIndex]]

        # calculate the time difference between each time record and the first record
        df.loc[largeGaps[gIndex]:largeGaps[gIndex+1], "minutesFromFirstRecord"] = \
            (chunk - firstRecordChunk).dt.days*(86400/(60)) + (chunk - firstRecordChunk).dt.seconds/(60)

        # then round to the nearest X Minutes
        # NOTE: the ".000001" ensures that mulitples of 2:30 always rounds up.
        df.loc[largeGaps[gIndex]:largeGaps[gIndex+1], "roundedMinutesFromFirstRecord"] = \
            round((df.loc[largeGaps[gIndex]:largeGaps[gIndex+1],
                          "minutesFromFirstRecord"] / timeIntervalMinutes) + 0.000001) * (timeIntervalMinutes)

        roundedFirstRecord = (firstRecordChunk + pd.Timedelta("1microseconds")).round(str(timeIntervalMinutes) + "min")
        df.loc[largeGaps[gIndex]:largeGaps[gIndex+1], roundedTimeFieldName] = \
            roundedFirstRecord + \
            pd.to_timedelta(df.loc[largeGaps[gIndex]:largeGaps[gIndex+1],
                                   "roundedMinutesFromFirstRecord"], unit="m")

    # sort by time and drop fieldsfields
    df.sort_values(by=timeField, ascending=startWithFirstRecord, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if verbose is False:
        df.drop(columns=["timeBetweenRecords",
                         "minutesFromFirstRecord",
                         "roundedMinutesFromFirstRecord"], inplace=True)

    return df


def removeDuplicates(df, criteriaDF):
    # ONLY PASS IN NON-NULL VALUES FOR the main time criterion!

    nBefore = len(df)
    df = df.loc[~(df[criteriaDF].duplicated())]
    df = df.reset_index(drop=True)
    nDuplicatesRemoved = nBefore - len(df)

    return df, nDuplicatesRemoved


def removeCgmDuplicates(df, timeCriterion):
    if timeCriterion in df:
        df.sort_values(by=[timeCriterion, "uploadTime"],
                       ascending=[False, False],
                       inplace=True)
        dfIsNull = df[df[timeCriterion].isnull()]
        dfNotNull = df[df[timeCriterion].notnull()]
        dfNotNull, nDuplicatesRemoved = removeDuplicates(dfNotNull, [timeCriterion, "value"])
        df = pd.concat([dfIsNull, dfNotNull])
        df.sort_values(by=[timeCriterion, "uploadTime"],
                       ascending=[False, False],
                       inplace=True)
    else:
        nDuplicatesRemoved = 0

    return df, nDuplicatesRemoved


def removeCgmRoundedTimeDuplicates(df):
    if "roundedTime" in df:
        df.sort_values(by=["roundedTime", "uploadTime"],
                       ascending=[False, False],
                       inplace=True)
        dfIsNull = df[df["roundedTime"].isnull()]
        dfNotNull = df[df["roundedTime"].notnull()]
        dfNotNull, nRoundedTimeDuplicatesRemoved = removeDuplicates(dfNotNull, ["roundedTime"])
        df = pd.concat([dfIsNull, dfNotNull])
        df.sort_values(by=["roundedTime", "uploadTime"],
                       ascending=[False, False],
                       inplace=True)
    else:
        nRoundedTimeDuplicatesRemoved = 0

    return df, nRoundedTimeDuplicatesRemoved


# %% ANONYMIZE DATA FUNCTIONS
def hashScheduleNames(df, salt, userID):

    scheduleNames = ["basalSchedules",
                     "bgTargets",
                     "carbRatios",
                     "insulinSensitivities",
                     "bolusSchedules"]

    nameKeys = set()

    # loop through each of the scheduleNames that exist
    for scheduleName in scheduleNames:
        # if scheduleName exists, find the rows that have the scheduleName
        if scheduleName in list(df):
            scheduleNameDataFrame = df[df[scheduleName].notnull()].copy()
            scheduleNameRows = scheduleNameDataFrame[scheduleName].index

            # loop through each schedule name row
            for scheduleNameRow in scheduleNameRows:
                # this is for the csv version, which loads the data as string
                if isinstance(scheduleNameDataFrame.loc[scheduleNameRow, scheduleName], str):
                    scheduleNameDataFrame.loc[scheduleNameRow, [scheduleName]] = \
                        [ast.literal_eval(scheduleNameDataFrame.loc[scheduleNameRow, scheduleName])]

                scheduleNameKeys = \
                    list(scheduleNameDataFrame[scheduleName]
                    [scheduleNameRow].keys())

                # Some bolusSchedules have no keys, and this helps prevent the
                # internal values from being hashed
                if scheduleName == "bolusSchedules":
                    scheduleNameKeys = set(nameKeys) & set(scheduleNameKeys)
                else:
                    nameKeys = nameKeys | set(scheduleNameKeys) - set(nameKeys)

                # loop through each key and replace with hashed version
                for scheduleNameKey in scheduleNameKeys:
                    if(type(scheduleNameKey) == int):
                        pass
                        #print(scheduleName + " scheduleNameKey: " + str(scheduleNameKey) + " (" + str(type(scheduleNameKey)) + ")")
                    #print("args.salt: " + str(args.salt) + " (" + str(type(args.salt)) + ")")
                    if(type(userID) == int):
                        pass
                        #print("userID: " + str(userID) + " (" + str(type(userID)) + ")")
                    hashedScheduleName = \
                    hashlib.sha256((str(scheduleNameKey) + str(salt) + str(userID)).
                               encode()).hexdigest()[0:8]
                    scheduleNameDataFrame[scheduleName][scheduleNameRow][hashedScheduleName] = \
                        scheduleNameDataFrame[scheduleName][scheduleNameRow].pop(scheduleNameKey)

            # drop and reattach the new data
            df = df.drop(columns=scheduleName)
            df = pd.merge(df, scheduleNameDataFrame.loc[:, ["id", scheduleName]], how="left", on="id")
    return df


def hashData(df, columnHeading, lengthOfHash, salt, userID):

    df[columnHeading] = \
        (df[columnHeading].astype(str) + salt + userID).apply(
        lambda s: hashlib.sha256(s.encode()).hexdigest()[0:lengthOfHash])

    return df


def anonymizeData(df, hashSaltFields, salt, userID):

    for hashSaltField in hashSaltFields:
        if hashSaltField in df.columns.values:
            df.loc[df[hashSaltField].notnull(), hashSaltField] = \
                hashData(pd.DataFrame(df.loc[df[hashSaltField].notnull(),
                                             hashSaltField]),
                         hashSaltField, 8, salt, userID)

    # also hash the schedule names
    df = hashScheduleNames(df, salt, userID)

    return df


def hashUserId(userID, salt):

    usr_string = userID + salt
    hash_user = hashlib.sha256(usr_string.encode())
    hashID = hash_user.hexdigest()

    return hashID


# %% EXPORT DATA FUNCTIONS
def filterAndSort(groupedDF, filterByField, sortByField):
    filterDF = groupedDF.get_group(filterByField).dropna(axis=1, how="all")
    filterDF = filterDF.sort_values(sortByField)
    return filterDF


def removeManufacturersFromAnnotationsCode(df):
    # remove manufacturer from annotations.code
    manufacturers = ["animas/",
                     "bayer/",
                     "carelink/",
                     "insulet/",
                     "medtronic/",
                     "medtronic600/",
                     "tandem/"]

    annotationFields = [
        "annotations.code",
        "suppressed.annotations.code",
        "suppressed.suppressed.annotations.code"
        ]

    for annotationField in annotationFields:
        if annotationField in df.columns.values:
            if sum(df[annotationField].notnull()) > 0:
                df[annotationField] = \
                    df[annotationField].str. \
                    replace("|".join(manufacturers), "")

    return df


def mergeWizardWithBolus(df, exportDirectory):

    if (("bolus" in set(df.type)) and ("wizard" in set(df.type))):
        bolusData = pd.read_csv(exportDirectory + "bolus.csv",
                                low_memory=False)
        wizardData = pd.read_csv(exportDirectory + "wizard.csv",
                                 low_memory=False)


        # merge the wizard data with the bolus data
        wizardData["calculatorId"] = wizardData["id"]
        wizardDataFields = [
            "bgInput",
            "bgTarget.start",
            "bgTarget.high",
            "bgTarget.low",
            "bgTarget.range",
            "bgTarget.target",
            "bolus",
            "carbInput",
            "calculatorId",
            "insulinCarbRatio",
            "insulinOnBoard",
            "insulinSensitivity",
            "recommended.carb",
            "recommended.correction",
            "recommended.net",
            "units",
        ]
        keepTheseWizardFields = \
            set(wizardDataFields).intersection(list(wizardData))
        bolusData = pd.merge(bolusData,
                             wizardData[list(keepTheseWizardFields)],
                             how="left",
                             left_on="id",
                             right_on="bolus")

        mergedBolusData = bolusData.drop("bolus", axis=1)
    else:
        mergedBolusData = pd.DataFrame()

    return mergedBolusData


def mergeWizardWithBolusInMemory(df):

    if (("bolus" in set(df.type)) and ("wizard" in set(df.type))):
        bolusData = df[df.type == 'bolus'].copy()
        wizardData = df[df.type == 'wizard'].copy()
        temp_df = df[~((df.type == 'bolus') | (df.type == 'wizard'))]
        # merge the wizard data with the bolus data
        wizardData["calculatorId"] = wizardData["id"]
        wizardDataFields = [
            "bgInput",
            "bgTarget.high",
            "bgTarget.low",
            "bgTarget.range",
            "bgTarget.target",
            "bolus",
            "carbInput",
            "calculatorId",
            "insulinCarbRatio",
            "insulinOnBoard",
            "insulinSensitivity",
            "recommended.carb",
            "recommended.correction",
            "recommended.net",
            "units",
        ]
        keepTheseWizardFields = \
            set(wizardDataFields).intersection(list(wizardData))
        dropBolusColumns = keepTheseWizardFields & set(list(bolusData))
        bolusData.drop(columns=dropBolusColumns, inplace=True)
        bolusData = pd.merge(bolusData,
                             wizardData[list(keepTheseWizardFields)],
                             how="left",
                             left_on="id",
                             right_on="bolus")

        df = pd.concat([temp_df, bolusData], sort=False)
        df = sortColumns(df)
        df = df.drop(columns="bolus")
    else:
        df = sortColumns(df)

    df.reset_index(drop=True, inplace=True)

    return df


def cleanDiretory(exportFolder, fileName):

    # if there is a failure during an export, you will want to clear out
    # the remnants before trying to export again, so delete files if they exist
    hiddenCsvExportFolder = os.path.join(exportFolder,
                                         "." + fileName + "-csvs", "")
    if os.path.exists(hiddenCsvExportFolder):
        shutil.rmtree(hiddenCsvExportFolder)

    os.makedirs(hiddenCsvExportFolder)

    unhiddenCsvExportFolder = os.path.join(exportFolder,
                                           fileName + "-csvs", "")

    for fType in ["xlsx", "json", "csv"]:
        fName = os.path.join(exportFolder, fileName + "." + fType)
        if os.path.exists(fName):
            os.remove(fName)

    # if unhiddenCsvExportFolder folder exists, delete it
    if os.path.exists(unhiddenCsvExportFolder):
        shutil.rmtree(unhiddenCsvExportFolder)

    return hiddenCsvExportFolder


def exportCsvFiles(df, exportFolder, fileName, mergeCalculatorData):
    hiddenCsvExportFolder = cleanDiretory(exportFolder, fileName)
    groupedData = df.groupby(by="type")

    for dataType in set(df[df.type.notnull()].type):
        csvData = filterAndSort(groupedData, dataType, "time")
        csvData = sortColumns(csvData)
        csvData.to_csv(hiddenCsvExportFolder + dataType + ".csv", index=False)

    # merge wizard data with bolus data, and delete wizard data
    if mergeCalculatorData:
        bolusWithWizardData = mergeWizardWithBolus(df, hiddenCsvExportFolder)
        if len(bolusWithWizardData) > 0:
            bolusWithWizardData = sortColumns(bolusWithWizardData)
            bolusWithWizardData.to_csv(hiddenCsvExportFolder + "bolus.csv",
                                       index=False)
        if os.path.exists(hiddenCsvExportFolder + "wizard.csv"):
            os.remove(hiddenCsvExportFolder + "wizard.csv")

    return hiddenCsvExportFolder


def exportSingleCsv(exportFolder, fileName, exportDirectory, fileType):
    # first load in all csv files
    csvFiles = glob.glob(exportDirectory + "*.csv")
    bigTable = pd.DataFrame()
    for csvFile in csvFiles:
        bigTable = pd.concat([bigTable, pd.read_csv(csvFile, low_memory=False)], sort=False)

    # first sort by time and then put columns in alphabetical order
    bigTable = bigTable.sort_values("time")
    bigTable = sortColumns(bigTable)
    if (("csv" in fileType) | ("all" in fileType)):
        bigTable.to_csv(os.path.join(exportFolder, fileName + ".csv"), index=False)

    return bigTable


def formatKeyValue(key, val):
    if str(val) in ["True", "False"]:
        output = '\n  "{0}":{1}'.format(key, str(val).lower())
    elif isinstance(val, str):
        output = '\n  "{0}":"{1}"'.format(key, val)
    else:
        output = '\n  "{0}":{1}'.format(key, val)

    return output


def formatRow(oneRow):
    keyValList = [formatKeyValue(k, v) for k, v in oneRow.items()]
    keyValString = ",".join(keyValList)
    rowString = '\n {' + keyValString + '\n }'

    return rowString


def rowToDict(rowData):
    rowDict = formatRow(rowData[rowData.notnull()].to_dict())
    return rowDict


def exportPrettyJson(df, exportFolder, fileName):
    jsonExportFileName = os.path.join(exportFolder, fileName + ".json")
    outfile = open(jsonExportFileName, 'w')
    rowList = df.apply(rowToDict, axis=1)
    allRows = ",".join(rowList)
    jsonString = '[' + allRows + '\n]'
    outfile.write(jsonString)
    outfile.close()

    return


def exportExcelFile(exportDirectory, exportFolder, fileName):
    mylen = np.vectorize(len)
    writer = pd.ExcelWriter(os.path.join(exportFolder, fileName + ".xlsx"),
                            engine='xlsxwriter')

    workbook = writer.book
    header_format = workbook.add_format({'bold': True,
                                         'valign': 'center',
                                         'border': False,
                                         'align': 'center'})

    cell_format = workbook.add_format({'align': 'center'})

    csvFiles = sorted(os.listdir(exportDirectory))
    for csvFile in csvFiles:
        dataName = csvFile[:-4]

        tempCsvData = pd.read_csv(
                os.path.join(exportDirectory, dataName + ".csv"),
                low_memory=False)

        # put the date time columns in an excel interpretable format
        for col_heading in list(tempCsvData):
            if "time" in col_heading.lower()[-4:]:
                tempCsvData[col_heading] = \
                    pd.to_datetime(tempCsvData[col_heading])

        tempCsvData.to_excel(writer, dataName, startrow=1, header=False,
                             index=False, freeze_panes=(1, 0))

        worksheet = writer.sheets[dataName]
        workbook.add_format({'align': 'center'})

        # Write the column headers with the defined format
        for col_num, value in enumerate(tempCsvData.columns.values):
            worksheet.write(0, col_num, value, header_format)
            colWidth = max(len(value), max(mylen(tempCsvData.iloc[:, col_num].astype(str))))
            worksheet.set_column(col_num, col_num, colWidth, cell_format)

    writer.save()

    return


def readXlsxData(xlsxPathAndFileName):
    # load xlsx
    df = pd.read_excel(xlsxPathAndFileName, sheet_name=None, ignore_index=True)
    cdf = pd.concat(df.values(), ignore_index=True)
    cdf = cdf.set_index('rowIndex')

    return cdf


def exportData(df, fileName, fileType, exportDirectory, mergeCalculatorData):
    # create output folder(s)
    if not os.path.exists(exportDirectory):
        os.makedirs(exportDirectory)

    # sort data by time
    df = df.sort_values("time")

    # all of the exports are based off of csvs table, which are needed to
    # merge the bolus and wizard (AKA calculator) data
    csvExportFolder = exportCsvFiles(df, exportDirectory, fileName, mergeCalculatorData)

    if (("csv" in fileType) | ("json" in fileType) | ("all" in fileType)):
        allData = exportSingleCsv(exportDirectory, fileName, csvExportFolder, fileType)

    if (("json" in fileType) | ("all" in fileType)):
        exportPrettyJson(allData, exportDirectory, fileName)

    if (("xlsx" in fileType) | ("all" in fileType)):
        exportExcelFile(csvExportFolder, exportDirectory, fileName)

    if (("csvs" in fileType) | ("all" in fileType)):
        # unhide the csv files
        unhiddenCsvExportFolder = \
            os.path.join(exportDirectory, fileName + "-csvs", "")
        os.rename(csvExportFolder, unhiddenCsvExportFolder)
    else:
        shutil.rmtree(csvExportFolder)

    return


def flattenInpenData(df):

    bolus_df = df[df['type'] == 'bolus']

    is_inpen = bolus_df.origin.str.contains('InPen') == True
    inpen_idx = np.array(bolus_df.index[is_inpen])

    if len(inpen_idx) > 0:
        #Separate inpen and data
        inpen_df = bolus_df.loc[inpen_idx].copy()
        holdout_df = df.drop(inpen_idx)

        #Convert dicts to dataframes
        eval_df = inpen_df.payload.apply(ast.literal_eval)
        dict_df = pd.DataFrame.from_records(eval_df.values, index=eval_df.index)
        inpen_columns = ['Duration Of Insulin Action In Seconds',
                        'Insulin Concentration',
                        'Insulin Name',
                        'HKWasUserEntered']

        inpen_new_names = ['durationOfInsulinActionInSeconds',
                           'insulinConcentration',
                           'insulinName',
                           'insulinManuallyEntered']

        dict_columns = list(dict_df)

        keep_columns = []
        new_column_names = []

        for col_ix in range(len(inpen_columns)):
            col_name = inpen_columns[col_ix]
            new_name = inpen_new_names[col_ix]
            if col_name in dict_columns:
                keep_columns.append(col_name)
                new_column_names.append(new_name)

        dict_df = dict_df[keep_columns].copy()
        dict_df.columns = new_column_names

        dict_df['insulinManuallyEntered'] = dict_df['insulinManuallyEntered'].astype(bool)

        inpen_df[list(dict_df)] = dict_df
        df = pd.concat([holdout_df, inpen_df], sort=True)

    return df


def cleanHealthkitColumns(df):
    """Clean healthkit data columns"""

    # combine nutrition.carbohydrate(s) data into single carbohydrate field
    if ('nutrition.carbohydrate.net' not in list(df)) & ('nutrition.carbohydrates.net' in list(df)):
        df['nutrition.carbohydrate.net'] = df['nutrition.carbohydrates.net']
        df.drop(columns=['nutrition.carbohydrates.net'], inplace=True)

    elif ('nutrition.carbohydrate.net' in list(df)) & ('nutrition.carbohydrates.net' in list(df)):
        merge_columns = df['nutrition.carbohydrate.net'].isnull() & \
                        df['nutrition.carbohydrates.net'].notnull()

        df.loc[merge_columns, 'nutrition.carbohydrate.net'] = \
            df.loc[merge_columns, 'nutrition.carbohydrates.net']

        df.drop(columns=['nutrition.carbohydrates.net'], inplace=True)

    if ('nutrition.carbohydrate.units' not in list(df)) & ('nutrition.carbohydrates.units' in list(df)):
        df['nutrition.carbohydrate.units'] = df['nutrition.carbohydrates.units']
        df.drop(columns=['nutrition.carbohydrates.units'], inplace=True)

    elif ('nutrition.carbohydrate.units' in list(df)) & ('nutrition.carbohydrates.units' in list(df)):
        merge_columns = df['nutrition.carbohydrate.units'].isnull() & \
                        df['nutrition.carbohydrates.units'].notnull()

        df.loc[merge_columns, 'nutrition.carbohydrate.units'] = \
            df.loc[merge_columns, 'nutrition.carbohydrates.units']

        df.drop(columns=['nutrition.carbohydrates.units'], inplace=True)

    return df


def correctForOverlappingUploads(day_df):

    day_uploadIds = day_df.uploadId.unique()

    upId_starts = []
    upId_ends = []

    for upId in day_uploadIds:
        upId_starts.append(day_df[day_df['uploadId'] == upId].utcTime.min())
        upId_ends.append(day_df[day_df['uploadId'] == upId].utcTime.max())

    latest_start = max(upId_starts)
    earliest_end = min(upId_ends)

    if((earliest_end - latest_start).total_seconds() > 0):
        # If the upload times overlap,
        # return only the most recent upload
        day_df[day_df.uploadTime == day_df.uploadTime.max()]

    return day_df


def deduplicateDaysByUploadId(df, typeField):
    """If two or more uploadids both contain the same type of data in the same
    day, keep only the most recent upload

    (Only applied to uploadids with more than 1 day of data)
    """

    type_df = df[df['type'] == typeField]
    start_size = len(type_df)
    the_rest_df = df[~(df['type'] == typeField)]

    uploadId_sizes = type_df.groupby('uploadId').apply(
            lambda x: (x['utcTime'].max() - x['utcTime'].min()).days)
    uploadIds_gt1day = pd.Series(uploadId_sizes[uploadId_sizes > 1].index)

    all_days = type_df[type_df.uploadId.isin(uploadIds_gt1day)].date.unique()

    df_to_deduplicate = type_df[type_df['date'].isin(all_days)]
    df_to_hold = type_df[~type_df['date'].isin(all_days)]

    df_to_deduplicate = df_to_deduplicate.groupby('date').apply(
            lambda x: correctForOverlappingUploads(x))

    type_df = pd.concat([df_to_deduplicate, df_to_hold], sort=False)
    end_size = len(type_df)
    nDuplicatesRemoved = start_size - end_size
    df = pd.concat([type_df, the_rest_df], sort=False)

    return df, nDuplicatesRemoved


def create_activityName_field(df):
    """Gets the healthkit activity name for all physicalActivity data types"""

    pa_df = df[df['type'] == 'physicalActivity']
    df['activityName'] = pa_df.name.str.split(' - ', expand=True)[0]

    return df


def anonymize_data(data,
                   userID,
                   start_date = "2010-01-01",
                   end_date = dt.datetime.now().strftime("%Y-%m-%d"),
                   salt = os.environ["BIGDATA_SALT"],
                   dataFieldExportList = "dataFieldExportList.csv",
                   mergeWizardDataWithBolusData = True,
                   anonymize = True,
                   keepUploadsAndSettings = True,
                   exportFormat = ['all']
                   ):

    # LOAD DATA
    #startTime = time.time()
    #print("loading data...", end="")
    # check input file and load data. File must be bigger than 2 bytes,
    # and in either json, xlsx, or csv format
    #data, userID = checkInputFile(args.inputFilePathAndName)
    #print("done, took", round(time.time() - startTime, 1), "seconds")
    anonymized_metadata = pd.DataFrame(index=[userID])

    # Flatten Inpen Data
    if ('origin' in list(data)):
        if(data['origin'].str.contains('inpen').sum() > 0):
            data = flattenInpenData(data)

    # Get activityName from 'name' field
    if ('physicalActivity' in list(data.type.unique())) & ('name' in list(data)):
        data = create_activityName_field(data)

    # FILTER DATA
    startTime = time.time()
    #print("filtering data...", end="")
    # check export/approved data field list
    outputFields, anonymizeFields = checkDataFieldList(dataFieldExportList)

    # remove data between start and end dates
    if keepUploadsAndSettings:
        data = filterByDatesExceptUploadsAndSettings(data,
                                                     start_date,
                                                     end_date)
    else:
        data = filterByDates(data, start_date, end_date)

    #print("done, took", round(time.time() - startTime, 1), "seconds")


    # CLEAN DATA
    startTime = time.time()
    #print("cleaning data...", end="")
    data['date'] = data['time'].str.split('T', expand=True)[0]
    data["utcTime"] = pd.to_datetime(data["time"], utc=True)
    data["utcTime"] = data["utcTime"].dt.tz_localize(None)

    data = add_uploadDateTime(data)

   # for upload_type in data.type.unique():
   #     metadata_name = 'nDuplicatesRemovedByUploadId_' + upload_type
   #     data, anonymized_metadata[metadata_name] = \
   #         deduplicateDaysByUploadId(data, upload_type)

    # remove negative durations
    data, anonymized_metadata['nNegativeDurationsRemoved'] = \
        removeNegativeDurations(data)

    # Tslim calibration bug fix
    data, anonymized_metadata['numberOfTandemAndPayloadCalReadings'] = \
         tslim_calibration_fix(data)

    # get rid of cgm values too low/high (< 38 & > 402 mg/dL)
    data, anonymized_metadata['numberOfInvalidCgmValuesRemoved'] = \
        removeInvalidCgmValues(data)

    # Separate and Drop CGM
    cgm_df = data[data['type']=='cbg'].copy()
    data = data[~(data['type']=='cbg')].copy()

    cgm_df, anonymized_metadata['numberOfInvalidCgmValues'] = \
        removeInvalidCgmValues(cgm_df)
    cgm_df, anonymized_metadata['nDuplicatesRemovedDeviceTime'] = \
        removeCgmDuplicates(cgm_df, "deviceTime")
    cgm_df, anonymized_metadata['nDuplicatesRemovedUtcTime'] = \
        removeCgmDuplicates(cgm_df, "time")
    #cgm_df, anonymized_metadata['nDuplicatesRemovedEstLocalTime'] = \
    #    removeCgmDuplicates(cgm_df, "est.localTime")

    # round time to the nearest 5 minutes
    cgm_df = round_time(
        cgm_df,
        timeIntervalMinutes=5,
        timeField="time",
        roundedTimeFieldName="roundedTime",
        verbose=False
    )

    # get rid of duplicates that have the same "roundedTime" and value
    cgm_df, anonymized_metadata['nDuplicatesRemovedRoundedTimeAndValue'] = \
        removeCgmDuplicates(cgm_df, "roundedTime")

    # remove all other cgm roundedTimes, keeping only the most recent upload's values
    cgm_df, anonymized_metadata['nDuplicatesRemovedRoundedTimeOnly'] = \
        removeCgmRoundedTimeDuplicates(cgm_df)

    # round local time to the nearest 5 minutes
    #cgm_df = round_time(
    #    cgm_df,
    #    timeIntervalMinutes=5,
    #    timeField="est.localTime",
    #    roundedTimeFieldName="roundedEstLocalTime",
    #    verbose=False
    #)

    # get rid of duplicates that have the same "roundedTime"
    #cgm_df, anonymized_metadata['nDuplicatesRemovedRoundedEstLocalTime'] = \
    #    removeCgmDuplicates(cgm_df, "roundedEstLocalTime")


    # Add cleaned CGM data back into dataframe
    data = pd.concat([data, cgm_df], sort=False)
    data.drop(columns=['roundedTime', 'uploadTime'], inplace=True)

    # Reset index after all deduplication
    data.reset_index(drop=True, inplace=True)

    #print("done, took", round(time.time() - startTime, 1), "seconds")

    # ANONYMIZE DATA

    if anonymize:
        startTime = time.time()
        #print("anonymzing data...", end="")
        # remove manufacturer from annotations.code
        data = removeManufacturersFromAnnotationsCode(data)
        data = createBolusSchedules(data)
        # hash the required data fields
        data = anonymizeData(data, anonymizeFields, salt, userID)
        hashID = hashUserId(userID, salt)
        #print("done, took", round(time.time() - startTime, 1), "seconds")
    else:
        pass
        #print("skipping anonymization")

    # Use a whitelist of fields you want to flatten
    columns_to_flatten = ["bgTarget",
                          "activityDuration",
                          "change",
                          "distance",
                          "dose",
                          "duration",
                          "energy",
                          "formulation",
                          "lowAlerts",
                          "highAlerts",
                          "nutrition",
                          "outOfRangeAlerts",
                          "rateOfChangeAlerts",
                          "reason",
                          "recommended",
                          "units"]

    # flatten a specific list of embedded json, if it exists
    data = flattenJson(data, columns_to_flatten)

    columns_before_filter = list(data)

    if mergeWizardDataWithBolusData:
        data = mergeWizardWithBolusInMemory(data)

    # only keep the data fields that are approved
    data = filterByApprovedDataFields(data, outputFields)

    data = cleanHealthkitColumns(data)

    # Final alphabetical column sort
    data.sort_index(axis=1, inplace=True)

    # Sort by time ascending and reset index
    data.sort_values(by='time', ascending=True, inplace=True)
    data.reset_index(inplace=True, drop=True)

    columns_after_filter = list(data)

    removed_columns = list(set(columns_before_filter) -
                           set(columns_after_filter)
                           )

    anonymized_metadata['hashID'] = hashID
    anonymized_metadata.reset_index(inplace=True, drop=False)

    return data, hashID, anonymized_metadata, removed_columns, columns_after_filter
    