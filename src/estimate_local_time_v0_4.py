#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
description: Estimate local time
version: 0.0.4
created: 2018-04-30
last_modified: 2019-10
author: Ed Nykaza
dependencies:
    * wikipedia-timezone-aliases-2018-04-28.csv
license: BSD-2-Clause
'''


# %% REQUIRED LIBRARIES
import os
import sys
import pytz
import numpy as np
import pandas as pd
import datetime as dt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# %% CONSTANTS
MGDL_PER_MMOLL = 18.01559
ESTIMATE_LOCAL_TIME_VERSION = "0.0.4"


# %% FUNCTIONS
'''
the functions that are called in this script,
which includes notes of where the functions came from,
and whether they were refactored
'''


def readXlsxData(xlsxPathAndFileName):
    # load xlsx
    df = pd.read_excel(xlsxPathAndFileName, sheet_name=None, ignore_index=True)
    cdf = pd.concat(df.values(), ignore_index=True)
    cdf = cdf.set_index('jsonRowIndex')

    return cdf


def check_and_load_input_file(inputFile):
    if os.path.isfile(inputFile):
        if os.stat(inputFile).st_size > 1000:
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

    return inputData, fileName


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


def get_embedded_field(ts, embedded_field):
    '''get a field that is nested in more than 1 embedded dictionary (json)

    Args:
        ts: a pandas time series of the field that has embedded json
        embedded_field (str): the location of the field that is deeply nested
            (e.g., "origin.payload.device.model")

    Raise:
        ValueError: if you don't pass in a pandas time series

    Returns:
        new_ts: a new time series of the key of interest

    NOTE:
        this is new function
        the "." notation is used to reference nested json

    '''
    field_list = embedded_field.split(".")
    if len(field_list) < 2:
        raise ValueError('Expecting at least 1 embedded field')

    new_ts = expand_embedded_dict(ts, field_list[1])
    for i in range(2, len(field_list)):
        new_ts = expand_embedded_dict(new_ts, field_list[i])

    return new_ts


def add_upload_info_to_cgm_records(groups, df):
    upload_locations = [
        "upload.uploadId",
        "upload.deviceManufacturers",
        "upload.deviceModel",
        "upload.deviceSerialNumber",
        "upload.deviceTags"
    ]

    if "upload" in groups["type"].unique():
        upload = groups.get_group("upload").dropna(
            axis=1, how="all"
        ).add_prefix("upload.")

        df = pd.merge(
            left=df,
            right=upload[list(set(upload_locations) & set(list(upload)))],
            left_on="uploadId",
            right_on="upload.uploadId",
            how="left"
        )

    return df


def expand_heathkit_cgm_fields(df):
    # TODO: refactor the code/function that originally grabs
    # these fields, so we are only doing it once, and so
    # we don't have to drop the columns for the code below to work.
    drop_columns = [
        'origin.payload.device.name',
        'origin.payload.device.manufacturer',
        'origin.payload.sourceRevision.source.name'
    ]
    for drop_col in drop_columns:
        if drop_col in list(df):
            df.drop(columns=[drop_col], inplace=True)

    healthkit_locations = [
        "origin",
        "origin.payload",
        "origin.payload.device",
        "origin.payload.sourceRevision",
        "origin.payload.sourceRevision.source",
        "payload",
    ]

    for hk_loc in healthkit_locations:
        if hk_loc in list(df):
            temp_df = (
                expand_entire_dict(df[hk_loc].copy()).add_prefix(hk_loc + ".")
            )
            df = pd.concat([df, temp_df], axis=1)

    return df


def get_healthkit_timezone(df):
    '''
    TODO: refactor to account for more efficient way to get embedded json
    '''
    if "payload" in list(df):
        df["payload.HKTimeZone"] = (
            expand_embedded_dict(df["payload"], "HKTimeZone")
        )
        if "timezone" not in list(df):
            if "payload.HKTimeZone" in list(df):
                hk_tz_idx = df["payload.HKTimeZone"].notnull()
                df.loc[hk_tz_idx, "deviceType"] = "healthkit"
                df.rename(
                    columns={"payload.HKTimeZone": "timezone"},
                    inplace=True
                )

            else:
                df["timezone"] = np.nan
                df["deviceType"] = np.nan
        else:
            if "payload.HKTimeZone" in list(df):
                hk_tz_idx = df["payload.HKTimeZone"].notnull()
                df.loc[hk_tz_idx, "timezone"] = (
                    df.loc[hk_tz_idx, "payload.HKTimeZone"]
                )
                df.loc[hk_tz_idx, "deviceType"] = "healthkit"
            else:
                df["timezone"] = np.nan
                df["deviceType"] = np.nan

    else:
        df["timezone"] = np.nan
        df["deviceType"] = np.nan

    return df[["timezone", "deviceType"]]


def make_tz_unaware(date_time):
    return date_time.replace(tzinfo=None)


def to_utc_datetime(df):
    '''
    this is new to deal with perfomance issue with the previous method
    of converting to string to datetime with pd.to_datetime()
    '''
    utc_time_tz_aware = pd.to_datetime(
        df["time"],
        format="%Y-%m-%dT%H:%M:%S",
        utc=True
    )
    utc_tz_unaware = utc_time_tz_aware.apply(make_tz_unaware)

    return utc_tz_unaware


# apply the large timezone offset correction (AKA Darin's fix)
def timezone_offset_bug_fix(df):
    '''
    this is taken from estimate-local-time.py
    TODO: add in unit testing where there is no TZP that is > 840 or < -720
    '''

    if "timezoneOffset" in list(df):

        while ((df.timezoneOffset > 840).sum() > 0):
            df.loc[df.timezoneOffset > 840, ["conversionOffset"]] = (
                df.loc[df.timezoneOffset > 840, ["conversionOffset"]]
                - (1440 * 60 * 1000)
                )

            df.loc[df.timezoneOffset > 840, ["timezoneOffset"]] = (
                df.loc[df.timezoneOffset > 840, ["timezoneOffset"]] - 1440
            )

        while ((df.timezoneOffset < -720).sum() > 0):
            df.loc[df.timezoneOffset < -720, ["conversionOffset"]] = (
                df.loc[df.timezoneOffset < -720, ["conversionOffset"]]
                + (1440 * 60 * 1000)
            )

            df.loc[df.timezoneOffset < -720, ["timezoneOffset"]] = (
                df.loc[df.timezoneOffset < -720, ["timezoneOffset"]] + 1440
            )

    return df


def convert_deprecated_timezone_to_alias(df, tzAlias):
    if "timezone" in df:
        uniqueTimezones = df.timezone.unique()
        uniqueTimezones = uniqueTimezones[pd.notnull(df.timezone.unique())]

        for uniqueTimezone in uniqueTimezones:
            alias = tzAlias.loc[tzAlias.tz.str.endswith(uniqueTimezone),
                                ["alias"]].values
            if len(alias) == 1:
                df.loc[df.timezone == uniqueTimezone, ["timezone"]] = alias

    return df


def create_contiguous_day_series(df):
    first_day = df["date"].min()
    last_day = df["date"].max()
    rng = pd.date_range(first_day, last_day).date
    contiguousDaySeries = \
        pd.DataFrame(rng, columns=["date"]).sort_values(
                "date", ascending=False).reset_index(drop=True)

    return contiguousDaySeries


def add_device_type(df):
    col_headings = list(df)
    if "deviceType" not in col_headings:
        df["deviceType"] = np.nan
    if "deviceTags" in col_headings:
        # first make sure deviceTag is in string format
        df["deviceTags"] = df.deviceTags.astype(str)
        # filter by type not null device tags
        ud = df[df["deviceTags"].notnull()].copy()
        # define a device type (e.g., pump, cgm, or healthkit)
        ud.loc[
            ((ud["deviceTags"].str.contains("pump"))
             & (ud["deviceType"].isnull())),
            ["deviceType"]
        ] = "pump"

        # define a device type (e.g., cgm)
        ud.loc[
            ((ud["deviceTags"].str.contains("cgm"))
             & (ud["deviceType"].isnull())),
            ["deviceType"]
        ] = "cgm"

        return ud["deviceType"]
    else:
        return np.nan


def get_timezone_offset(currentDate, currentTimezone):
    try:
        tz = pytz.timezone(currentTimezone)
        # add 1 day to the current date to account for changes to/from DST
        tzoNum = int(
            tz.localize(currentDate + dt.timedelta(days=1)).strftime("%z")
        )
        tzoHours = np.floor(tzoNum / 100)
        tzoMinutes = round((tzoNum / 100 - tzoHours) * 100, 0)
        tzoSign = np.sign(tzoHours)
        tzo = int((tzoHours * 60) + (tzoMinutes * tzoSign))

    except Exception as e:
        # Return an empty timezone if the currentTimezone does not exist
        # or throws an error
        if 'GMT' in currentTimezone:
            # edge case in format of GMT-04:00 or GMT+01:00
            if ":" in currentTimezone:
                tzo = (
                    float(currentTimezone.split("T")[1].split(":")[0]) * 60
                    + float(currentTimezone.split("T")[1].split(":")[1])
                )
            # this is the case where GMT-0400
            else:
                tzo = (
                    float(
                            currentTimezone.split("T")[1].split(":")[0][0:3]
                    ) * 60
                    + float(currentTimezone.split("T")[1].split(":")[0][3:])
                )
        else:
            print(e, "error with timezone = ", currentTimezone)
            tzo = np.nan

    return tzo


def add_device_day_series(df, dfContDays, deviceTypeName):
    if len(df) > 0:
        dfDayGroups = df.groupby("date")
        if "timezoneOffset" in df:
            dfDaySeries = pd.DataFrame(dfDayGroups["timezoneOffset"].median())
        else:
            dfDaySeries = pd.DataFrame(columns=["timezoneOffset"])
            dfDaySeries.index.name = "date"

        if "upload" in deviceTypeName:
            if (("timezone" in df) & (df["timezone"].notnull().sum() > 0)):
                dfDaySeries["timezone"] = (
                    dfDayGroups.timezone.describe()["top"]
                )
                # get the timezone offset for the timezone
                for i in dfDaySeries.index:
                    if pd.notnull(dfDaySeries.loc[i, "timezone"]):
                        tzo = get_timezone_offset(
                                pd.to_datetime(i),
                                dfDaySeries.loc[i, "timezone"])
                        dfDaySeries.loc[i, ["timezoneOffset"]] = tzo
                if "timeProcessing" in dfDaySeries:
                    dfDaySeries["timeProcessing"] = \
                        dfDayGroups.timeProcessing.describe()["top"]
                else:
                    dfDaySeries["timeProcessing"] = np.nan

        dfDaySeries = dfDaySeries.add_prefix(deviceTypeName + "."). \
            rename(columns={deviceTypeName + ".date": "date"})

        dfContDays = pd.merge(dfContDays, dfDaySeries.reset_index(),
                              on="date", how="left")

    else:
        dfContDays[deviceTypeName + ".timezoneOffset"] = np.nan

    return dfContDays


def impute_upload_records(df, contDays, deviceTypeName):
    daySeries = \
        add_device_day_series(df, contDays, deviceTypeName)

    if ((len(df) > 0) & (deviceTypeName + ".timezone" in daySeries)):
        for i in daySeries.index[1:]:
            if pd.isnull(daySeries[deviceTypeName + ".timezone"][i]):
                daySeries.loc[i, [deviceTypeName + ".timezone"]] = (
                    daySeries.loc[i-1, deviceTypeName + ".timezone"]
                )
            if pd.notnull(daySeries[deviceTypeName + ".timezone"][i]):
                tz = daySeries.loc[i, deviceTypeName + ".timezone"]
                tzo = get_timezone_offset(
                    pd.to_datetime(daySeries.loc[i, "date"]),
                    tz
                )
                daySeries.loc[i, deviceTypeName + ".timezoneOffset"] = tzo

            if pd.notnull(daySeries[deviceTypeName + ".timeProcessing"][i-1]):
                daySeries.loc[i, deviceTypeName + ".timeProcessing"] = \
                    daySeries.loc[i-1, deviceTypeName + ".timeProcessing"]

    else:
        daySeries[deviceTypeName + ".timezone"] = np.nan
        daySeries[deviceTypeName + ".timeProcessing"] = np.nan

    return daySeries


def add_home_timezone(df, contDays):

    if (("timezone" in df) & (df["timezone"].notnull().sum() > 0)):
        homeTimezone = df["timezone"].describe()["top"]
        tzo = contDays.date.apply(
                lambda x: get_timezone_offset(pd.to_datetime(x), homeTimezone))

        contDays["home.imputed.timezoneOffset"] = tzo
        contDays["home.imputed.timezone"] = homeTimezone

    else:
        contDays["home.imputed.timezoneOffset"] = np.nan
        contDays["home.imputed.timezone"] = np.nan
    contDays["home.imputed.timeProcessing"] = np.nan

    return contDays


def estimateTzAndTzoWithUploadRecords(cDF):

    cDF["est.type"] = np.nan
    cDF["est.gapSize"] = np.nan
    cDF["est.timezoneOffset"] = cDF["upload.timezoneOffset"]
    cDF["est.annotations"] = np.nan

    if "upload.timezone" in cDF:
        cDF.loc[cDF["upload.timezone"].notnull(), ["est.type"]] = "UPLOAD"
        cDF["est.timezone"] = cDF["upload.timezone"]
        cDF["est.timeProcessing"] = cDF["upload.timeProcessing"]
    else:
        cDF["est.timezone"] = np.nan
        cDF["est.timeProcessing"] = np.nan

    cDF.loc[((cDF["est.timezoneOffset"] !=
              cDF["home.imputed.timezoneOffset"]) &
            (pd.notnull(cDF["est.timezoneOffset"]))),
            "est.annotations"] = "travel"

    return cDF


def assignTzoFromImputedSeries(df, i, imputedSeries):
    df.loc[i, ["est.type"]] = "DEVICE"

    df.loc[i, ["est.timezoneOffset"]] = \
        df.loc[i, imputedSeries + ".timezoneOffset"]

    df.loc[i, ["est.timezone"]] = \
        df.loc[i, imputedSeries + ".timezone"]

    df.loc[i, ["est.timeProcessing"]] = \
        df.loc[i, imputedSeries + ".timeProcessing"]

    return df


def compareDeviceTzoToImputedSeries(df, sIdx, device):
    for i in sIdx:
        # if the device tzo = imputed tzo, then chose the imputed tz and tzo
        # note, dst is accounted for in the imputed tzo
        for imputedSeries in ["pump.upload.imputed", "cgm.upload.imputed",
                              "healthkit.upload.imputed", "home.imputed"]:
            # if the estimate has not already been made
            if pd.isnull(df.loc[i, "est.timezone"]):

                if df.loc[i, device + ".timezoneOffset"] == \
                  df.loc[i, imputedSeries + ".timezoneOffset"]:

                    assignTzoFromImputedSeries(df, i, imputedSeries)

                    df = addAnnotation(df, i,
                                       "tz-inferred-from-" + imputedSeries)

                # if the imputed series has a timezone estimate, then see if
                # the current day is a dst change day
                elif (pd.notnull(df.loc[i, imputedSeries + ".timezone"])):
                    imputedTimezone = df.loc[i, imputedSeries + ".timezone"]
                    if isDSTChangeDay(df.loc[i, "date"], imputedTimezone):

                        dstRange = getRangeOfTZOsForTimezone(imputedTimezone)
                        if ((df.loc[i, device + ".timezoneOffset"] in dstRange)
                          & (df.loc[i, imputedSeries + ".timezoneOffset"] in dstRange)):

                            assignTzoFromImputedSeries(df, i, imputedSeries)

                            df = addAnnotation(df, i, "dst-change-day")
                            df = addAnnotation(
                                    df, i, "tz-inferred-from-" + imputedSeries)

    return df


def estimateTzAndTzoWithDeviceRecords(cDF):

    # 2A. use the TZO of the pump or cgm device if it exists on a given day. In
    # addition, compare the TZO to one of the imputed day series (i.e., the
    # upload and home series to see if the TZ can be inferred)

    for deviceType in ["pump", "cgm"]:
        # find the indices of days where a TZO estimate has not been made
        # AND where the device (e.g., pump or cgm) TZO has data
        sIndices = cDF[(
                (cDF["est.timezoneOffset"].isnull())
                & (cDF[deviceType + ".timezoneOffset"].notnull())
        )].index

        # compare the device TZO to the imputed series to infer time zone
        cDF = compareDeviceTzoToImputedSeries(cDF, sIndices, deviceType)

    # 2B. if the TZ cannot be inferred with 2A, then see if the TZ can be
    # inferred from the previous day's TZO. If the device TZO is equal to the
    # previous day's TZO, AND if the previous day has a TZ estimate, use the
    # previous day's TZ estimate for the current day's TZ estimate
    for deviceType in ["pump", "cgm"]:
        sIndices = cDF[(
            (cDF["est.timezoneOffset"].isnull())
            & (cDF[deviceType + ".timezoneOffset"].notnull())
        )].index

        cDF = compareDeviceTzoToPrevDayTzo(cDF, sIndices, deviceType)

    # 2C. after 2A and 2B, check the DEVICE estimates to make sure that the
    # pump and cgm tzo do not differ by more than 60 minutes. If they differ
    # by more that 60 minutes, then mark the estimate as UNCERTAIN. Also, we
    # allow the estimates to be off by 60 minutes as there are a lot of cases
    # where the devices are off because the user changes the time for DST,
    # at different times
    sIndices = cDF[((cDF["est.type"] == "DEVICE") &
                    (cDF["pump.timezoneOffset"].notnull()) &
                    (cDF["cgm.timezoneOffset"].notnull()) &
                    (cDF["pump.timezoneOffset"] != cDF["cgm.timezoneOffset"])
                    )].index

    tzoDiffGT60 = abs(cDF.loc[sIndices, "cgm.timezoneOffset"] -
                      cDF.loc[sIndices, "pump.timezoneOffset"]) > 60

    idx = tzoDiffGT60.index[tzoDiffGT60]

    cDF.loc[idx, ["est.type"]] = "UNCERTAIN"
    for i in idx:
        cDF = addAnnotation(cDF, i, "pump-cgm-tzo-mismatch")

    return cDF


def imputeTzAndTzo(cDF):

    sIndices = cDF[cDF["est.timezoneOffset"].isnull()].index
    hasTzoIndices = cDF[cDF["est.timezoneOffset"].notnull()].index
    if len(hasTzoIndices) > 0:
        if len(sIndices) > 0:
            lastDay = max(sIndices)

            while ((sIndices.min() < max(hasTzoIndices)) &
                   (len(sIndices) > 0)):

                currentDay, prevDayWithDay, nextDayIdx = \
                    getImputIndices(cDF, sIndices, hasTzoIndices)

                cDF = imputeByTimezone(cDF, currentDay,
                                       prevDayWithDay, nextDayIdx)

                sIndices = cDF[(
                    (cDF["est.timezoneOffset"].isnull())
                    & (~cDF["est.annotations"].str.contains("unable-to-impute-tzo").fillna(False))
                )].index

                hasTzoIndices = cDF[cDF["est.timezoneOffset"].notnull()].index

            # try to impute to the last day (earliest day) in the dataset
            # if the last record has a timezone that is the home record, then
            # impute using the home timezone
            if len(sIndices) > 0:
                currentDay = min(sIndices)
                prevDayWithDay = currentDay - 1
                gapSize = lastDay - currentDay

                for i in range(currentDay, lastDay + 1):
                    if cDF.loc[prevDayWithDay, "est.timezoneOffset"] == \
                      cDF.loc[prevDayWithDay, "home.imputed.timezoneOffset"]:

                        cDF.loc[i, ["est.type"]] = "IMPUTE"

                        cDF.loc[i, ["est.timezoneOffset"]] = \
                            cDF.loc[i, "home.imputed.timezoneOffset"]

                        cDF.loc[i, ["est.timezone"]] = \
                            cDF.loc[i, "home.imputed.timezone"]

                        cDF = addAnnotation(cDF, i, "gap=" + str(gapSize))
                        cDF.loc[i, ["est.gapSize"]] = gapSize

                    else:
                        cDF.loc[i, ["est.type"]] = "UNCERTAIN"
                        cDF = addAnnotation(cDF, i, "unable-to-impute-tzo")
    else:
        cDF["est.type"] = "UNCERTAIN"
        cDF["est.annotations"] = "unable-to-impute-tzo"

    return cDF


def getRangeOfTZOsForTimezone(tz):
    minMaxTzo = [get_timezone_offset(pd.to_datetime("1/1/2017"), tz),
                 get_timezone_offset(pd.to_datetime("5/1/2017"), tz)]

    rangeOfTzo = np.arange(int(min(minMaxTzo)), int(max(minMaxTzo))+1, 15)

    return rangeOfTzo


def getListOfDSTChangeDays(cDF):

    # get a list of DST change days for the home time zone
    dstChangeDays = \
        cDF[abs(cDF["home.imputed.timezoneOffset"] -
                cDF["home.imputed.timezoneOffset"].shift(-1)) > 0].date

    return dstChangeDays


def correctEstimatesAroundDst(df, cDF):

    # get a list of DST change days for the home time zone
    dstChangeDays = getListOfDSTChangeDays(cDF)

    # loop through the df within 2 days of a daylight savings time change
    for d in dstChangeDays:
        dstIndex = df[(df.date > (d + dt.timedelta(days=-2))) &
                      (df.date < (d + dt.timedelta(days=2)))].index
        for dIdx in dstIndex:
            if pd.notnull(df.loc[dIdx, "est.timezone"]):
                tz = pytz.timezone(df.loc[dIdx, "est.timezone"])
                tzRange = getRangeOfTZOsForTimezone(str(tz))
                minHoursToLocal = min(tzRange)/60
                tzoNum = int(
                    tz.localize(
                        df.loc[dIdx, "utcTime"]
                        + dt.timedelta(hours=minHoursToLocal)
                    ).strftime("%z")
                )

                tzoHours = np.floor(tzoNum / 100)
                tzoMinutes = round((tzoNum / 100 - tzoHours) * 100, 0)
                tzoSign = np.sign(tzoHours)
                tzo = int((tzoHours * 60) + (tzoMinutes * tzoSign))
                localTime = \
                    df.loc[dIdx, "utcTime"] + pd.to_timedelta(tzo, unit="m")
                df.loc[dIdx, ["est.localTime"]] = localTime
                df.loc[dIdx, ["est.timezoneOffset"]] = tzo
    return df


def applyLocalTimeEstimates(df, cDF):
    df = pd.merge(df, cDF, how="left", on="date")
    df["est.localTime"] = \
        df["utcTime"] + pd.to_timedelta(df["est.timezoneOffset"], unit="m")

    df = correctEstimatesAroundDst(df, cDF)

    return df["est.localTime"].values


def isDSTChangeDay(currentDate, currentTimezone):
    tzoCurrentDay = get_timezone_offset(
        pd.to_datetime(currentDate),
        currentTimezone
    )
    tzoPreviousDay = get_timezone_offset(
        pd.to_datetime(currentDate) + dt.timedelta(days=-1),
        currentTimezone
    )

    return (tzoCurrentDay != tzoPreviousDay)


def tzoRangeWithComparisonTz(df, i, comparisonTz):
    # if we have a previous timezone estimate, then calcuate the range of
    # timezone offset values for that time zone
    if pd.notnull(comparisonTz):
        rangeTzos = getRangeOfTZOsForTimezone(comparisonTz)
    else:
        comparisonTz = np.nan
        rangeTzos = np.array([])

    return rangeTzos


def tzAndTzoRangePreviousDay(df, i):
    # if we have a previous timezone estimate, then calcuate the range of
    # timezone offset values for that time zone
    comparisonTz = df.loc[i-1, "est.timezone"]

    rangeTzos = tzoRangeWithComparisonTz(df, i, comparisonTz)

    return comparisonTz, rangeTzos


def assignTzoFromPreviousDay(df, i, previousDayTz):

    df.loc[i, ["est.type"]] = "DEVICE"
    df.loc[i, ["est.timezone"]] = previousDayTz
    df.loc[i, ["est.timezoneOffset"]] = \
        get_timezone_offset(pd.to_datetime(df.loc[i, "date"]), previousDayTz)

    df.loc[i, ["est.timeProcessing"]] = df.loc[i-1, "est.timeProcessing"]
    df = addAnnotation(df, i, "tz-inferred-from-prev-day")

    return df


def assignTzoFromDeviceTzo(df, i, device):

    df.loc[i, ["est.type"]] = "DEVICE"
    df.loc[i, ["est.timezoneOffset"]] = \
        df.loc[i, device + ".timezoneOffset"]
    df.loc[i, ["est.timeProcessing"]] = \
        df.loc[i, device + ".upload.imputed.timeProcessing"]

    df = addAnnotation(df, i, "likely-travel")
    df = addAnnotation(df, i, "tzo-from-" + device)

    return df


def tzAndTzoRangeWithHomeTz(df, i):
    # if we have a previous timezone estimate, then calcuate the range of
    # timezone offset values for that time zone
    comparisonTz = df.loc[i, "home.imputed.timezone"]

    rangeTzos = tzoRangeWithComparisonTz(df, i, comparisonTz)

    return comparisonTz, rangeTzos


def compareDeviceTzoToPrevDayTzo(df, sIdx, device):

    for i in sIdx[sIdx > 0]:

        # first see if the previous record has a tzo
        if (pd.notnull(df.loc[i-1, "est.timezoneOffset"])):

            previousDayTz, dstRange = tzAndTzoRangePreviousDay(df, i)
            timeDiff = abs((df.loc[i, device + ".timezoneOffset"]) -
                           df.loc[i-1, "est.timezoneOffset"])

            # next see if the previous record has a tz
            if (pd.notnull(df.loc[i-1, "est.timezone"])):

                if timeDiff == 0:
                    assignTzoFromPreviousDay(df, i, previousDayTz)

                # see if the previous day's tzo and device tzo are within the
                # dst range (as that is a common problem with this data)
                elif ((df.loc[i, device + ".timezoneOffset"] in dstRange)
                      & (df.loc[i-1, "est.timezoneOffset"] in dstRange)):

                    # then see if it is DST change day
                    if isDSTChangeDay(df.loc[i, "date"], previousDayTz):

                        df = addAnnotation(df, i, "dst-change-day")
                        assignTzoFromPreviousDay(df, i, previousDayTz)

                    # if it is not DST change day, then mark this as uncertain
                    else:
                        # also, check to see if the difference between device.
                        # tzo and prev.tzo is less than the expected dst
                        # difference. There is a known issue where the BtUTC
                        # procedure puts clock drift into the device.tzo,
                        # and as a result the tzo can be off by 15, 30,
                        # or 45 minutes.
                        if (((df.loc[i, device + ".timezoneOffset"] ==
                              min(dstRange)) |
                            (df.loc[i, device + ".timezoneOffset"] ==
                             max(dstRange))) &
                           ((df.loc[i-1, "est.timezoneOffset"] ==
                             min(dstRange)) |
                            (df.loc[i-1, "est.timezoneOffset"] ==
                             max(dstRange)))):

                            df.loc[i, ["est.type"]] = "UNCERTAIN"
                            df = addAnnotation(df, i,
                                               "likely-dst-error-OR-travel")

                        else:

                            df.loc[i, ["est.type"]] = "UNCERTAIN"
                            df = addAnnotation(df, i,
                                               "likely-15-min-dst-error")

                # next see if time difference between device.tzo and prev.tzo
                # is off by 720 minutes, which is indicative of a common
                # user AM/PM error
                elif timeDiff == 720:
                    df.loc[i, ["est.type"]] = "UNCERTAIN"
                    df = addAnnotation(df, i, "likely-AM-PM-error")

                # if it doesn't fall into any of these cases, then the
                # tzo difference is likely due to travel
                else:
                    df = assignTzoFromDeviceTzo(df, i, device)

            elif timeDiff == 0:
                df = assignTzoFromDeviceTzo(df, i, device)

        # if there is no previous record to compare with check for dst errors,
        # and if there are no errors, it is likely a travel day
        else:

            comparisonTz, dstRange = tzAndTzoRangeWithHomeTz(df, i)
            timeDiff = abs((df.loc[i, device + ".timezoneOffset"]) -
                           df.loc[i, "home.imputed.timezoneOffset"])

            if ((df.loc[i, device + ".timezoneOffset"] in dstRange)
               & (df.loc[i, "home.imputed.timezoneOffset"] in dstRange)):

                # see if it is DST change day
                if isDSTChangeDay(df.loc[i, "date"], comparisonTz):

                    df = addAnnotation(df, i, "dst-change-day")
                    df.loc[i, ["est.type"]] = "DEVICE"
                    df.loc[i, ["est.timezoneOffset"]] = \
                        df.loc[i, device + ".timezoneOffset"]
                    df.loc[i, ["est.timezone"]] = \
                        df.loc[i, "home.imputed.timezone"]
                    df.loc[i, ["est.timeProcessing"]] = \
                        df.loc[i, device + ".upload.imputed.timeProcessing"]

                # if it is not DST change day, then mark this as uncertain
                else:
                    # also, check to see if the difference between device.
                    # tzo and prev.tzo is less than the expected dst
                    # difference. There is a known issue where the BtUTC
                    # procedure puts clock drift into the device.tzo,
                    # and as a result the tzo can be off by 15, 30,
                    # or 45 minutes.
                    if (((df.loc[i, device + ".timezoneOffset"] ==
                          min(dstRange)) |
                        (df.loc[i, device + ".timezoneOffset"] ==
                         max(dstRange))) &
                       ((df.loc[i, "home.imputed.timezoneOffset"] ==
                         min(dstRange)) |
                        (df.loc[i, "home.imputed.timezoneOffset"] ==
                         max(dstRange)))):

                        df.loc[i, ["est.type"]] = "UNCERTAIN"
                        df = addAnnotation(df, i, "likely-dst-error-OR-travel")

                    else:

                        df.loc[i, ["est.type"]] = "UNCERTAIN"
                        df = addAnnotation(df, i, "likely-15-min-dst-error")

            # next see if time difference between device.tzo and prev.tzo
            # is off by 720 minutes, which is indicative of a common
            # user AM/PM error
            elif timeDiff == 720:
                df.loc[i, ["est.type"]] = "UNCERTAIN"
                df = addAnnotation(df, i, "likely-AM-PM-error")

            # if it doesn't fall into any of these cases, then the
            # tzo difference is likely due to travel

            else:
                df = assignTzoFromDeviceTzo(df, i, device)

    return df


def getImputIndices(df, sIdx, hIdx):

    lastDayIdx = len(df) - 1

    currentDayIdx = sIdx.min()
    tempList = pd.Series(hIdx) - currentDayIdx
    prevDayIdx = currentDayIdx - 1
    nextDayIdx = \
        min(currentDayIdx + min(tempList[tempList >= 0]), lastDayIdx)

    return currentDayIdx, prevDayIdx, nextDayIdx


def imputeByTimezone(df, currentDay, prevDaywData, nextDaywData):

    gapSize = (nextDaywData - currentDay)

    if prevDaywData >= 0:

        if df.loc[prevDaywData, "est.timezone"] == \
          df.loc[nextDaywData, "est.timezone"]:

            tz = df.loc[prevDaywData, "est.timezone"]

            for i in range(currentDay, nextDaywData):

                df.loc[i, ["est.timezone"]] = tz

                df.loc[i, ["est.timezoneOffset"]] = \
                    get_timezone_offset(pd.to_datetime(df.loc[i, "date"]), tz)

                df.loc[i, ["est.type"]] = "IMPUTE"

                df = addAnnotation(df, i, "gap=" + str(gapSize))
                df.loc[i, ["est.gapSize"]] = gapSize

        # TODO: this logic should be updated to handle the edge case
        # where the day before and after the gap have differing TZ, but
        # the same TZO. In that case the gap should be marked as UNCERTAIN
        elif (
                df.loc[prevDaywData, "est.timezoneOffset"] ==
                df.loc[nextDaywData, "est.timezoneOffset"]
        ):

            for i in range(currentDay, nextDaywData):

                df.loc[i, ["est.timezoneOffset"]] = \
                    df.loc[prevDaywData, "est.timezoneOffset"]

                df.loc[i, ["est.type"]] = "IMPUTE"

                df = addAnnotation(df, i, "gap=" + str(gapSize))
                df.loc[i, ["est.gapSize"]] = gapSize

        else:
            for i in range(currentDay, nextDaywData):
                df.loc[i, ["est.type"]] = "UNCERTAIN"
                df = addAnnotation(df, i, "unable-to-impute-tzo")

    else:
        for i in range(currentDay, nextDaywData):
            df.loc[i, ["est.type"]] = "UNCERTAIN"
            df = addAnnotation(df, i, "unable-to-impute-tzo")

    return df


def addAnnotation(df, idx, annotationMessage):
    if pd.notnull(df.loc[idx, "est.annotations"]):
        df.loc[idx, ["est.annotations"]] = df.loc[idx, "est.annotations"] + \
            ", " + annotationMessage
    else:
        df.loc[idx, ["est.annotations"]] = annotationMessage

    return df


def estimate_local_time(df):

    timezone_aliases = pd.read_csv(
        "wikipedia-timezone-aliases-2018-04-28.csv",
        low_memory=False
    )

    # fix large timzoneOffset bug in utcbootstrapping
    df = timezone_offset_bug_fix(df.copy())

    # add healthkit timezome information
    # TODO: refactor this function to only require fields that might have hk tz
    df[["timezone", "deviceType"]] = get_healthkit_timezone(df.copy())

    # convert deprecated timezones to their aliases
    df = convert_deprecated_timezone_to_alias(df, timezone_aliases)

    #  explicity define utc time
    df["utcTime"] = to_utc_datetime(df[["time"]].copy())
    df["date"] = df["utcTime"].dt.date  # TODO: change this to utcDate later
    contiguous_days = create_contiguous_day_series(df)

    df["deviceType"] = add_device_type(df)
    cDays = add_device_day_series(df, contiguous_days, "upload")

    # create day series for cgm df
    if "timezoneOffset" not in list(df):
        df["timezoneOffset"] = np.nan

    cgmdf = df[(df["type"] == "cbg") & (df["timezoneOffset"].notnull())].copy()
    cDays = add_device_day_series(cgmdf, cDays, "cgm")

    # create day series for pump df
    pumpdf = df[(df.type == "bolus") & (df.timezoneOffset.notnull())].copy()
    cDays = add_device_day_series(pumpdf, cDays, "pump")

    # interpolate between upload records of the same deviceType, and create a
    # day series for interpolated pump, non-hk-cgm, and healthkit uploads
    for deviceType in ["pump", "cgm", "healthkit"]:
        tempUploaddf = df[df["deviceType"] == deviceType].copy()
        cDays = impute_upload_records(
            tempUploaddf, cDays, deviceType + ".upload.imputed"
        )

    # add a home timezone that also accounts for daylight savings time changes
    cDays = add_home_timezone(df, cDays)

    # 1. USE UPLOAD RECORDS TO ESTIMATE TZ AND TZO
    cDays = estimateTzAndTzoWithUploadRecords(cDays)

    # 2. USE DEVICE TZOs TO ESTIMATE TZO AND TZ (IF POSSIBLE)
    # estimates can be made from pump and cgm df that have a TZO
    # NOTE: the healthkit and dexcom-api cgm df are excluded
    cDays = estimateTzAndTzoWithDeviceRecords(cDays)

    # 3. impute, infer, or interpolate gaps in the estimated tzo and tz
    cDays = imputeTzAndTzo(cDays)

    # 4. APPLY LOCAL TIME ESTIMATES TO ALL df
    df["est.localTime"] = applyLocalTimeEstimates(df, cDays)

    # round to the nearest second
    df["est.localTime"] = df["est.localTime"].dt.round("1s")

    # Convert est.localTime format to YYYY-MM-DDTHH:MM:SS string
    df["est.localTime"] = df['est.localTime'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    estimated_local_time_cols = [
        "date",
        "est.timezoneOffset",
        "est.type",
        "est.gapSize",
        "est.annotations",
        "est.timeProcessing",
    ]

    df = pd.merge(
        left=df,
        right=cDays[estimated_local_time_cols],
        how="left",
        on="date"
    )

    df["est.version"] = ESTIMATE_LOCAL_TIME_VERSION

    return df, cDays


# %% MAIN FUNCTION
if __name__ == "__main__":

    #  CHECK INPUTS AND OUTPUTS
    # check inputs and load data. File must be bigger than 1 KB,
    # and in either json, xlsx, or csv format
    input_path_and_name = "example-csv.csv"
    data, fileName = check_and_load_input_file(input_path_and_name)

    # estimate the local time
    data, local_time_metadata = estimate_local_time(data.copy())
