import pandas as pd
import numpy as np
import os
import time
import datetime as dt

#%%
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
    last_upload_time = df.loc[df['uploadId'].isin(uploadIds_missing_uploadRecords), ['uploadId', 'time']].groupby('uploadId').time.max()
    last_upload_time = pd.DataFrame(last_upload_time).reset_index()
    last_upload_time.columns = ["uploadId", "uploadTime"]
    uploadTimes = pd.concat([uploadTimes, last_upload_time]).reset_index(drop=True)

    df = pd.merge(df, uploadTimes, how='left', on='uploadId')
    df["uploadTime"] = pd.to_datetime(df["uploadTime"])

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


def snr(cgm_ts):
    cgm_signal = cgm_ts.rolling(window=3, center=True).mean()
    cgm_noise = cgm_ts - cgm_signal
    rms_cgm_signal = np.sqrt(np.mean(cgm_signal**2))
    rms_cgm_noise = np.sqrt(np.mean(cgm_noise**2))
    snr_cgm = (rms_cgm_signal / rms_cgm_noise) ** 2
    snr_cgm_dB = 10 * np.log10(snr_cgm)
    return snr_cgm_dB


def get_cgm_noise_days(cgm_df):
    cgm_df['date'] = pd.to_datetime(cgm_df['time']).dt.date
    daily_cgm_snr = cgm_df.groupby('date').apply(lambda x: snr(x['value']))
    cgm_days_below_threshold = (daily_cgm_snr < 25).sum()

    return cgm_days_below_threshold


def get_basal_durations_days(basal_df):
    # Healthkit basals do not have a deviceTime
    # basal_df['date'] = pd.to_datetime(basal_df['deviceTime']).dt.date
    basal_df['date'] = pd.to_datetime(basal_df['time']).dt.date
    daily_duration_sum = basal_df.groupby('date').apply(lambda x: x['duration'].astype(float).sum()/1000/60/60)
    basal_days_above_threshold = (daily_duration_sum > 36).sum()

    return basal_days_above_threshold


def get_data_stats(data_df, file_name):
    #print(file_loc)
    #file_name = file_list[file_loc]
    #data_df = []

    firstDay = np.nan
    lastDay = np.nan
    daySpan = np.nan
    uniqueDays = np.nan

    first_cgm_pump_day = np.nan
    last_cgm_pump_day = np.nan
    cgm_pump_day_range = np.nan
    actual_cgm_pump_days = np.nan
    percent_cgm_pump_days = np.nan

    physical_activity_entries = np.nan

    max_temp_basal_count = np.nan
    mean_temp_basal_count = np.nan
    max_suspend_basal_count = np.nan
    mean_suspend_basal_count = np.nan
    max_scheduled_basal_count = np.nan
    mean_scheduled_basal_count = np.nan
    contains_HCL = False
    basal_duplicates = np.nan
    deduplicated_max_temp_basal_count = np.nan
    deduplicated_mean_temp_basal_count = np.nan
    deduplicated_max_suspend_basal_count = np.nan
    deduplicated_mean_suspend_basal_count = np.nan
    deduplicated_max_scheduled_basal_count = np.nan
    deduplicated_mean_schedule_basal_count = np.nan
    deduplicated_contains_HCL = False

    max_bolus_count = np.nan
    mean_bolus_count = np.nan
    bolus_duplicates = np.nan
    deduplicated_max_bolus_count = np.nan
    deduplicated_mean_bolus_count = np.nan

    max_cgm_count = np.nan
    mean_cgm_percent = np.nan
    cgm_days_lte288 = np.nan
    cgm_days_gt288 = np.nan
    cgm_duplicates = np.nan
    deduplicated_max_cgm_count = np.nan
    deduplicated_mean_cgm_percent = np.nan

    qualify_cgm_duplicates = np.nan
    qualify_deduplicated_daily_cgm_count = np.nan
    qualify_deduplicated_max_cgm_count = np.nan
    qualify_deduplicated_mean_cgm_percent = np.nan

    nDuplicatesRemovedDeviceTime = np.nan
    nDuplicatesRemovedUtcTime = np.nan
    nDuplicatesRemovedRoundedTime = np.nan
    nDuplicatesRemovedLocalTime = np.nan
    nDuplicatesRemovedRoundedLocalTime = np.nan

    basal_duration_days = np.nan
    cgm_noise_days = np.nan

    if(len(data_df) > 0):

        data_df = add_uploadDateTime(data_df)

        data_df['date'] = data_df['time'].str.split('T', expand=True)[0]
        firstDay = data_df['date'].min()
        lastDay = data_df['date'].max()
        daySpan = (pd.to_datetime(lastDay)-pd.to_datetime(firstDay)).days
        uniqueDays = len(set(data_df['date'].values))

        typesPerDay = pd.DataFrame(
            data_df.groupby('date').apply(lambda x: np.unique(list(x['type']))),
                    columns=['type_list']
                    ).reset_index()

        data_types = set(data_df['type'].values)

        for data_type in data_types:
            typesPerDay[data_type] = \
                typesPerDay['type_list'].apply(lambda x: data_type in x)

        if (('cbg' in data_types) & ('bolus' in data_types) & ('basal' in data_types)):
            typesPerDay['cgm+pump'] = typesPerDay['cbg'] & typesPerDay['bolus'] & typesPerDay['basal']
            cgm_pump_days = typesPerDay.loc[typesPerDay['cgm+pump'],'date']

            first_cgm_pump_day = cgm_pump_days.min()
            last_cgm_pump_day = cgm_pump_days.max()
            cgm_pump_day_range = sum((typesPerDay['date'] >= first_cgm_pump_day) & (typesPerDay['date'] <= last_cgm_pump_day))
            actual_cgm_pump_days = typesPerDay['cgm+pump'].sum()
            percent_cgm_pump_days = actual_cgm_pump_days/cgm_pump_day_range

        if 'type' in list(data_df):
            physical_activity_entries = len(data_df[data_df['type'] == 'physicalActivity'])

        # Check if bolus type and origin column name exists
        if ('basal' in set(data_df['type'].values)):
            basal_df = data_df[data_df['type']=='basal'].copy()

            basal_duration_days = get_basal_durations_days(basal_df)
            daily_temp_basal = basal_df.loc[basal_df['deliveryType']=='temp'].groupby('date').apply(lambda x: len(x))

            if(len(daily_temp_basal > 0)):
                max_temp_basal_count = daily_temp_basal.max()
                mean_temp_basal_count = daily_temp_basal.mean()
            else:
                max_temp_basal_count = 0
                mean_temp_basal_count = 0

            contains_HCL = any(daily_temp_basal > 30)

            daily_suspend_basal = basal_df[basal_df['deliveryType']=='suspend'].groupby('date').apply(lambda x: len(x))
            max_suspend_basal_count = daily_suspend_basal.max()
            mean_suspend_basal_count = daily_suspend_basal.mean()

            daily_scheduled_basal = basal_df[basal_df['deliveryType']=='scheduled'].groupby('date').apply(lambda x: len(x))
            max_scheduled_basal_count = daily_scheduled_basal.max()
            mean_scheduled_basal_count = daily_scheduled_basal.mean()

            deduplicated_basal_df = basal_df.drop_duplicates(subset='time')
            basal_duplicates = len(basal_df) - len(deduplicated_basal_df)

            deduplicated_daily_temp_basal_count = deduplicated_basal_df.loc[deduplicated_basal_df['deliveryType']=='temp'].groupby('date').apply(lambda x: len(x))

            if(len(deduplicated_daily_temp_basal_count > 0)):
                deduplicated_max_temp_basal_count = deduplicated_daily_temp_basal_count.max()
                deduplicated_mean_temp_basal_count = deduplicated_daily_temp_basal_count.mean()
            else:
                deduplicated_max_temp_basal_count = 0
                deduplicated_mean_temp_basal_count = 0

            deduplicated_contains_HCL = any(deduplicated_daily_temp_basal_count > 30)

        if ('bolus' in set(data_df['type'].values)):
            bolus_df = data_df[data_df['type']=='bolus']

            daily_bolus_count = bolus_df.groupby('date').apply(lambda x: x["normal"].count())
            max_bolus_count = daily_bolus_count.max()
            mean_bolus_count = daily_bolus_count.mean()

            deduplicated_bolus_df = bolus_df.drop_duplicates(subset='time')
            bolus_duplicates = len(bolus_df) - len(deduplicated_bolus_df)
            deduplicated_daily_bolus_count = deduplicated_bolus_df.groupby('date').apply(lambda x: x["normal"].count())
            deduplicated_max_bolus_count = deduplicated_daily_bolus_count.max()
            deduplicated_mean_bolus_count = deduplicated_daily_bolus_count.mean()

        if ('cbg' in set(data_df['type'].values)):
            cgm_df = data_df[data_df['type']=='cbg'].copy()
            cgm_noise_days = get_cgm_noise_days(cgm_df)
            daily_cgm_count = cgm_df.groupby('date').apply(lambda x: x["value"].count())
            max_cgm_count = daily_cgm_count.max()
            mean_cgm_percent = daily_cgm_count.mean()/288
            cgm_days_lte288 = sum(daily_cgm_count <= 288)
            cgm_days_gt288 = sum(daily_cgm_count > 288)

            deduplicated_cgm_df = cgm_df.drop_duplicates(subset='time')
            cgm_duplicates = len(cgm_df) - len(deduplicated_cgm_df)
            deduplicated_daily_cgm_count = deduplicated_cgm_df.groupby('date').apply(lambda x: x["value"].count())
            deduplicated_max_cgm_count = deduplicated_daily_cgm_count.max()
            deduplicated_mean_cgm_percent = deduplicated_daily_cgm_count.mean()/288

            cgm_df_dedup_qualify, nDuplicatesRemovedDeviceTime = \
                removeCgmDuplicates(cgm_df, "deviceTime")
            cgm_df_dedup_qualify, nDuplicatesRemovedUtcTime = \
                removeCgmDuplicates(cgm_df_dedup_qualify, "time")
            cgm_df_dedup_qualify, nDuplicatesRemovedLocalTime = \
                removeCgmDuplicates(cgm_df_dedup_qualify, "est.localTime")

            cgm_df_dedup_qualify = round_time(
                    cgm_df_dedup_qualify,
                    timeIntervalMinutes=5,
                    timeField="time",
                    roundedTimeFieldName="roundedTime",
                    verbose=False
                )

            cgm_df_dedup_qualify, nDuplicatesRemovedRoundedTime = \
                removeCgmDuplicates(cgm_df_dedup_qualify, "roundedTime")

            cgm_df_dedup_qualify = round_time(
                    cgm_df_dedup_qualify,
                    timeIntervalMinutes=5,
                    timeField="est.localTime",
                    roundedTimeFieldName="roundedLocalTime",
                    verbose=False
                )

            cgm_df_dedup_qualify, nDuplicatesRemovedRoundedLocalTime = \
                removeCgmDuplicates(cgm_df_dedup_qualify, "roundedLocalTime")

            qualify_cgm_duplicates = len(cgm_df) - len(cgm_df_dedup_qualify)
            qualify_deduplicated_daily_cgm_count = cgm_df_dedup_qualify.groupby('date').apply(lambda x: x["value"].count())
            qualify_deduplicated_max_cgm_count = qualify_deduplicated_daily_cgm_count.max()
            qualify_deduplicated_mean_cgm_percent = qualify_deduplicated_daily_cgm_count.mean()/288

    return file_name, firstDay, lastDay, daySpan, uniqueDays, first_cgm_pump_day, last_cgm_pump_day, cgm_pump_day_range, actual_cgm_pump_days, percent_cgm_pump_days, physical_activity_entries, contains_HCL, max_temp_basal_count, mean_temp_basal_count, max_suspend_basal_count, mean_suspend_basal_count, max_scheduled_basal_count, mean_scheduled_basal_count, basal_duplicates, deduplicated_contains_HCL, deduplicated_max_temp_basal_count, deduplicated_mean_temp_basal_count, max_bolus_count, mean_bolus_count, bolus_duplicates, deduplicated_max_bolus_count, deduplicated_mean_bolus_count, max_cgm_count, mean_cgm_percent, cgm_days_lte288, cgm_days_gt288, cgm_duplicates, deduplicated_max_cgm_count, deduplicated_mean_cgm_percent, qualify_cgm_duplicates, qualify_deduplicated_max_cgm_count, qualify_deduplicated_mean_cgm_percent, nDuplicatesRemovedDeviceTime, nDuplicatesRemovedUtcTime, nDuplicatesRemovedRoundedTime, nDuplicatesRemovedLocalTime, nDuplicatesRemovedRoundedLocalTime, basal_duration_days, cgm_noise_days

def get_anonymized_stats(data, file_name):
    results_array = get_data_stats(data, file_name)
    # Convert results into dataframe
    results_df = pd.DataFrame(results_array).T
    column_names = ['file_name',
                    'firstDay',
                    'lastDay',
                    'daySpan',
                    'uniqueDays',
                    'first_cgm_pump_day',
                    'last_cgm_pump_day',
                    'cgm_pump_day_range',
                    'actual_cgm_pump_days',
                    'percent_cgm_pump_days',
                    'physical_activity_entries',
                    'contains_HCL',
                    'max_temp_basal_count',
                    'mean_temp_basal_count',
                    'max_suspend_basal_count',
                    'mean_suspend_basal_count',
                    'max_scheduled_basal_count',
                    'mean_scheduled_basal_count',
                    'basal_duplicates',
                    'deduplicated_contains_HCL',
                    'deduplicated_max_temp_basal_count',
                    'deduplicated_mean_temp_basal_count',
                    'max_bolus_count',
                    'mean_bolus_count',
                    'bolus_duplicates',
                    'deduplicated_max_bolus_count',
                    'deduplicated_mean_bolus_count',
                    'max_cgm_count',
                    'mean_cgm_percent',
                    'cgm_days_lte288',
                    'cgm_days_gt288',
                    'cgm_duplicates',
                    'deduplicated_max_cgm_count',
                    'deduplicated_mean_cgm_percent',
                    'qualify_cgm_duplicates',
                    'qualify_deduplicated_max_cgm_count',
                    'qualify_deduplicated_mean_cgm_percent',
                    'nDuplicatesRemovedDeviceTime',
                    'nDuplicatesRemovedUtcTime',
                    'nDuplicatesRemovedRoundedTime',
                    'nDuplicatesRemovedLocalTime',
                    'nDuplicatesRemovedRoundedLocalTime',
                    'basal_duration_days',
                    'cgm_noise_days']

    results_df.columns = column_names

    return results_df
    