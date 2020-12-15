#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Donor Data Pipeline
===================
:File: donor_data_pipeline.py
:Description: A complete wrapper for all donor data pipeline functions
:Version: 0.0.1
:Created: 2019-09-28
:Authors: Jason Meno (jam)
:Dependencies:
    - A Tidepool .env file
    -
:License: BSD-2-Clause
"""

# %% Import libraries
import pandas as pd
import numpy as np
import datetime as dt
import time
import os
import argparse
import pickle
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Import custom local modules
import accept_new_donors_and_get_donor_list
import get_single_tidepool_dataset
import get_single_donor_metadata
import estimate_local_time_v0_4
import vector_qualify
import anonymize_and_export
import split_test_train_datasets
import anonymized_stats_qa
import dataset_summary_viz
import local_time_summary_viz


# %% Parse Argument Function
def get_args():
    codeDescription = "A complete wrapper for donor data pipeline functions"

    parser = argparse.ArgumentParser(description=codeDescription)

    parser.add_argument("-userid",
                        dest="userid",
                        default="",
                        help="Tidepool userid to download")

    parser.add_argument("-donor_group",
                        dest="donor_group",
                        default="",
                        help="Tidepool donor_group to download data from")

    parser.add_argument("-export_directory",
                        dest="export_directory",
                        default="",
                        help="Directory for pipeline results to be exported")

    parser.add_argument("-user_loc",
                        dest="user_loc",
                        default=0,
                        help="User location in a donor list if used")

    parser.add_argument("-import_data",
                        dest="importExistingData",
                        default="False",
                        help="Whether or not to import existing data")

    parser.add_argument("-save_new_data",
                        dest="saveDataDownload",
                        default="False",
                        help="Whether or not to save newly downloaded data")

    parser.add_argument("-data_path",
                        dest="data_path",
                        default="",
                        help="If importing data, the entire path to the data")

    parser.add_argument("-dataset_type",
                        dest="dataset_type",
                        default="all",
                        help="Type of dataset being processed \
                        (all, sap, hcl, pa, basalIQ, inpen). \
                        Default  \"all\" includes all data.")

    parser.add_argument("-test_set_days",
                        dest="test_set_days",
                        default=90,
                        help="The number of days in the test set")

    parser.add_argument("-add_tier_prefix",
                        dest="add_tier_prefix",
                        default="False",
                        help="Add a prefix to the .csv data (e.g. T1_, T2_)")

    parser.add_argument("-custom_start_date",
                        dest="custom_start_date",
                        default=np.nan,
                        help="Custom start date for anonymization")

    parser.add_argument("-custom_end_date",
                        dest="custom_end_date",
                        default=np.nan,
                        help="Custom end date for anonymization")

    parser.add_argument("-api_sleep_buffer",
                        dest="api_sleep_buffer",
                        default="False",
                        help="Adds a random 1-30 second sleep buffer before getting metadata")

    args = parser.parse_args()

    return args


# %% Helper Functions
def get_unique_donor_list(today_timestamp,
                          saveUniqueDonorList,
                          donor_list_file
                          ):

    phi_donor_list = []

    # If donor_list file is not defined, accept and get donors
    if donor_list_file == "":
        phi_donor_list = \
            accept_new_donors_and_get_donor_list.accept_and_get_list()

        phi_donor_list.columns = ['userid', 'donorGroup']

        if(saveUniqueDonorList):
            phi_donor_list.to_csv('PHI-unique-donor-list-' +
                                  today_timestamp +
                                  '.csv')
    else:
        try:
            phi_donor_list = \
                pd.read_csv(donor_list_file, low_memory=False)

        except Exception as e:
            print("Could not read donor list from: " +
                  donor_list_file + "\n" +
                  str(e)
                  )

    return phi_donor_list


def get_unique_length_values(data, postfix):
    """Get the unique lengths of items in each column and their values

    Useful for finding changes in dataframes after processing

    """

    unique_len_values = pd.DataFrame([], index=[np.arange(300)])
    for col in list(data):
        # print(col)

        unique_lengths = \
            data[col].\
            astype(str).\
            str.\
            len().\
            drop_duplicates().\
            sort_values(ascending=True)

        col_vals_by_len = \
            data.loc[unique_lengths.index, col].astype(str).values

        if(len(unique_lengths) > 300):
            unique_lengths = unique_lengths[:300]
            col_vals_by_len = col_vals_by_len[:300]

        unique_len_values.loc[range(len(col_vals_by_len)),
                              col +
                              "--" +
                              postfix +
                              ".LEN"
                              ] = unique_lengths.values

        unique_len_values.loc[range(len(col_vals_by_len)),
                              col +
                              "--" +
                              postfix] = col_vals_by_len

    return unique_len_values


def get_dataframe_changes(begin_df, end_df, begin_postfix, end_postfix):

    begin_columns = [col.replace("--" + begin_postfix + ".LEN", "")
                     for col in list(begin_df)]
    begin_columns = [col.replace("--" + begin_postfix, "")
                     for col in begin_columns]

    end_columns = [col.replace("--" + end_postfix + ".LEN", "")
                   for col in list(end_df)]

    end_columns = [col.replace("--" + end_postfix, "")
                   for col in end_columns]

    begin_columns = set(begin_columns)
    end_columns = set(end_columns)

    removed_columns = begin_columns - end_columns
    added_columns = end_columns - begin_columns

    df_changes = pd.concat([begin_df, end_df], sort=True)

    for removed_col in removed_columns:
        df_changes[removed_col + "--" + end_postfix + ".LEN"] = "REMOVED"
        df_changes[removed_col + "--" + end_postfix] = "REMOVED"

    for added_col in added_columns:
        df_changes[added_col + "--" + begin_postfix + ".LEN"] = "ADDED"
        df_changes[added_col + "--" + begin_postfix] = "ADDED"

    df_changes = \
        df_changes.\
        drop_duplicates().\
        reindex(sorted(df_changes.columns), axis=1)

    for col_idx in range(0, len(list(df_changes)), 2):
        value_col = list(df_changes)[col_idx]
        len_col = list(df_changes)[col_idx+1]
        df_changes[[value_col, len_col]] = \
            df_changes[[value_col, len_col]].sort_values(by=len_col).values

    return df_changes


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
    daily_cgm_snr = pd.DataFrame(daily_cgm_snr, columns=['daily_cgm_snr']).reset_index(drop=False)

    return daily_cgm_snr


def get_basal_duration_sum_days(basal_df):
    # Healthkit basals do not have a deviceTime
    # basal_df['date'] = pd.to_datetime(basal_df['deviceTime']).dt.date
    basal_df['date'] = pd.to_datetime(basal_df['time']).dt.date
    daily_duration_sum = basal_df.groupby('date').apply(lambda x: x['duration'].astype(float).sum()/1000/60/60)
    daily_duration_sum = pd.DataFrame(daily_duration_sum, columns=['daily_duration_hours_sum']).reset_index(drop=False)

    return daily_duration_sum


def visualize_dropped_data(keep_df, drop_df, export_directory, hashID):

    keep_cgm = keep_df[keep_df['type'] == 'cbg'].copy()
    keep_cgm_snr = get_cgm_noise_days(keep_cgm)
    drop_cgm = drop_df[drop_df['type'] == 'cbg']

    keep_bolus = keep_df[keep_df['type'] == 'bolus']
    drop_bolus = drop_df[drop_df['type'] == 'bolus']

    keep_basal = keep_df[keep_df['type'] == 'basal'].copy()
    keep_basal_daily_duration_sums =  get_basal_duration_sum_days(keep_basal)

    drop_basal = drop_df[drop_df['type'] == 'basal']

    if(len(keep_basal)>0):

        fig = make_subplots(
                rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02
        )

        # Add the Basal traces
        fig.add_trace(go.Scattergl(x=pd.to_datetime(keep_basal['time'], utc=True),
                                   y=keep_basal['rate'],
                                   mode="markers",
                                   name="keep_basal",
                                   marker=dict(size=12,
                                               symbol="circle-open")),
                      row=3, col=1)

        fig.add_trace(go.Scattergl(x=pd.to_datetime(drop_basal['time'], utc=True),
                                   y=drop_basal['rate'],
                                   mode="markers",
                                   name="drop_basal",
                                   marker=dict(size=8,
                                               symbol="cross")),
                      row=3, col=1)

        fig.add_trace(go.Scattergl(x=keep_cgm_snr['date'],
                               y=keep_cgm_snr['daily_cgm_snr'],
                               mode="markers",
                               name="daily_cgm_snr",
                               marker=dict(size=12,
                                           symbol="circle-open")),
                  row=4, col=1)

        fig.add_trace(go.Scattergl(x=keep_basal_daily_duration_sums['date'],
                               y=keep_basal_daily_duration_sums['daily_duration_hours_sum'],
                               mode="markers",
                               name="daily_duration_hours_sum",
                               marker=dict(size=12,
                                           symbol="circle-open")),
                  row=5, col=1)

    else:

        fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02
        )

    # Add the CGM traces
    fig.add_trace(go.Scattergl(x=pd.to_datetime(keep_cgm['time'], utc=True),
                               y=keep_cgm['value'],
                               mode="markers",
                               name="keep_cgm",
                               marker=dict(size=12,
                                           symbol="circle-open")),
                  row=1, col=1)

    fig.add_trace(go.Scattergl(x=pd.to_datetime(drop_cgm['time'], utc=True),
                               y=drop_cgm['value'],
                               mode="markers",
                               name="drop_cgm",
                               marker=dict(size=8,
                                           symbol="cross")),
                  row=1, col=1)

    # Add the Bolus traces
    fig.add_trace(go.Scattergl(x=pd.to_datetime(keep_bolus['time'], utc=True),
                               y=keep_bolus['normal'],
                               mode="markers",
                               name="keep_bolus",
                               marker=dict(size=12,
                                           symbol="circle-open")),
                  row=2, col=1)

    fig.add_trace(go.Scattergl(x=pd.to_datetime(drop_bolus['time'], utc=True),
                               y=drop_bolus['normal'],
                               mode="markers",
                               name="drop_bolus",
                               marker=dict(size=8,
                                           symbol="cross")),
                  row=2, col=1)

    fig.update_layout(title_text=hashID + "\n Kept/Dropped Data")
    plotly_filename = \
        export_directory + \
        "/QA/vizQA/plotly-dropped-data-vizQA/" + \
        hashID+"-plotly-dropped-data-vizQA.html"

    plot(fig, filename=plotly_filename, auto_open=False)

    return


def create_summary_metadata(file_name,
                            data_info,
                            df,
                            vector_stats,
                            dataset_type,
                            qualifiedStartDate,
                            qualifiedEndDate):

    df['date'] = df['time'].str.split('T', expand=True)[0].astype(str)
    firstDate = df['date'].min()
    lastDate = df['date'].max()

    qualifiedDaySpan = len(pd.date_range(qualifiedStartDate, qualifiedEndDate, freq='d'))

    if(dataset_type == "all"):
        qualifiedDaysWithData = \
            len(df.loc[(df['date'] >= qualifiedStartDate) & \
                       (df['date'] <= qualifiedEndDate), 'date'].unique())
    else:
        qualifiedDaysWithData = \
            vector_stats.loc[(vector_stats['date'] >= qualifiedStartDate) & \
                             (vector_stats['date'] <= qualifiedEndDate),
                              'is_' + dataset_type].sum()

    percentDaysWithQualifiedData = qualifiedDaysWithData / qualifiedDaySpan

    birthDate = data_info['birthday'][0]
    diagnosisDate = data_info['diagnosisDate'][0]

    ageStart = np.floor((pd.to_datetime(qualifiedStartDate) - pd.to_datetime(birthDate)).days/365)
    ageEnd = np.floor((pd.to_datetime(qualifiedEndDate) - pd.to_datetime(birthDate)).days/365)
    yearsLivingWithDiabetesStart = np.floor((pd.to_datetime(qualifiedStartDate) - pd.to_datetime(diagnosisDate)).days/365)
    yearsLivingWithDiabetesEnd = np.floor((pd.to_datetime(qualifiedEndDate) - pd.to_datetime(diagnosisDate)).days/365)
    diagnosisType = data_info['diagnosisType'][0]
    biologicalSex = data_info['biologicalSex'][0]

    summary_metadata = pd.DataFrame([
                        file_name,
                        firstDate,
                        lastDate,
                        qualifiedStartDate,
                        qualifiedEndDate,
                        qualifiedDaysWithData,
                        percentDaysWithQualifiedData,
                        ageStart,
                        ageEnd,
                        yearsLivingWithDiabetesStart,
                        yearsLivingWithDiabetesEnd,
                        diagnosisType,
                        biologicalSex]).T

    summary_metadata.columns = ['file_name',
                                'firstDate',
                                'lastDate',
                                "qualifiedStartDate",
                                "qualifiedEndDate",
                                "qualifiedDaysWithData",
                                "percentDaysWithQualifiedData",
                                'ageStart',
                                'ageEnd',
                                'yearsLivingWithDiabetesStart',
                                'yearsLivingWithDiabetesEnd',
                                'diagnosisType',
                                'biologicalSex']

    df.drop(columns='date', inplace=True)

    return summary_metadata


# %% MAIN PIPELINE WRAPPER

def pipeline_wrapper(userid,
                     donor_group,
                     export_directory,
                     user_loc,
                     importExistingData,
                     saveDataDownload,
                     data_path,
                     dataset_type,
                     test_set_days,
                     add_tier_prefix,
                     custom_start_date,
                     custom_end_date,
                     api_sleep_buffer):

    ##################
    # PIPELINE START #
    ##################
    # print(current_process().name)
    print(str(user_loc) + " STARTING PIPELINE")
    data = pd.DataFrame([])
    pipeline_metadata = pd.DataFrame([], index=[user_loc])

    if userid == "":
      print("No UserID Provided!")
      userid = input("Please enter a Tidepool userid:\n")

    pipeline_metadata['userid'] = userid
    dataset_begin_time = time.time()

    ################
    # GET METADATA #
    ################

    startTime = time.time()

    print(str(user_loc) + " sleeping and getting profile metadata...", end="")

    if(api_sleep_buffer):
      # Sleep for 0-30 seconds before getting metadata
      # This helps reduce the possibility of 504 timeouts
      # from too many multiprocessing requests
      time.sleep(np.random.randint(30))

    metadata, _ = get_single_donor_metadata.get_metadata(
        donor_group=donor_group,
        userid_of_shared_user=userid
    )

    get_metadata_runtime = round(time.time() - startTime, 1)

    print("done, took", str(get_metadata_runtime), "seconds")
    pipeline_metadata['get_metadata_runtime'] = get_metadata_runtime


    ################
    # GETTING DATA #
    ################
    startTime = time.time()

    if(importExistingData):
        print(str(user_loc) + " Importing dataset...", end="")

        try:
            data = pd.read_csv(data_path, low_memory=False)
        except Exception as e:
            print(e)
            print("No dataset exists at: " + data_path)
            print("\n Check if importExistingData = True")

        import_data_runtime = round(time.time() - startTime, 1)
        print("done, took", str(import_data_runtime), "seconds")
        pipeline_metadata['import_data_runtime'] = import_data_runtime
        pipeline_metadata['download_data_runtime'] = np.nan
        pipeline_metadata['save_data_runtime'] = np.nan

    else:
        print(str(user_loc) + " Downloading dataset...", end="")

        data = get_single_tidepool_dataset.get_and_return_dataset(
            donor_group=donor_group,
            userid_of_shared_user=userid
        )

        download_data_runtime = round(time.time() - startTime, 1)
        print("done, took", str(download_data_runtime), "seconds")
        pipeline_metadata['download_data_runtime'] = download_data_runtime
        pipeline_metadata['import_data_runtime'] = np.nan

        if(saveDataDownload):
            startTime = time.time()
            print("Saving dataset...", end="")

            csv_dir = export_directory + "/PHI-csvData/"
            data_path = csv_dir + "PHI-" + userid + ".csv.gz"

            data.to_csv(data_path, index=False, compression='gzip')

            save_data_runtime = round(time.time() - startTime, 1)
            print("done, took", str(save_data_runtime), "seconds")
            pipeline_metadata['save_data_runtime'] = save_data_runtime
        else:
            pipeline_metadata['save_data_runtime'] = np.nan

    if len(data) == 0:
        print("No Data Exists!")
        #return None

    #######################
    # ESTIMATE LOCAL TIME #
    #######################
    startTime = time.time()
    print(str(user_loc) + " Estimating Local Time...", end="")

    # data_old_lte, cDays_old_lte = estimate_local_time.estimate_local_time(data.copy())
    # data_new_lte, cDays_new_lte = estimate_local_time_v0_4.estimate_local_time(data.copy())
    data, cDays_LTE = estimate_local_time_v0_4.estimate_local_time(data)
    qa_LTE_cDays_path = \
        export_directory + \
        "/QA/PHI-LTE-cDays/" + \
        "PHI-" + userid + "-LTE-cDays.csv"

    cDays_LTE.to_csv(qa_LTE_cDays_path, index=False)
    estlocaltime_runtime = round(time.time() - startTime, 1)

    print("done, took", str(estlocaltime_runtime), "seconds")
    pipeline_metadata['estlocaltime_runtime'] = estlocaltime_runtime

    qualify_metadata = pd.DataFrame(["original qualify script not run"])
    ##########################
    # VECTOR QUALIFY DATASET #
    ##########################
    startTime = time.time()
    print(str(user_loc) + " Qualifying dataset (Vectors)...", end="")

    # Sort data first in ascending order for vector qualify
    data.sort_values(by='time', ascending=True, inplace=True)
    data.reset_index(inplace=True, drop=True)

    file_name = "PHI-" + userid + ".csv.gz"

    vector_stats, vector_qualify_metadata = \
        vector_qualify.get_vector_summary(data, file_name)

    qa_vector_qualified_cDays_path = \
        export_directory + \
        "/QA/PHI-vector-qualified-days/" + \
        "PHI-" + userid + "-vector-qualified-days.csv"

    vector_stats.to_csv(qa_vector_qualified_cDays_path, index=False)

    vector_qualify_data_runtime = round(time.time() - startTime, 1)

    print("done, took", str(vector_qualify_data_runtime), "seconds")
    pipeline_metadata['vector_qualify_data_runtime'] = \
        vector_qualify_data_runtime

    if(dataset_type == "all"):
        anonymized_start_date = vector_qualify_metadata['firstDay'][0]
        anonymized_end_date = vector_qualify_metadata['lastDay'][0]
    else:
        anonymized_start_date = vector_qualify_metadata[dataset_type + '_start'][0]
        anonymized_end_date = vector_qualify_metadata[dataset_type +'_end'][0]

    if(pd.notnull(custom_start_date) & pd.notnull(custom_end_date)):
        anonymized_start_date = custom_start_date
        anonymized_end_date = custom_end_date

    # Checkpoint for getting the unique "beginning" dataframe
    begin_postfix = "BEGIN"
    begin_df = \
        get_unique_length_values(data, begin_postfix)

    #####################
    # ANONYMIZE DATASET #
    #####################
    startTime = time.time()
    print(str(user_loc) + " Anonymizing dataset...", end="")

    anonymized_data, \
        hashID, \
        anonymized_metadata, \
        removed_columns, \
        remaining_columns = \
            anonymize_and_export.anonymize_data(data,
                                                userid,
                                                anonymized_start_date,
                                                anonymized_end_date)

    anonymize_data_runtime = round(time.time() - startTime, 1)

    print("done, took", str(anonymize_data_runtime), "seconds")
    pipeline_metadata['anonymize_data_runtime'] = anonymize_data_runtime
    pipeline_metadata['hashid'] = hashID

    #######################
    # GET DATASET CHANGES #
    #######################
    end_postfix = "END"
    end_df = \
        get_unique_length_values(anonymized_data, end_postfix)

    df_changes = get_dataframe_changes(begin_df,
                                       end_df,
                                       begin_postfix,
                                       end_postfix)

    dropped_data = data[~(data.id.isin(anonymized_data.id))]

    visualize_dropped_data(anonymized_data, dropped_data, export_directory, hashID)

    del data

    startTime = time.time()

    #################
    # SPLIT DATASET #
    #################
    if(test_set_days > 0):
        print(str(user_loc) + " Splitting dataset...", end="")
        train_df, \
            test_df, \
            split_date = \
                split_test_train_datasets.split_dataset(anonymized_data,
                                                        test_set_days)

        train_filename = "train_" + hashID + ".csv"
        test_filename = "test_" + hashID + ".csv"

        qualifiedTrainStart = anonymized_start_date
        qualifiedTrainEnd = (pd.to_datetime(split_date) - pd.Timedelta(1, 'D')).strftime('%Y-%m-%d')

        qualifiedTestStart = split_date
        qualifiedTestEnd = anonymized_end_date

    else:
        train_df = anonymized_data.copy()
        test_df = pd.DataFrame(index=[0], columns=list(train_df))

        qualifiedTrainStart = anonymized_start_date
        qualifiedTrainEnd = anonymized_end_date

        # Test dataset is empty and these dates are filler
        qualifiedTestStart = anonymized_start_date
        qualifiedTestEnd = anonymized_end_date

        if(add_tier_prefix):
            tier_prefix_letter = 'T'
            tier_names = np.array([tier_prefix_letter + str(num) for num in np.arange(7)])
            tier_min_days = np.array([0, 30, 100, 200, 366, 731, 1096])
            dataset_days = vector_qualify_metadata[dataset_type + '_days'][0]
            tier_loc = np.argmax(tier_min_days[dataset_days >= tier_min_days])
            dataset_tier = tier_names[tier_loc]

            train_filename = dataset_tier + "_" + hashID + ".csv"
            test_filename = "EMPTY_TEST_" + hashID + ".csv"

        else:
            train_filename = hashID + ".csv"
            test_filename = "EMPTY_TEST_" + hashID + ".csv"

        pipeline_metadata['file_name'] = train_filename

    train_summary_metadata = \
        create_summary_metadata(train_filename,
                                metadata,
                                train_df,
                                vector_stats,
                                dataset_type,
                                qualifiedTrainStart,
                                qualifiedTrainEnd)

    test_summary_metadata = \
        create_summary_metadata(test_filename,
                                metadata,
                                test_df,
                                vector_stats,
                                dataset_type,
                                qualifiedTestStart,
                                qualifiedTestEnd)

    split_data_runtime = round(time.time() - startTime, 1)

    print("done, took", str(split_data_runtime), "seconds")
    pipeline_metadata['split_data_runtime'] = split_data_runtime

    del anonymized_data

    ##########################
    # EXPORT TRAIN/TEST DATA #
    ##########################

    startTime = time.time()
    print(str(user_loc) + " Exporting train/test datasets...", end="")

    train_df.to_csv(export_directory +
                    "/train/train-data/" +
                    train_filename, index=False)

    test_df.to_csv(export_directory +
                   "/test/test-data/" +
                   test_filename, index=False)

    export_train_test_data_runtime = round(time.time() - startTime, 1)

    print("done, took", str(export_train_test_data_runtime), "seconds")
    pipeline_metadata['export_train_test_data_runtime'] = \
        export_train_test_data_runtime

    ############
    # QA STATS #
    ############

    startTime = time.time()
    print(str(user_loc) + " Calculating train QA stats...", end="")

    train_anonymized_stats_qa = \
        anonymized_stats_qa.get_anonymized_stats(train_df, train_filename)

    train_qa_stats_runtime = round(time.time() - startTime, 1)

    print("done, took", str(train_qa_stats_runtime), "seconds")
    pipeline_metadata['train_qa_stats_runtime'] = train_qa_stats_runtime

    startTime = time.time()
    print(str(user_loc) + " Calculating test QA stats...", end="")
    test_anonymized_stats_qa = \
        anonymized_stats_qa.get_anonymized_stats(test_df, test_filename)

    test_qa_stats_runtime = round(time.time() - startTime, 1)

    print("done, took", str(test_qa_stats_runtime), "seconds")
    pipeline_metadata['test_qa_stats_runtime'] = test_qa_stats_runtime

    #############################
    # VISUALIZE DATASET SUMMARY #
    #############################

    startTime = time.time()
    print(str(user_loc) +
          " Creating summary viz QA figures for train/test data...", end="")

    qa_train_summary_viz_dir = export_directory + \
        "/QA/vizQA/train-dataset-summary-vizQA/"

    qa_test_summary_viz_dir = export_directory + \
        "/QA/vizQA/test-dataset-summary-vizQA/"

    qa_train_local_time_viz_dir = export_directory + \
        "/QA/vizQA/train-local-time-vizQA/"

    qa_test_local_time_viz_dir = export_directory + \
        "/QA/vizQA/test-local-time-vizQA/"

    qa_train_vector_qualify_viz_dir = export_directory + \
        "/QA/vizQA/train-vector-qualify-vizQA/"

    qa_test_vector_qualify_viz_dir = export_directory + \
        "/QA/vizQA/test-vector-qualify-vizQA/"

    dataset_summary_viz.data_summary_viz(train_df,
                                         qa_train_summary_viz_dir,
                                         train_filename)

    if(len(test_df) > 1):
        dataset_summary_viz.data_summary_viz(test_df,
                                             qa_test_summary_viz_dir,
                                             test_filename)

    summary_viz_qa_runtime = round(time.time() - startTime, 1)
    print("done, took", str(summary_viz_qa_runtime), "seconds")

    pipeline_metadata['summary_viz_qa_runtime'] = summary_viz_qa_runtime

    ################################
    # VISUALIZE LOCAL TIME OFFSETS #
    ################################

    startTime = time.time()
    print(str(user_loc) +
          " Creating local time viz QA figures for train/test data...", end="")

    local_time_summary_viz.local_time_summary_viz(train_df,
                                                  qa_train_local_time_viz_dir,
                                                  train_filename)
    if(len(test_df) > 1):
        local_time_summary_viz.local_time_summary_viz(test_df,
                                                  qa_test_local_time_viz_dir,
                                                  test_filename)

    local_time_viz_qa_runtime = round(time.time() - startTime, 1)

    print("done, took", str(local_time_viz_qa_runtime), "seconds")

    pipeline_metadata['local_time_viz_qa_runtime'] = local_time_viz_qa_runtime

    #####################################
    # VISUALIZE VECTORIZED QUALIFY DATA #
    #####################################

    startTime = time.time()
    print(str(user_loc) + " Creating vector_qualify viz " +
          "QA figures for train/test data...", end="")

    vector_qualify.vector_stats_viz(train_df,
                                    qa_train_vector_qualify_viz_dir,
                                    train_filename)
    if(len(test_df) > 1):
        vector_qualify.vector_stats_viz(test_df,
                                    qa_test_vector_qualify_viz_dir,
                                    test_filename)

    vector_qualify_viz_qa_runtime = round(time.time() - startTime, 1)

    print("done, took", str(vector_qualify_viz_qa_runtime), "seconds")

    pipeline_metadata['vector_qualify_viz_qa_runtime'] = \
        vector_qualify_viz_qa_runtime

    ####################
    # PIPELINE WRAP-UP #
    ####################
    complete_runtime = round(round(time.time() - dataset_begin_time, 1)/60, 2)

    print("Dataset " +
          str(user_loc) +
          " completed in " +
          str(complete_runtime) +
          " minutes at " +
          dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") +
          "")

    pipeline_metadata['complete_runtime_minutes'] = complete_runtime

    results_array = [remaining_columns,
                     removed_columns,
                     qualify_metadata,
                     vector_qualify_metadata,
                     anonymized_metadata,
                     train_summary_metadata,
                     test_summary_metadata,
                     train_anonymized_stats_qa,
                     test_anonymized_stats_qa,
                     pipeline_metadata,
                     df_changes
                     ]

    results_export_path = \
        export_directory + \
        '/PHI-pipeline-results/' + \
        'PHI-' + userid + "-pipeline-results.data"

    # Write mixed data list results to pickle
    with open(results_export_path, 'wb') as pickle_file:
        pickle.dump(results_array, pickle_file)

    done_statement = str(user_loc) + " is complete and returned successfully"

    return done_statement

# %%

if __name__ == "__main__":

    pipeline_args = get_args()
    
    export_directory = pipeline_args.export_directory

    # If no export directory is provided, it will be created
    if export_directory == "":
      today_timestamp = dt.datetime.now().strftime("%Y-%m-%d")
      export_directory = "PHI-pipeline-export-" + today_timestamp
      pipeline_args.export_directory = export_directory

    pipeline_results_dir = export_directory + '/PHI-pipeline-results/'
    train_data_dir = export_directory + "/train/train-data/"
    test_data_dir = export_directory + "/test/test-data/"

    csv_dir = export_directory + "/PHI-csvData/"
    qa_dir = export_directory + "/QA/"
    qa_LTE_cDays_dir = qa_dir + "PHI-LTE-cDays/"
    qa_vector_qualified_cDays_dir = qa_dir + "PHI-vector-qualified-days/"
    qa_train_summary_viz_dir = qa_dir + "vizQA/train-dataset-summary-vizQA/"
    qa_test_summary_viz_dir = qa_dir + "vizQA/test-dataset-summary-vizQA/"
    qa_train_local_time_viz_dir = qa_dir + "vizQA/train-local-time-vizQA/"
    qa_test_local_time_viz_dir = qa_dir + "vizQA/test-local-time-vizQA/"
    qa_train_vector_qualify_viz_dir = qa_dir + \
        "vizQA/train-vector-qualify-vizQA/"
    qa_test_vector_qualify_viz_dir = qa_dir + \
        "vizQA/test-vector-qualify-vizQA/"
    qa_plotly_dropped_data_viz_dir = qa_dir + \
     "vizQA/plotly-dropped-data-vizQA/"

    directories = [pipeline_results_dir,
                   train_data_dir,
                   test_data_dir,
                   csv_dir,
                   qa_dir,
                   qa_LTE_cDays_dir,
                   qa_vector_qualified_cDays_dir,
                   qa_train_summary_viz_dir,
                   qa_test_summary_viz_dir,
                   qa_train_local_time_viz_dir,
                   qa_test_local_time_viz_dir,
                   qa_train_vector_qualify_viz_dir,
                   qa_test_vector_qualify_viz_dir,
                   qa_plotly_dropped_data_viz_dir
                   ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    pipeline_wrapper(pipeline_args.userid,
                     pipeline_args.donor_group,
                     pipeline_args.export_directory,
                     int(pipeline_args.user_loc),
                     eval(pipeline_args.importExistingData),
                     eval(pipeline_args.saveDataDownload),
                     pipeline_args.data_path,
                     pipeline_args.dataset_type,
                     int(pipeline_args.test_set_days),
                     eval(pipeline_args.add_tier_prefix),
                     pipeline_args.custom_start_date,
                     pipeline_args.custom_end_date,
                     eval(pipeline_args.api_sleep_buffer))
