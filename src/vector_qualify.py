import pandas as pd
import numpy as np
import os
import time
import datetime as datetime
import traceback
import sys
import matplotlib
import datetime as dt
# Hide Figures with Agg backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from pandas.plotting import register_matplotlib_converters
import colorsys
register_matplotlib_converters()

#%%
def rle(inarray):
    """ run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values)


    Returns 3 arrays:
        - The lengths of each run
        - The location of the start of each run
        - The values contained in each run
    """
    ia = np.asarray(inarray)                  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])      # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)    # must include last element position
        z = np.diff(np.append(-1, i))        # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return(z, p, ia[i])


def calculateDailyVectorStats(df):
    """returns a dataframe containing the daily vector stats of each day"""

    #if('date' not in list(df)):
    df['date'] = df['time'].str.split('T', expand=True)[0]

    data_types = set(df['type'].values)

    vector_list = ['date',
                   'contains_CGM',
                   'contains_bolus',
                   'contains_basal',
                   'contains_physicalActivity',
                   'contains_inpen',
                   'contains_food',
                   'contains_loop',
                   'CGM_percent_gte70',
                   'bolus_count_gte1',
                   'basal_temp_count_gte30',
                   'basal_suspend_count_gte3',
                   'isTandemPump',
                   'is_after_basalIQ_launch',
                   'is_basalIQ',
                   'is_hcl',
                   'is_sap',
                   'is_pa',
                   'is_inpen',
                   'is_inpen_with_food']

    date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='1d').strftime('%Y-%m-%d')
    vector_stats = pd.DataFrame([], columns=vector_list, index=date_range)
    vector_stats[vector_list] = False
    vector_stats['date'] = date_range

    if('cbg' in data_types):
        cgm_df = df[df['type']=='cbg']
        vector_stats['contains_CGM'] = vector_stats['date'].isin(cgm_df.date)
        vector_stats['CGM_percent_gte70'] = cgm_df.drop_duplicates(['time','value']).groupby('date').apply(lambda x: len(x)/288 >= 0.7)

    if('bolus' in data_types):
        bolus_df = df[df['type']=='bolus']
        vector_stats['contains_bolus'] = vector_stats['date'].isin(bolus_df.date)
        vector_stats['bolus_count_gte1'] = bolus_df.drop_duplicates(['time','normal']).groupby('date').apply(lambda x: len(x) >= 1)

    if('basal' in data_types):
        basal_df = df[df['type']=='basal']
        vector_stats['contains_basal'] = vector_stats['date'].isin(basal_df.date)
        temp_df = basal_df[basal_df['deliveryType']=='temp']
        if(len(temp_df) > 0):
            vector_stats['basal_temp_count_gte30'] = temp_df.drop_duplicates(['time','duration']).groupby('date').apply(lambda x: len(x) >= 30)
        else:
            vector_stats['basal_temp_count_gte30'] = False

    if('physicalActivity' in data_types):
        physicalActivity_df = df[df['type']=='physicalActivity']
        vector_stats['contains_physicalActivity'] = vector_stats['date'].isin(physicalActivity_df.date)

    if('origin' in list(df)):
        vector_stats['contains_inpen'] = vector_stats['date'].isin(df[df.origin.astype(str).str.contains('inpen')].date)
        vector_stats['contains_loop'] = vector_stats['date'].isin(df[df.origin.astype(str).str.contains('loop')].date)

    if( ('deviceId' in list(df)) & ('deliveryType' in list(df)) ):
        vector_stats['isTandemPump'] = vector_stats['date'].isin(df[df.deviceId.astype(str).str.contains('tandem')].date)
        vector_stats['is_after_basalIQ_launch'] = df.groupby('date').apply(lambda x: x['date'].astype(str).min() > '2018-08-01')
        suspend_df = df[(df['type']=='basal') & (df['deliveryType']=='suspend')]
        if(len(suspend_df) > 0):
            vector_stats['basal_suspend_count_gte3'] = suspend_df.drop_duplicates(['time','duration']).groupby('date').apply(lambda x: len(x) >= 3)
        else:
            vector_stats['basal_suspend_count_gte3'] = False

    if('food' in data_types):
        food_df = df[df['type']=='food']
        vector_stats['contains_food'] = vector_stats['date'].isin(food_df.date)

    vector_stats.fillna(False, inplace=True)
    vector_stats['is_basalIQ'] = vector_stats['contains_CGM'] & vector_stats['isTandemPump'] & vector_stats['is_after_basalIQ_launch'] & vector_stats['basal_suspend_count_gte3']
    first_basalIQ_day = vector_stats.loc[vector_stats['is_basalIQ'] == True, 'date'].min()
    last_tandem_day = vector_stats.loc[vector_stats['isTandemPump'] == True, 'date'].max()

    # Set the basalIQ to true for all data from a tandem pump
    # after the first basalIQ day is found

    vector_stats.loc[(vector_stats['date'] >= first_basalIQ_day) &
                     (vector_stats['date'] <= last_tandem_day),
                     'is_basalIQ'] = True


    vector_stats['is_hcl'] = vector_stats['CGM_percent_gte70'] & vector_stats['bolus_count_gte1'] & vector_stats['basal_temp_count_gte30'] & ~vector_stats['is_basalIQ']
    vector_stats['is_sap'] = vector_stats['CGM_percent_gte70'] & vector_stats['bolus_count_gte1'] & vector_stats['contains_basal'] & ~vector_stats['is_hcl'] & ~vector_stats['is_basalIQ']

    if(vector_stats['is_hcl'].any() & vector_stats['is_sap'].any()):
        # Check to see if SAP and HCL windows overlap
        first_HCL_day = vector_stats.loc[vector_stats['is_hcl'] == True, 'date'].min()
        last_HCL_day = vector_stats.loc[vector_stats['is_hcl'] == True, 'date'].max()
        first_SAP_day = vector_stats.loc[vector_stats['is_sap'] == True, 'date'].min()
        last_SAP_day = vector_stats.loc[vector_stats['is_sap'] == True, 'date'].max()

        latest_start = max(first_HCL_day, first_SAP_day)
        earliest_end = min(last_HCL_day, last_SAP_day)
        overlap = (pd.to_datetime(earliest_end) - pd.to_datetime(latest_start)).days + 1

        # If overlapping, set all SAP days after first HCL day to False

        # === Note ===
        # There are many days in HCL datasets when the closed-loop system is turned
        # off, or open-looping - which can create an SAP-qualifying day.
        # This creates stretches of SAP data segments with large gaps and hurts
        # datasets with good SAP data sections. The only downside is that this
        # may disqualify some datasets with good SAP data after an HCL segment.

        if overlap > 0:
            vector_stats.loc[(vector_stats['date'] >= first_HCL_day),
                             'is_sap'] = False

    vector_stats['is_inpen'] = vector_stats['contains_inpen'] & vector_stats['CGM_percent_gte70'] & vector_stats['bolus_count_gte1']
    vector_stats['is_inpen_with_food'] = vector_stats['is_inpen'] & vector_stats['contains_food']
    vector_stats['is_pa'] = vector_stats['contains_physicalActivity'] & (vector_stats['is_hcl'] | vector_stats['is_sap'] | vector_stats['is_basalIQ'] | vector_stats['is_inpen'])

    vector_stats.reset_index(drop=True, inplace=True)

    return vector_stats


def get_vector_summary(data_df, file_name):
    # print(file_loc)
    vector_stats = []
    firstDay = np.nan
    lastDay = np.nan
    daySpan = np.nan
    uniqueDays = np.nan

    cgm_days = np.nan
    cgm_daily_count_mean = np.nan

    basal_days = np.nan
    basal_duration_daily_mean = np.nan
    basal_daily_upload_mean = np.nan

    bolus_days = np.nan
    physicalActivity_days = np.nan
    loop_days = np.nan
    hcl_days = np.nan
    hcl_start = np.nan
    hcl_end = np.nan
    hcl_span_pct = np.nan
    hcl_top5_gaps = np.nan
    hcl_max_gap_pct = np.nan
    sap_days = np.nan
    sap_start = np.nan
    sap_end = np.nan
    sap_span_pct = np.nan
    sap_top5_gaps = np.nan
    sap_max_gap_pct = np.nan
    basalIQ_days = np.nan
    basalIQ_start = np.nan
    basalIQ_end = np.nan
    basalIQ_span_pct = np.nan
    basalIQ_top5_gaps = np.nan
    basalIQ_max_gap_pct = np.nan
    pa_days = np.nan
    pa_start = np.nan
    pa_end = np.nan
    pa_span_pct = np.nan
    pa_top5_gaps = np.nan
    pa_max_gap_pct = np.nan
    inpen_days = np.nan
    inpen_start = np.nan
    inpen_end = np.nan
    inpen_span_pct = np.nan
    inpen_top5_gaps = np.nan
    inpen_max_gap_pct = np.nan
    inpen_with_food_days = np.nan
    inpen_with_food_start = np.nan
    inpen_with_food_end = np.nan
    inpen_with_food_span_pct = np.nan
    inpen_with_food_top5_gaps = np.nan
    inpen_with_food_max_gap_pct = np.nan

    if(len(data_df) > 0):

        #if 'date' not in list(data_df):
        data_df['date'] = data_df['time'].str.split('T', expand=True)[0]
        firstDay = data_df['date'].min()
        lastDay = data_df['date'].max()
        daySpan = (pd.to_datetime(lastDay)-pd.to_datetime(firstDay)).days + 1
        uniqueDays = len(set(data_df['date'].values))

        if 'type' in list(data_df):
            vector_stats = calculateDailyVectorStats(data_df)

            cgm_days = vector_stats['contains_CGM'].sum()

            if(cgm_days > 0):
                cgm_df = data_df[data_df['type']=='cbg']
                cgm_daily_count_mean = cgm_df.groupby('date').apply(lambda x: x.value.count()).mean()

            basal_days = vector_stats['contains_basal'].sum()

            if(basal_days > 0):
                basal_df = data_df[data_df['type']=='basal']
                basal_duration_daily_mean = basal_df.duration.astype(float).sum()/1000/60/60/len(basal_df.date.unique())
                basal_daily_upload_mean = basal_df.groupby('date').apply(lambda x: len(x.uploadId.unique())).mean()

            bolus_days = vector_stats['contains_bolus'].sum()

            physicalActivity_days = vector_stats['contains_physicalActivity'].sum()

            loop_days = vector_stats['contains_loop'].sum()

            hcl_days = vector_stats['is_hcl'].sum()
            hcl_start = vector_stats.loc[vector_stats['is_hcl'] == True, 'date'].min()
            hcl_end = vector_stats.loc[vector_stats['is_hcl'] == True, 'date'].max()
            hcl_span = (pd.to_datetime(hcl_end)-pd.to_datetime(hcl_start)).days + 1
            hcl_span_pct = hcl_days/hcl_span

            hcl_rle_results = \
                rle(
                    vector_stats.loc[
                            (vector_stats['date'] >= hcl_start) &
                            (vector_stats['date'] <= hcl_end),
                            'is_hcl'
                            ]
                    )

            if hcl_rle_results[0] is not None:
                hcl_top5_gaps = -np.sort(-hcl_rle_results[0][~hcl_rle_results[2]])[:5]
                if len(hcl_top5_gaps) > 0:
                    hcl_max_gap_pct = hcl_top5_gaps.max()/hcl_span
                else:
                    hcl_max_gap_pct = 0

            sap_days = vector_stats['is_sap'].sum()
            sap_start = vector_stats.loc[vector_stats['is_sap'] == True, 'date'].min()
            sap_end = vector_stats.loc[vector_stats['is_sap'] == True, 'date'].max()
            sap_span = (pd.to_datetime(sap_end)-pd.to_datetime(sap_start)).days + 1
            sap_span_pct = sap_days/sap_span
            sap_rle_results = \
                rle(
                    vector_stats.loc[
                            (vector_stats['date'] >= sap_start) &
                            (vector_stats['date'] <= sap_end),
                            'is_sap'
                            ]
                    )
            if sap_rle_results[0] is not None:
                sap_top5_gaps = -np.sort(-sap_rle_results[0][~sap_rle_results[2]])[:5]
                if len(sap_top5_gaps) > 0:
                    sap_max_gap_pct = sap_top5_gaps.max()/sap_span
                else:
                    sap_max_gap_pct = 0

            basalIQ_days = vector_stats['is_basalIQ'].sum()
            basalIQ_start = vector_stats.loc[vector_stats['is_basalIQ'] == True, 'date'].min()
            basalIQ_end = vector_stats.loc[vector_stats['is_basalIQ'] == True, 'date'].max()
            basalIQ_span = (pd.to_datetime(basalIQ_end)-pd.to_datetime(basalIQ_start)).days + 1
            basalIQ_span_pct = basalIQ_days/basalIQ_span
            basalIQ_rle_results = \
                rle(
                    vector_stats.loc[
                            (vector_stats['date'] >= basalIQ_start) &
                            (vector_stats['date'] <= basalIQ_end),
                            'is_basalIQ'
                            ]
                    )
            if basalIQ_rle_results[0] is not None:
                basalIQ_top5_gaps = -np.sort(-basalIQ_rle_results[0][~basalIQ_rle_results[2]])[:5]
                if len(basalIQ_top5_gaps) > 0:
                    basalIQ_max_gap_pct = basalIQ_top5_gaps.max()/basalIQ_span
                else:
                    basalIQ_max_gap_pct = 0

            inpen_days = vector_stats['is_inpen'].sum()
            inpen_start = vector_stats.loc[vector_stats['is_inpen'] == True, 'date'].min()
            inpen_end = vector_stats.loc[vector_stats['is_inpen'] == True, 'date'].max()
            inpen_span = (pd.to_datetime(inpen_end)-pd.to_datetime(inpen_start)).days + 1
            inpen_span_pct = inpen_days/inpen_span
            inpen_rle_results = \
                rle(
                    vector_stats.loc[
                            (vector_stats['date'] >= inpen_start) &
                            (vector_stats['date'] <= inpen_end),
                            'is_inpen'
                            ]
                    )
            if inpen_rle_results[0] is not None:
                inpen_top5_gaps = -np.sort(-inpen_rle_results[0][~inpen_rle_results[2]])[:5]
                if len(inpen_top5_gaps) > 0:
                    inpen_max_gap_pct = inpen_top5_gaps.max()/inpen_span
                else:
                    inpen_max_gap_pct = 0

            inpen_with_food_days = vector_stats['is_inpen_with_food'].sum()
            inpen_with_food_start = vector_stats.loc[vector_stats['is_inpen_with_food'] == True, 'date'].min()
            inpen_with_food_end = vector_stats.loc[vector_stats['is_inpen_with_food'] == True, 'date'].max()
            inpen_with_food_span = (pd.to_datetime(inpen_with_food_end)-pd.to_datetime(inpen_with_food_start)).days + 1
            inpen_with_food_span_pct = inpen_with_food_days/inpen_with_food_span
            inpen_with_food_rle_results = \
                rle(
                    vector_stats.loc[
                            (vector_stats['date'] >= inpen_with_food_start) &
                            (vector_stats['date'] <= inpen_with_food_end),
                            'is_inpen'
                            ]
                    )
            if inpen_with_food_rle_results[0] is not None:
                inpen_with_food_top5_gaps = -np.sort(-inpen_with_food_rle_results[0][~inpen_with_food_rle_results[2]])[:5]
                if len(inpen_with_food_top5_gaps) > 0:
                    inpen_with_food_max_gap_pct = inpen_with_food_top5_gaps.max()/inpen_with_food_span
                else:
                    inpen_with_food_max_gap_pct = 0

            pa_days = vector_stats['is_pa'].sum()
            pa_start = vector_stats.loc[vector_stats['is_pa'] == True, 'date'].min()
            pa_end = vector_stats.loc[vector_stats['is_pa'] == True, 'date'].max()
            pa_span = (pd.to_datetime(pa_end)-pd.to_datetime(pa_start)).days + 1
            pa_span_pct = pa_days/pa_span
            pa_rle_results = \
                rle(
                    vector_stats.loc[
                            (vector_stats['date'] >= pa_start) &
                            (vector_stats['date'] <= pa_end),
                            'is_pa'
                            ]
                    )
            if pa_rle_results[0] is not None:
                pa_top5_gaps = -np.sort(-pa_rle_results[0][~pa_rle_results[2]])[:5]
                if len(pa_top5_gaps) > 0:
                    pa_max_gap_pct = pa_top5_gaps.max()/pa_span
                else:
                    pa_max_gap_pct = 0

    results = [file_name,
               firstDay,
               lastDay,
               daySpan,
               uniqueDays,
               cgm_days,
               cgm_daily_count_mean,
               basal_days,
               basal_duration_daily_mean,
               basal_daily_upload_mean,
               bolus_days,
               physicalActivity_days,
               loop_days,
               hcl_days,
               hcl_start,
               hcl_end,
               hcl_span_pct,
               hcl_top5_gaps,
               hcl_max_gap_pct,
               sap_days,
               sap_start,
               sap_end,
               sap_span_pct,
               sap_top5_gaps,
               sap_max_gap_pct,
               basalIQ_days,
               basalIQ_start,
               basalIQ_end,
               basalIQ_span_pct,
               basalIQ_top5_gaps,
               basalIQ_max_gap_pct,
               pa_days,
               pa_start,
               pa_end,
               pa_span_pct,
               pa_top5_gaps,
               pa_max_gap_pct,
               inpen_days,
               inpen_start,
               inpen_end,
               inpen_span_pct,
               inpen_top5_gaps,
               inpen_max_gap_pct,
               inpen_with_food_days,
               inpen_with_food_start,
               inpen_with_food_end,
               inpen_with_food_span_pct,
               inpen_with_food_top5_gaps,
               inpen_with_food_max_gap_pct
               ]

    vector_summary = pd.DataFrame(results).T

    column_names = ['file_name',
                    'firstDay',
                    'lastDay',
                    'daySpan',
                    'uniqueDays',
                    'cgm_days',
                    'cgm_daily_count_mean',
                    'basal_days',
                    'basal_duration_daily_mean',
                    'basal_daily_upload_mean',
                    'bolus_days',
                    'physicalActivity_days',
                    'loop_days',
                    'hcl_days',
                    'hcl_start',
                    'hcl_end',
                    'hcl_span_pct',
                    'hcl_top5_gaps',
                    'hcl_max_gap_pct',
                    'sap_days',
                    'sap_start',
                    'sap_end',
                    'sap_span_pct',
                    'sap_top5_gaps',
                    'sap_max_gap_pct',
                    'basalIQ_days',
                    'basalIQ_start',
                    'basalIQ_end',
                    'basalIQ_span_pct',
                    'basalIQ_top5_gaps',
                    'basalIQ_max_gap_pct',
                    'pa_days',
                    'pa_start',
                    'pa_end',
                    'pa_span_pct',
                    'pa_top5_gaps',
                    'pa_max_gap_pct',
                    'inpen_days',
                    'inpen_start',
                    'inpen_end',
                    'inpen_span_pct',
                    'inpen_top5_gaps',
                    'inpen_max_gap_pct',
                    'inpen_with_food_days',
                    'inpen_with_food_start',
                    'inpen_with_food_end',
                    'inpen_with_food_span_pct',
                    'inpen_with_food_top5_gaps',
                    'inpen_with_food_max_gap_pct'
                    ]

    vector_summary.columns = column_names

    return vector_stats, vector_summary


def get_colors(n):
    colors = []

    for x in range(n):
        # colors.append(tuple(np.random.randint(256, size=3)))
        hue = (1/n) + (1/n)*x
        # colors.append(tuple([val, 255, 255]))
        (r, g, b) = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        colors.append(tuple([R, G, B]))

    hex_out = []
    for rgb in colors:
        hex_out.append('#%02x%02x%02x' % rgb)

    return hex_out


"""
def get_colors(N):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out
"""


def createMatplotlibPlot(vector_stats, export_location, file_name, yaxis_names=np.nan):
    """Creates the combined daily data type plot"""

    # Limit vector_stats to just the cgm+pump range
    #start_date = vector_stats.loc[vector_stats['cgm+pump'] == True, 'date'].min()
    #end_date = vector_stats.loc[vector_stats['cgm+pump'] == True, 'date'].max()

    #vector_stats = vector_stats[(vector_stats['date'] >= start_date) & (vector_stats['date'] <= end_date)].copy()

    if type(yaxis_names) == float:
        yaxis_names = list(vector_stats)[1:]

    #converts daily boolean datatypes to y-axis values for plotting
    yaxis_values = np.arange(len(yaxis_names)) + 1
    vector_stats[yaxis_names] = vector_stats[yaxis_names] * yaxis_values
    vector_stats[yaxis_names] = vector_stats[yaxis_names].replace(0, np.nan)

    vector_stats['date'] = pd.to_datetime(vector_stats['date'])

    fig = plt.figure()
    ax  = fig.add_subplot(111)

    colors = get_colors(len(yaxis_names))

    for label_loc in range(len(yaxis_names)):
        yaxis_label = yaxis_names[label_loc]
        #ax.scatter(vector_stats['date'],
        #    vector_stats[yaxis_label],
        #    marker='|' ,
        #    label=yaxis_label,
        #    linewidth=2,
        #    s=1)

        ax.vlines(
            x=vector_stats.loc[vector_stats[yaxis_label].notnull(), 'date'],
            ymin=vector_stats[yaxis_label].min()-0.4,
            ymax=vector_stats[yaxis_label].max()+0.4,
            label=yaxis_label,
            color=colors[label_loc],
            linewidth=1
            )

    plt.xlim(pd.to_datetime('2014-01-01'),
         pd.to_datetime(dt.datetime.today().date())
         )
    plt.yticks(yaxis_values, yaxis_names)
    plt.xticks(rotation=90)
    plt.title(file_name[0:20], size=20, y=1.01)
    #leg = plt.legend()
    plt.tight_layout()
    #plt.autoscale()
    plt.subplots_adjust(hspace = 0.3, top = 2, right = 2)
    #plt.show()

    #today_timestamp = dt.datetime.now().strftime("%Y-%m-%d")
    fig.set_size_inches(12, 5)
    plt.savefig(export_location + file_name + ".png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def vector_stats_viz(data, export_location, file_name):

    vector_stats, vector_summary = get_vector_summary(data, file_name)
    createMatplotlibPlot(vector_stats.copy(), export_location, file_name)
