#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Tidepool Device Data Type Time Offset Summary
=======================================================
:File: dataset_time_summary_viz.py
:Description: Visualizes the timezone offsets of the different
timeseries of data types and metrics available
:Version: 0.0.1
:Created: 2019-09-27
:Authors: Jason Meno (jam)
:Last Modified: 2019-09-27 (jam)
:Dependencies:
    - A Tidepool dataset to qualify
:License: BSD-2-Clause
"""
# %% Import Dependencies
import pandas as pd
import numpy as np

import matplotlib
# Hide Figures with Agg backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib import cm
import plotly.graph_objs as go
import datetime as dt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import os
import time
from pandas.plotting import register_matplotlib_converters
import colorsys
register_matplotlib_converters()

# %% Functions
def importData(file_path):
    """imports csv data"""

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except:
        df = pd.DataFrame()
        print("Failed to import: " + file_path)

    return df

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

def get_daily_offset_mode(offset_df):

    offset_mode = offset_df.mode()

    if len(offset_mode) > 0:
        offset_mode = offset_mode[0]
    else:
        offset_mode = np.nan

    return offset_mode

def calculateTimeOffsets(df):
    """returns a dataframe containing the time offsets of the deviceTime and
    est.localTime of each data type in the entire time series"""
    if('date' not in list(df)):
        df['date'] = df['time'].str.split('T', expand=True)[0]

    if('deviceTime' not in list(df)):
        df['deviceTime'] = np.nan

    original_time = df['time']
    original_localTime = df['est.localTime']
    original_deviceTime = df['deviceTime']
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['deviceTime'] = pd.to_datetime(df['deviceTime'])#, utc=True)
    df['est.localTime'] = pd.to_datetime(df['est.localTime'])#, utc=True)

    df['time'] = df['time'].dt.tz_localize(None)
    #df['deviceTime'] = df['deviceTime'].dt.tz_localize(None)
    #df['est.localTime'] = df['est.localTime'].dt.tz_localize(None)

    df['deviceTimeOffset'] = round((df['deviceTime'] - df['time']).dt.total_seconds()/60/60,1)
    #df['deviceTimeOffset'] = \
    #    round(
    #            df['deviceTimeOffset'].dt.days*24 + \
    #            df['deviceTimeOffset'].dt.seconds/60/60,
    #            1
    #        )
    df['deviceTimeOffset'].fillna(-1000, inplace=True)
    df['est.localTimeOffset'] = round((df['est.localTime'] - df['time']).dt.total_seconds()/60/60, 1)
    #df['est.localTimeOffset'] = \
    #    round(
    #        df['est.localTimeOffset'].dt.days*24 + \
    #        df['est.localTimeOffset'].dt.seconds/60/60,
    #        1
    #        )
    df['est.localTimeOffset'].fillna(-1000, inplace=True)
    # Return times to original format
    df['time'] = original_time
    df['est.localTime'] = original_localTime
    df['deviceTime'] = original_deviceTime
    #timeOffsetByType = pd.DataFrame(df['time'].unique(), columns=['time'])

    data_types = set(df['type'].values)

    timeOffsetByType = pd.DataFrame(
            df.groupby('date').apply(lambda x: np.unique(list(x['type']))),
                    columns=['type_list']
                    ).reset_index()


    if(df['type'].notnull().sum() > 0):

        for data_type in data_types:
            timeOffsetByType[data_type] = \
                timeOffsetByType['type_list'].apply(lambda x: data_type in x)

            timeOffsetByType[data_type + "_deviceTimeOffset"] = \
                df.groupby('date').apply(lambda x: get_daily_offset_mode(x.loc[x['type']==data_type, 'deviceTimeOffset'])).values

            timeOffsetByType[data_type + "_deviceTimeOffset"].fillna(-1000, inplace=True)

            timeOffsetByType[data_type + "_est.localTimeOffset"] = \
                df.groupby('date').apply(lambda x: get_daily_offset_mode(x.loc[x['type']==data_type, 'est.localTimeOffset'])).values

            timeOffsetByType[data_type + "_est.localTimeOffset"].fillna(-1000, inplace=True)

    return timeOffsetByType

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


def createMatplotlibPlot(timeOffsetByType, export_location, file_name, yaxis_names=np.nan):
    """Creates the combined daily data type plot"""

    # Limit typesPerDay to just the cgm+pump range
    # start_date = typesPerDay.loc[typesPerDay['cgm+pump'] == True, 'date'].min()
    # end_date = typesPerDay.loc[typesPerDay['cgm+pump'] == True, 'date'].max()

    # typesPerDay = typesPerDay[(typesPerDay['date'] >= start_date) & (typesPerDay['date'] <= end_date)].copy()

    #converts daily boolean datatypes to y-axis values for plotting
    yaxis_columns = list(timeOffsetByType)[2:len(timeOffsetByType):3]
    deviceTime_columns = list(timeOffsetByType)[3:len(timeOffsetByType):3]
    localTime_columns = list(timeOffsetByType)[4:len(timeOffsetByType):3]
    color_columns = deviceTime_columns + localTime_columns

    color_values = []
    for color_col in range(len(yaxis_columns)):
        deviceTime_col = deviceTime_columns[color_col]
        localTime_col = localTime_columns[color_col]

        color_values = \
            color_values + \
            list(timeOffsetByType[deviceTime_col].unique()) + \
            list(timeOffsetByType[localTime_col].unique())

    color_values = np.sort(list(pd.Series(color_values).unique()))
    #colors = get_colors(len(color_values))
    if(color_values[0] == -1000):
        colors = ["#000000"] + get_colors(len(color_values)-1)
        timeOffsetByType[color_columns] = timeOffsetByType[color_columns].replace(color_values, colors)
        color_values[0] = "NaN"
    else:
        colors = get_colors(len(color_values))
        timeOffsetByType[color_columns] = timeOffsetByType[color_columns].replace(color_values, colors)

    patches = []

    for color_idx in range(len(colors)):
        #color_name = color_values
        patches.append(mpatches.Patch(color=colors[color_idx], label=color_values[color_idx]))
    #converts daily boolean datatypes to y-axis values for plotting
    yaxis_values = np.arange(1, len(yaxis_columns)*2, 2)
    timeOffsetByType[yaxis_columns] = timeOffsetByType[yaxis_columns] * yaxis_values
    timeOffsetByType[yaxis_columns] = timeOffsetByType[yaxis_columns].replace(0, np.nan)

    #timeOffsetByType['time'] = pd.to_datetime(timeOffsetByType['time'])
    timeOffsetByType['date'] = pd.to_datetime(timeOffsetByType['date'])

    fig = plt.figure()
    ax  = fig.add_subplot(111)

    for data_loc in range(len(yaxis_columns)):
        col_name = yaxis_columns[data_loc]
        deviceTime_colors = timeOffsetByType[deviceTime_columns[data_loc]]
        localTime_colors = timeOffsetByType[localTime_columns[data_loc]]
        """
        ax.scatter(
                timeOffsetByType['date'],
                timeOffsetByType[col_name],
                label=col_name+'_deviceTime',
                marker='|',
                s=15,
                c=deviceTime_colors)

        ax.scatter(
                timeOffsetByType['date'],
                timeOffsetByType[col_name]+1,
                label=col_name+'_localTime',
                marker='|',
                edgecolors='red',
                s=15,
                c=localTime_colors)
        """
        ax.vlines(
            x=timeOffsetByType.loc[timeOffsetByType[col_name].notnull(), 'date'],
            ymin=timeOffsetByType[col_name].min()-0.4,
            ymax=timeOffsetByType[col_name].max()+0.4,
            label=col_name+'_deviceTime',
            color=deviceTime_colors[timeOffsetByType[col_name].notnull()]
            #linewidth=1
            )

        ax.vlines(
            x=timeOffsetByType.loc[timeOffsetByType[col_name].notnull(), 'date'],
            ymin=timeOffsetByType[col_name].min()+0.6,
            ymax=timeOffsetByType[col_name].max()+1.4,
            label=col_name+'_est.localTime',
            color=localTime_colors[timeOffsetByType[col_name].notnull()]
            #linewidth=1
            )

    deviceTime_yaxis_names = list(pd.Series(yaxis_columns) + '_deviceTime')
    localTime_yaxis_names = list(pd.Series(yaxis_columns) + '_est.localTime')

    yaxis_names = []

    for idx in range(len(deviceTime_yaxis_names)):
        yaxis_names.append(deviceTime_yaxis_names[idx])
        yaxis_names.append(localTime_yaxis_names[idx])

    plt.xlim(pd.to_datetime('2014-01-01'),
             pd.to_datetime(dt.datetime.today().date())
             )

    plt.yticks(np.arange(1, len(yaxis_columns)*2+1, 1), yaxis_names)
    plt.xticks(rotation=90)
    plt.title(file_name[0:20], size=20, y=1.01)
    #leg = plt.legend()
    plt.legend(handles=patches, bbox_to_anchor=(1.26, 1))
    plt.tight_layout()
    #plt.autoscale()
    plt.subplots_adjust(hspace = 0.3, top = 2, right = 2)
    #plt.show()

    #today_timestamp = dt.datetime.now().strftime("%Y-%m-%d")
    fig.set_size_inches(12, 5)
    plt.savefig(export_location + file_name + ".png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def createPlotlyPlot(timeOffsetByType):
    """Creates the combined daily data type plot"""

    yaxis_columns = list(timeOffsetByType)[2:len(timeOffsetByType):2]
    color_columns = list(timeOffsetByType)[1:len(timeOffsetByType):2]

    #converts daily boolean datatypes to y-axis values for plotting
    yaxis_values = np.arange(len(yaxis_columns)) + 1
    timeOffsetByType[yaxis_columns] = timeOffsetByType[yaxis_columns] * yaxis_values
    timeOffsetByType[yaxis_columns] = timeOffsetByType[yaxis_columns].replace(0, np.nan)

    timeOffsetByType['time'] = pd.to_datetime(timeOffsetByType['time'])

    fig = go.Figure()

    for data_loc in range(len(yaxis_columns)):
        col_name = yaxis_columns[data_loc]
        color_col_name = color_columns[data_loc]

        trace = go.Scattergl(
                    #x=timeOffsetByType['time'],
                    x=np.arange(len(timeOffsetByType)),
                    y=timeOffsetByType[col_name],
                    mode='markers',
                    name=col_name,
                       marker=dict(
                           #color=timeOffsetByType[color_col_name],
                           symbol='line-ns-open',
                           size=10,
                           line=dict(width=3)
                           )
                   )

        fig.add_trace(trace)

    fig.layout = go.Layout(
            yaxis = go.layout.YAxis(
                    tickmode = 'array',
                    tickvals = yaxis_values,
                    ticktext = yaxis_columns
                ),
            legend = dict(traceorder='reversed'),
            template = "plotly_white"
        )

    plot(fig, filename='test_localTime_viewer.html')

def local_time_summary_viz(data, export_location, file_name):

    timeOffsetByType = calculateTimeOffsets(data)
    createMatplotlibPlot(timeOffsetByType, export_location, file_name)
