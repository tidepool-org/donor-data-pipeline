#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Tidepool Device Data Summary
======================================
:File: dataset_summary_viz.py
:Description: Visualizes the timeseries of data types and metrics available
:Version: 0.0.1
:Created: 2019-09-22
:Authors: Jason Meno (jam)
:Last Modified: 2019-09-22 (jam)
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
import datetime as dt
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import os
import time
import colorsys
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
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


def calculateDailyTypes(df):
    """returns a dataframe containing the data types of each day"""

    if('date' not in list(df)):
        df['date'] = df['time'].str.split('T', expand=True)[0]

    typesPerDay = pd.DataFrame(
            df.groupby('date').apply(lambda x: np.unique(list(x['type']))),
                    columns=['type_list']
                    ).reset_index()

    data_types = set(df['type'].values)
    all_data_types = set(['cgm+pump',
                      'closedLoop',
                      'physicalActivity',
                      'cgmSettings',
                      'pumpSettings',
                      'deviceEvent',
                      'upload',
                      'smbg',
                      'food',
                      'insulin',
                      'bolus',
                      'basal',
                      'cbg'])

    for data_type in all_data_types:
        typesPerDay[data_type] = \
            typesPerDay['type_list'].apply(lambda x: data_type in x)

    if ('basal' in data_types):
        typesPerDay['closedLoop'] = df.groupby('date').apply(lambda x: (x['deliveryType']=='temp').sum() >= 30).values
        #typesPerDay['basal_deviceTime'] = df.groupby('date').apply(lambda x: any(x.loc[x['type'] == 'basal','deviceTime'])).values

    #if ('bolus' in data_types):
    #    typesPerDay['bolus_deviceTime'] = df.groupby('date').apply(lambda x: any(x.loc[x['type'] == 'bolus','deviceTime'])).values

    #if ('cbg' in data_types):
    #    typesPerDay['cbg_deviceTime'] = df.groupby('date').apply(lambda x: any(x.loc[x['type'] == 'cbg','deviceTime'])).values

    if (('cbg' in data_types) & ('bolus' in data_types) & ('basal' in data_types)):
        typesPerDay['cgm+pump'] = typesPerDay['cbg'] & typesPerDay['bolus'] & typesPerDay['basal']
     #   typesPerDay['cgm+pump_deviceTime'] = typesPerDay['cbg_deviceTime'] & typesPerDay['bolus_deviceTime'] & typesPerDay['basal_deviceTime']

    # HealthKit Data
    if 'origin' in list(df):
        typesPerDay['healthkit'] = df.groupby('date').apply(lambda x: any(x['origin'].str.contains("HealthKit") == True)).values
        typesPerDay['Loop'] = df.groupby('date').apply(lambda x: any(x['origin'].str.contains("Loop") == True)).values

    return typesPerDay

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

def createMatplotlibPlot(typesPerDay, export_location, file_name, yaxis_names=np.nan):
    """Creates the combined daily data type plot"""

    # Limit typesPerDay to just the cgm+pump range
    # start_date = typesPerDay.loc[typesPerDay['cgm+pump'] == True, 'date'].min()
    # end_date = typesPerDay.loc[typesPerDay['cgm+pump'] == True, 'date'].max()
    start_date = typesPerDay.date.min()
    end_date = typesPerDay.date.max()

    typesPerDay = typesPerDay[(typesPerDay['date'] >= start_date) & (typesPerDay['date'] <= end_date)].copy()

    if type(yaxis_names) == float:
        yaxis_names = list(typesPerDay)[2:]

    #converts daily boolean datatypes to y-axis values for plotting
    yaxis_values = np.arange(len(yaxis_names)) + 1
    typesPerDay[yaxis_names] = typesPerDay[yaxis_names] * yaxis_values
    typesPerDay[yaxis_names] = typesPerDay[yaxis_names].replace(0, np.nan)

    typesPerDay['date'] = pd.to_datetime(typesPerDay['date'])

    fig = plt.figure()
    ax  = fig.add_subplot(111)

    colors = get_colors(len(yaxis_names))

    for label_loc in range(len(yaxis_names)):
        yaxis_label = yaxis_names[label_loc]
        #ax.scatter(typesPerDay['date'],
        #    typesPerDay[yaxis_label],
        #    marker='|' ,
        #    label=yaxis_label,
        #    linewidth=2,
        #    s=1)

        ax.vlines(
            x=typesPerDay.loc[typesPerDay[yaxis_label].notnull(), 'date'],
            ymin=typesPerDay[yaxis_label].min()-0.4,
            ymax=typesPerDay[yaxis_label].max()+0.4,
            label=yaxis_label,
            color=colors[label_loc]
            #linewidth=1
            )

    plt.xlim(pd.to_datetime('2014-01-01'),
             pd.to_datetime(dt.datetime.today().date())
             )
    plt.yticks(yaxis_values, yaxis_names)
    plt.xticks(rotation=90)
    plt.title(file_name[0:20], size=20, y=1.01)
    #leg = plt.legend()
    #plt.tight_layout()
    #plt.autoscale()
    #plt.subplots_adjust(hspace = 0.3, top = 2, right = 2)
    #plt.show()

    #today_timestamp = dt.datetime.now().strftime("%Y-%m-%d")
    fig.set_size_inches(12, 5)
    plt.savefig(export_location + file_name + ".png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def createPlotlyPlot(typesPerDay, yaxis_names=np.nan):
    """Creates the combined daily data type plot"""

    if type(yaxis_names) == float:
        yaxis_names = list(typesPerDay)[2:]

    #converts daily boolean datatypes to y-axis values for plotting
    yaxis_values = np.arange(len(yaxis_names)) + 1
    typesPerDay[yaxis_names] = typesPerDay[yaxis_names] * yaxis_values
    typesPerDay[yaxis_names] = typesPerDay[yaxis_names].replace(0, np.nan)

    typesPerDay['date'] = pd.to_datetime(typesPerDay['date'])

    fig = go.Figure()

    for yaxis_label in yaxis_names:
        trace = go.Scattergl(x=typesPerDay['date'],
                           y=typesPerDay[yaxis_label],
                           mode='markers',
                           name=yaxis_label,
                           marker=dict(size=20)
                           )

        fig.add_trace(trace)

    fig.layout = go.Layout(
            yaxis = go.layout.YAxis(
                    tickmode = 'array',
                    tickvals = yaxis_values,
                    ticktext = yaxis_names
                ),
            legend = dict(traceorder='reversed'),
            template = "plotly_white"
        )

    plot(fig)

def data_summary_viz(data, export_location, file_name):

    typesPerDay = calculateDailyTypes(data)
    createMatplotlibPlot(typesPerDay, export_location, file_name)

