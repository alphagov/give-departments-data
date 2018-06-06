# coding: utf-8
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler #for matplotlib colors

"""
PLotting functions for use in EDA
"""  
__author__ = "Ellie King"
__copyright__ = "Government Digital Service, 29/05/2108"


def group_plot_time_metric(df, metric, aggregation='sum'):
    if aggregation=='mean':
        grouped = df.groupby([df.index, pd.Grouper(freq='D')])[metric].mean() #resample operation for each day in datime index, sum the metric
        grouped.index = grouped.index.droplevel()
        ax = grouped.plot(figsize=(10, 10))
        ax.set_ylabel(metric)
        ax.set_xlabel('Date')
    else:
        grouped = df.groupby([df.index, pd.Grouper(freq='D')])[metric].sum() #resample operation for each day in datime index, sum the metric
        grouped.index = grouped.index.droplevel()
        ax = grouped.plot(figsize=(10, 10))
        ax.set_ylabel(metric)
        ax.set_xlabel('Date')
    

    return ax

def plot_time_metric(grouped_var):
    ax = grouped_var.plot(figsize=(10, 10))
    ax.set_ylabel(grouped_var.name)
    ax.set_xlabel('Date')
    return ax

def plot_time_metric_byvar(df, metric, byvar):
    grouped = df.groupby([byvar, pd.Grouper(freq='D')])[metric].sum()
    by_day = grouped.unstack(byvar, fill_value=0)
    top = by_day.iloc[:, by_day.columns.isin(by_day.min().sort_values(ascending=False)[:10].index)]
    bottom = by_day.iloc[:, by_day.columns.isin(by_day.min().sort_values()[:10].index)]
    
    ax = top.plot()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel(metric)
    ax.set_xlabel('Date')
    ax.set_title('Top 10 {}s for {}'.format(byvar, metric))
    
    ay = bottom.plot()
    ay.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ay.set_ylabel(metric)
    ay.set_xlabel('Date')
    ay.set_title('Bottom 10 {}s for {}'.format(byvar, metric))

    return ax, ay

def scatter_byvar(df, x, y, byvar, ylog=False, xlog=False):
    groups = df.groupby(byvar)

    # Plot
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(groups)))

    fig, ax = plt.subplots()
    ax.set_prop_cycle(cycler('color', colors))
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    
    ax.set_xlabel(x)
    ax.set_title('{} and {} by {}'.format(x, y, byvar))
    
    if ylog==True and xlog==False:
        for name, group in groups:
            ax.plot(group[x], np.log(group[y]), marker='o', linestyle='',  label=name, alpha=0.5)
            ax.set_ylabel('log({})'.format(y))
            
    elif ylog==True and xlog==True:
        for name, group in groups:
            ax.plot(np.log(group[x]), np.log(group[y]), marker='o', linestyle='',  label=name, alpha=0.5)
            ax.set_ylabel('log({})'.format(y))
            ax.set_xlabel('log({})'.format(x))
    
    elif ylog==False and xlog==True:
        for name, group in groups:
            ax.plot(np.log(group[x]), group[y], marker='o', linestyle='',  label=name, alpha=0.5)
            ax.set_xlabel('log({})'.format(x))
    else:
        for name, group in groups:
            ax.plot(group[x], group[y], marker='o', linestyle='',  label=name, alpha=0.5 )
            ax.set_ylabel(y)
            ax.set_xlabel(x)
            
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax



