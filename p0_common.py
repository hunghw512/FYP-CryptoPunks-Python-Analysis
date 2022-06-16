import pandas as pd
import numpy as np
import itertools
import os
from datetime import datetime
from dateutil import relativedelta


def run_get_dataframe_name(df_file):
    name =[x for x in globals() if globals()[x] is df_file][0]
    return name#???

def read_csv(file_loc, file_name):
    file_name = pd.read_csv(os.path.join(file_loc, file_name + ".csv"))
    return file_name

def run_remove_anomalies(file, column_num, percentile):
    p1_threshold = np.percentile(file.iloc[:,column_num], percentile)      #2 stdev (5%) for anomalies
    p2_threshold = np.percentile(file.iloc[:,column_num], 100-percentile)  #2 stdev (5%) for anomalies
    anomalies = file[(file.iloc[:,column_num] < p1_threshold) | (file.iloc[:,column_num] > p2_threshold)]
    normalies = file[(file.iloc[:,column_num] >= p1_threshold) & (file.iloc[:,column_num] <= p2_threshold)]
    return anomalies, normalies

def run_label_percentile(df, bins, label, df_col):
    #percentile_bins = np.linspace(0,100,bin+1)     # bins for distribution analysis - 11 numbers for 10 bin distributed between 0 and 100 percentile
    percentile_bins = bins
    percentile_cutoffs = np.percentile(df.iloc[:,df_col], percentile_bins)
    print("\npercentile_bins:", percentile_bins, "\npercentile_cutoffs", percentile_cutoffs)

    blank = []
    num = range(len(percentile_bins))
    for row in df.iloc[:,df_col]:
        for l, n in zip(label, num):
            if percentile_cutoffs[n] <= row < percentile_cutoffs[n+1]: blank.append(l)
        if row == percentile_cutoffs[-1]: blank.append(label[-1])
    df['label'] = blank

    for l in label:
        num_rows = len(df.loc[df['label'] == l].index)
        print("Number of items in label {}: {}".format(l, num_rows))
    return df
    
    '''#Note: this is to get label for a specific column.
       Parameter: (1) df => input dataframe, type: <class 'pandas.core.frame.DataFrame'>
                  (2) bins => desired percentiles, example: bins = [0, 90, 100], this creates three cutoffs (0, 90%, 100%)
                  (3) label => desired labels, type: <class 'list'>, example: label = ["0-90", "90-100"]
                  (4) df_col => the column index, type: <class 'int'>'''

def read_file_run_label(file_loc, file_name, bins, label, df_col):      
    #read csv
    file_name = pd.read_csv(os.path.join(file_loc, file_name + ".csv"))
    #get label
    df2 = run_label_percentile(file_name, bins, label, df_col)
    #print(df2, "end")
    return df2
    '''#Note: this is to read a file, and get label for a specific column.
              the difference between read_file_run_label and run_label is reading in csv file.
       Parameter: (1) file_loc => "this_is_the_folder" or ""(if no folder)
                  (2) file_name => "this_is_the_csv_name"
                  (3) bins => desired percentiles, example: bins = [0, 90, 100], this creates three cutoffs (0, 90%, 100%)
                  (4) label => desired labels, type: <class 'list'>, example: label = ["0-90", "90-100"]
                  (5) df_col => the column index, type: <class 'int'>                     
       #Example of use: file_loc = "__filtered_result__"
                        bins = [0, 90, 100]
                        label = ["0-90", "90-100"]
                        df_example = read_file_run_label(file_loc, "this_is_the_csv_name", bins, label, 1)'''

def run_label_exact_value(input, bins, label, df_col):
    print("\ntransaction_cutoffs:", bins)

    blank = []
    num = range(len(bins))
    for row in input.iloc[:,df_col]:
        for l, n in zip(label, num):
            if bins[n] <= row < bins[n+1]: blank.append(l)
        if row == bins[-1]: blank.append(label[-1])
    input['label'] = blank

    for l in label:
        num_rows = len(input.loc[input['label'] == l].index)
        print("Number of items in label {}: {}".format(l, num_rows))
    return input

def run_holding_period(df, group_by_col, datetime_col): #more simplified function to cal holding period
    df = df.sort_values([group_by_col, datetime_col], ascending=[True, False])
    blk_time = pd.to_datetime(df[datetime_col])
    df['previous'] = blk_time.shift(periods=-1, fill_value=0)
    df['holding_period'] = (blk_time-df['previous'])/np.timedelta64(1, 'M')
    #df['holding_period'] = df['holding_period'].astype(int)
    df = df.drop_duplicates(subset=[group_by_col], keep='first')
    output = df.reset_index(drop=True)
    return output
    #Question: when import this function to other py, why the outcome of holding period will turn into int (without any decimal)??
    #Alright, sometime it works; sometime not...