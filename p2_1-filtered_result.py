import os
import pandas as pd
import numpy as np
from p0_common import run_remove_anomalies
from p1_data_cleaning import op_clean
def run_get_dataframe_name(df_file):
    name =[x for x in globals() if globals()[x] is df_file][0]
    return name
'''------------------------------------------------------------Content------------------------------------------------------------------------------------------------------------------------------------
a: filter out txn count
b: generate destinated dataframes
c: perform pct change and diff
d: remove_anomalies'''

####-----------------------------------------------------------filter out txn count------------------------------------------------------------------------------------------------
#filter out txn count <= 1
n_transaction_threshold = 1
op_clean = op_clean.groupby('punk_id').filter(lambda x: len(x) > n_transaction_threshold)
print('transaction has more than {} trades: {}'.format(n_transaction_threshold, len(op_clean.index)))

####-----------------------------------------------------------generate destinated dataframes------------------------------------------------------------------------------------------------
#last_two_txn
df_last_two_txn = op_clean.sort_values(['punk_id', 'block_time'], ascending=[True, True])
df_last_two_txn.to_csv('mp_df_last_two_txn.csv', index=False)
#selecting head and tail
df2_local = op_clean.sort_values(['punk_id', 'block_time'], ascending=[True, True])
df_head = df2_local.groupby('punk_id').head(1)
df_tail = df2_local.groupby('punk_id').tail(1)
df_headtail = pd.concat([df_head, df_tail])
df_headtail = df_headtail.sort_values(['punk_id', 'block_time'], ascending=[True, True])
df_headtail.to_csv('mp_df_headtail.csv', index=False)
#output from a1, a2
input_dataframe = (df_last_two_txn ,df_headtail)

####-----------------------------------------------------------perform pct change and diff-----------------------------------------------------------------------------------------------
#pct change and diff
def run_pct_change_diff_result(a):
    #grouping data
    df_groupby = a.groupby('punk_id')
    #pct_change calculation
    a['eth_pc_change'] = df_groupby['unit_in_eth'].apply(pd.Series.pct_change)    #Series.pct_change(periods=1) {A,B,C} => {"", B/A-1, C/B-1, new/old-1} time order = ascending, more recent data on bottom
    a['usd_pc_change'] = df_groupby['unit_in_usd'].apply(pd.Series.pct_change)
    #diff calculation
    a['eth_diff'] = df_groupby['unit_in_eth'].apply(pd.Series.diff)    
    a['usd_diff'] = df_groupby['unit_in_usd'].apply(pd.Series.diff)
    #keep lastest txn
    output = a.drop_duplicates(subset=['punk_id'], keep='last')
    #remove empty cell
    output = output.copy(deep=True)
    col_name = ['eth_pc_change', 'usd_pc_change', 'eth_diff', 'usd_diff']         #Note: remove empty cells from these four columns.
    for b in col_name: 
        output.loc[:, b].replace('', np.nan, inplace=True)
        output.dropna(subset=[b], inplace=True)
    return output
df_last_two = run_pct_change_diff_result(df_last_two_txn)
df_f2l = run_pct_change_diff_result(df_headtail)

####-----------------------------------------------------------remove_anomalies------------------------------------------------------------------------------------------------
#remove_anomalies
name_list = []
input_dataframe = (df_last_two ,df_f2l)
col_name = ['eth_pc_change', 'usd_pc_change', 'eth_diff', 'usd_diff']
for a in input_dataframe:
    for b in col_name:
        name = run_get_dataframe_name(a)                                          #Note: return the name of the df in string type.
        filtered_dataframe = a[['punk_id', b]]                                    #Note: create 4 different dataframes, each dataframe run through "run_remove_anomalies" function.
        anomalies, normalies = run_remove_anomalies(filtered_dataframe, 1, 0)     #Note: now percentile set to 0, no anomolies.
        df_result = (anomalies, normalies)
        #assign name and save each dataframe generate from above.
        for c in df_result: 
            name_2 = run_get_dataframe_name(c)
            save_filename = str(str(name_2) + "_" + str(name) + "_" + str(b)) + ".csv"
            name_list.append(save_filename)
            filepath = os.path.join("__filtered_result__", save_filename)
            c.to_csv(filepath, index=False)
            #print(save_filename, "\n", c)
#name list of both anomolies and normalies files, convenient for use later! 
df_name = pd.DataFrame(name_list)
df_name = df_name.sort_values(df_name.columns[0])
df_name.to_csv(os.path.join("__filtered_result__", "name_list.csv"), index=False)