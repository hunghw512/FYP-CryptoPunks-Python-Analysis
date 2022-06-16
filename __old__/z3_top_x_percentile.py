import pandas as pd
import numpy as np

# Read csv file
df = pd.read_csv('ip_nft.cryptopunks_trades.20220218.csv')

# extract relevant columns
df.rename({'block_time': 'block_time',
           'tx_hash': 'tx_hash',
           'nft_token_id': 'punk_id',
           'original_amount': 'unit_in_eth',
           'usd_amount': 'unit_in_usd'}, axis=1, inplace=True)
df = df[['punk_id', 'unit_in_eth', 'unit_in_usd', 'block_time', 'tx_hash', 'seller', 'buyer']]

####-----------------------------------------------------------Data cleaning process------------------------------------------------------------------------------------------------
# remove wash trade
def remove_wash_trade():
    # count unique addresses per punk
    df2 = df.groupby('punk_id')[['seller','buyer']].nunique().reset_index()
    # identify num of wash trade
    x = dict(df2['seller'])
    y = dict(df2['buyer'])
    shared_items = {k: x[k] for k in x if k in y and x[k] != y[k]}
    #print('# Wash trade identified =', len(shared_items))

    # tag the wash trade
    df2['Wash_Trade_True'] = np.where(df2['seller'] != df2['buyer'], True, False)

    # selecting rows based on condition
    df3 = df2[df2['Wash_Trade_True'] == True]

    # Select needed column and turn it into a list
    cond = df3['punk_id'].tolist()

    # remove wash trade
    indexNames = df[df['punk_id'].isin(cond) == True].index
    df.drop(indexNames , inplace=True)
    return
remove_wash_trade()

# remove unit = 0
df2 = df.loc[df['unit_in_eth'] != 0]

# remove punk_id = np.nan
df.loc[:, 'punk_id'].fillna(".", inplace=True)
df3 = df2.loc[df2['punk_id'] != "."]

#Output
op_clean = df3

####-----------------------------------------------------------上面 data cleaning process------------------------------------------------------------------------------------------------
op_clean = op_clean.sort_values(['punk_id', 'block_time'], ascending=[True, False])
head = op_clean.groupby('punk_id').head(1).reset_index(drop=True)

def run_remove_anomalies(file, column_num, percentile):
    p1_threshold = np.percentile(file.loc[:,column_num], 100-percentile)  #2 stdev (5%) for anomalies
    top_x = file[(file.loc[:,column_num] > p1_threshold)]
    return top_x

top_5_percentile = run_remove_anomalies(head, 'unit_in_eth', 5)
print(top_5_percentile)