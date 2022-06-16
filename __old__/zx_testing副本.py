import pandas as pd
import numpy as np
import sys

cryptopunks_trades = pd.read_csv('ip_nft.cryptopunks_trades.20220218.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)

### find the major focus characteristics which help us to analysis data
characteristics = cryptopunks_trades[['tx_hash','platform','nft_project','nft_token_id','trade_type','category','evt_type','buyer','seller','from','to']]
df = characteristics.nunique()
print(df)

### Since there are two type of trade type, we need to know what it is
trade_type_details = print('two trade type are:')
print(pd.value_counts(cryptopunks_trades['trade_type'])) # show what two type is

a = len(cryptopunks_trades['tx_hash'].value_counts())
b = len(cryptopunks_trades['tx_hash'].value_counts()) - len(cryptopunks_trades[cryptopunks_trades['trade_type'] == "Bundle Trade"])
c_1 = cryptopunks_trades.copy()
c_2 = c_1[c_1['original_amount'] != 0]
c_3 = c_2[c_2['usd_amount'] != 0]
c_3.loc[:, 'nft_token_id'].fillna(".", inplace=True)
c_4 = c_3.loc[c_3['nft_token_id'] != "."]
c_5 = len(c_4)

print("Total count before filter: {}".format(a))
print("Total count filtered by type (Single Item Trade): {}".format(b))
print("Total count filtered by non-zero transaction value: {}".format(c_5))



def transaction_change(cryptopunks_trades):

    df_calculation = cryptopunks_trades.copy

    ### take transaction >2 Cryptopunks and groupby token id to create new df
    n_transaction = 1
    df_groupby = df_calculation.groupby('nft_token_id')
    df_calculation = df_calculation.groupby['nft_token_id'].filter(lambda x: len(x) > n_transaction)
    print('Transaction have more than {} trades: {}', format(n_transaction, len(df_calculation.index)),file= sys.stderr, flush=True)

    ### find price pct change & ETH pct change
    df_calculation['USD_amount_pct_change'] = df_groupby['usd_amount'].apply(pd.Series.pct_change)
    df_calculation['ETH_amount_pct_change'] = df_groupby['original_amount'].apply(pd.Series.pct_change)

    ### find the block distance of same token id, and put other else to 0
    df_calculation['block_distance'] = np.where(df_calculation['nft_token_id'] == df_calculation['nft_token_id'].shift(1), df_calculation['block_number'].diff(),0)

    # exclude 1st and no record trade of USD and ETH
    df_calculation =  df_calculation.loc[df_calculation['USD_amount_pct_change'] != np.NaN]
    print('Excluding 1st and no record USD trade: {}', format(len(df_calculation.index)), file= sys.stderr, flush=True)
    df_calculation = df_calculation.loc[df_calculation['ETH_amount_pct_change'] != np.NaN]
    print('Excluding 1st and no record ETH trade: {}', format(len(df_calculation.index)), file= sys.stderr, flush=True)

    """df_calculation = df_calculation.loc[df_calculation['USD_amount_pct_change'] != np.inf]
    print('Excluding USD trade pct change with infinity: {}', format(len(df_calculation.index)), file=sys.stderr, flush=True)
    df_calculation = df_calculation.loc[df_calculation['ETH_amount_pct_change'] != np.inf]
    print('Excluding ETH trade pct change with infinity: {}', format(len(df_calculation.index)), file=sys.stderr, flush=True)"""

    return df_calculation