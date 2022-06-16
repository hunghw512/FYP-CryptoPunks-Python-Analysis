import pandas as pd
from p0_common import run_label_percentile, read_file_run_label, read_csv

####-----------------------------------------------------------Common parameters------------------------------------------------------------------------------------------------
#Common parameters to adjust
#Note: if want to use different labelling, please go down below
file_loc = "__filtered_result__"                                   #Note: most of the files are stored in folder
diff_file = "mp_df_negative_return"
pc_change_file = "mp_df_negative_return"
df_col = "1"                                                       #Note: the column to perform labelling

#below are the files to run
'''
normalies_df_f2l_eth_diff
normalies_df_f2l_usd_diff
normalies_df_last_two_eth_diff
normalies_df_last_two_usd_diff

normalies_df_f2l_eth_pc_change
normalies_df_f2l_usd_pc_change
normalies_df_last_two_eth_pc_change
normalies_df_last_two_usd_pc_change

mp_df_last_two_eth_r
mp_df_last_two_usd_r
mp_df_f2l_eth_r
mp_df_f2l_usd_r

mp_df_average
mp_df_negative_return
'''


'''------------------------------------------------------------Content------------------------------------------------------------------------------------------------------------------------------------
a: generate mp_final_labels_combined
b: generate mp_final_features'''

####-----------------------------------------------------------mp_final_labels_combined------------------------------------------------------------------------------------------------
#a1. label based on absolute value
bins = [0, 90, 100]                                                #Note: can adjust here if want to use different labelling
label = ["0-90", "90-100"]                                         #Note: can adjust here if want to use different labelling
name = diff_file + "_label"
name = read_file_run_label(file_loc, diff_file, bins, label, 1)
#a2. extract 75-100 percentile
def extract(a):
    a = a[a.iloc[:,2].isin(extract_range) == True]
    con = a.loc[:, 'punk_id'].tolist()
    #print("\n75-100 percentile\n", a)
    return con
extract_range = label                                              #Note: no extraction now.
#a3. match the lenght of diff_file and pc_change_file
pc_change_file = read_csv(file_loc, pc_change_file)
con = extract(name)
output = pc_change_file.loc[pc_change_file['punk_id'].isin(con) == True]
#a4. mp_final_labels_combined
bins2 = [0, 90, 100]                                               #Note: can adjust here if want to use different labelling
label2 = ["1", "2"]                                                #Note: can adjust here if want to use different labelling
final_labels_combined = run_label_percentile(output, bins2, label2, 1)
final_labels_combined.to_csv('mp_final_labels_combined.csv', index=False)

####-----------------------------------------------------------mp__final_features------------------------------------------------------------------------------------------------
#b. mp_final_features
#b1. read csv file
df_l = final_labels_combined
df_f = pd.read_csv('ip_df_without_noise.csv')
#b2. removing rows based on condition
cond = df_l['punk_id'].tolist()
final_feature = df_f[df_f['id'].isin(cond) == True]
final_feature.to_csv('mp_final_features.csv', index=False)