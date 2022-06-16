# FYP-CryptoPunks-Python-Classification Description

#For p0_common, we placed most of the self-defined functions here. Whenever you see a self-defined function in other py files, please check p0_common for a detailed look.

#For p1_data_cleaning, we performed most of the data cleaning processes here. The processes included:<br />
(1) extract relevant columns from the raw data input (ip_nft.cryptopunks_trades.20220218.csv) and rename the relevant column names,<br />
(2) remove wash trade based on the unique addresses,<br />
(3) remove zero value in the column of 'unit_in_eth',<br />
(4) remove empty value in the column of 'punk_id' (for some reasons, very few rows of 'punk_id' showed empty cell, yet the amount of empty cells is insignificant, we jsut removed them.)

#For p2_1-filtered_result, we created 16 csv files here.<br />These files are in two types, namely anomalies and normalies.<br />And each type is seperated by two different dimensions of focus (first to last transactions and last two transactions), two different unit of transactions (usd and eth), and at last two different calculation methods (percentage change and absolute return).<br />Therefore, there are in total of 16 csv files (2 x 2 x 2 x 2).<br />The name list of these files are stored together with the files inside the __filtered_result__ folder.<br />
After explaining the output, let us go through the coding here:<br />
(1) filter out transaction frequency count<br />
(2) generate required files (first to last transactions and last two transactions)<br />
(3) perform percentage change and absolute return (i.e., difference) on both eth and usd<br />
(4) remove anomalies (yet, for current objective, that is to look at extremely high return, we do not remove any outliners<br />(i.e., setting 0 in the third parameter of run_remove_anomalies function).<br />So, for files that start with anomalies, there are nothing inside.)
*Note: there is a content section for p2_filtered_result on the top of this py file.

#For p2_2-r_result, we created 4 csv files, which are alike above. These files are the compounded return of both usd and eth for transaction data that is first to last and last two. They are also stored in the __filtered_result__ folder.
After explaining the output, let us go through the coding here:
(1) import useful module (just mentioning it here, otherwise (1) will be missing.)
(2) read in csv files generated from p2_filtered_result and did a little bit of arrangement (specific to mention that mp csv files are also from p2_filtered_result)
(3) calulate the holding period
(4) calulate the compounded return (Note: There is no absolute return for compounded return.)
(5) exporting files
(6) checking weird result for compounded return (three types: inf, 0 and -1)
(7) function to plot graph
(8) run_remove_anomalies function
(9 & 10) plot scrattered diagrams to see the trend of compounded return versu holding period (We tired to remove some anomalies, so we can see a clearer trend.)

#For p3_1-4, we will produce the final outputs for both binary and multilabel classifers. 
For txn_freq files, they produce transaction frequency's label and feature. 
For pc_change files, they produce pc_change's label and feature in ETH units with two viewpoints (i.e. last two transactions and first to last transactions)

#For y1 and y2, they are run classifier in binary and multilabel scenario.
Be careful when running binary classification. 
If you are running txn freq, please change 'label_to_identify' (in line 20) to 1. 
If you are running price percentage change, please change 'label_to_identify' (in line 20) to 0. 

#For py files that start with z, those are irrelevant files and just for our own reference. Can ignore.

