import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import scikitplot as skplt
import matplotlib.pyplot as plt

# this reads the dataset
def read_feature_label_file():
    #
    features_df = pd.read_csv('mp_final_features.csv')
    features_df.drop(features_df.columns[[0,1]], axis=1, inplace=True)      #Note: need to check how many cols need to drop
    #
    label_df = pd.read_csv('mp_final_labels_combined.csv')
    label_df.drop(label_df.columns[[0]], axis=1, inplace=True)

    label_to_identify = 0           #Txn_freq = 1,  pc_change = 0
    return features_df.to_numpy(), label_df.to_numpy(), label_to_identify

def compute_scores(tag, classifier, x, y, label):
    predict_y = classifier.predict(x)
    accuracy = classifier.score(x, y)

    l = label.copy()
    print("occurence in label 0: {}".format((l == 0).sum()))
    print("occurence in label 1: {}".format((l == 1).sum()))
    print("occurence in predict 0: {}".format((predict_y == 0).sum()))
    print("occurence in predict 1: {}".format((predict_y == 1).sum()))

    tn, fp, fn, tp = confusion_matrix(y, predict_y).ravel()
    precision = precision_score(y, predict_y)
    recall = recall_score(y, predict_y)
    f1 = f1_score(y, predict_y)
    total = tn + fp + fn + tp
    total_positive = tp + fn
    total_negative = tn + fp
    tpr = tp / (tp + fn)
    tnr = tn / (fp + tn)
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)

    print("{} Confusion matrix: tp={}, tn={}, fp={}, fn={}".format(tag, tp, tn, fp, fn))
    print("{} Confusion matrix: tpr={}, fnr={}, fpr={}, fnr={}".format(tag, tpr, tnr, fpr, fnr))
    print("{} Accuracy = {}".format(tag, accuracy))
    print("{} Precision: {}".format(tag, precision))
    print("{} Recall: {}".format(tag, recall))
    print("{} F1 score: {}".format(tag, f1))
    print(" True positive rate (tpr) = {:.6f} [among all truth, model predicts as truth correctly, i.e., recall]".format(tpr))
    print(" True negative rate (tnr) = {:.6f} [among all false, model predicts as false correctly]".format(tnr))
    print("False psoitive rate (fpr) = {:.6f} [among all false, model predicts as truth incorrectly]".format(fpr))
    print("False negative rate (fnr) = {:.6f} [among all truth, model predicts as false incorrectly]".format(fnr))

    # save result to csv
    data_list = [[tag, total, total_positive, total_negative, tp, tn, fp, fn, tpr, tnr, fpr, fnr, accuracy, recall, precision, f1]]
    mp = pd.DataFrame(data_list, columns=['tag', 'total', 'total positive', 'total negative', 'tp', 'tn', 'fp', 'fn', 'tpr', 'tnr', 'fpr', 'fnr', 'accuracy', 'recall', 'precision', 'f1-score'])
    print(mp)
    return mp

def sgd_classifier(features, label, idx):
    data = pd.DataFrame(columns = ['tag', 'total', 'total positive', 'total negative', 'tp', 'tn', 'fp', 'fn', 'tpr', 'tnr', 'fpr', 'fnr', 'accuracy', 'recall', 'precision', 'f1-score', 'random_state'])
    for i in [40]:
        train_x, test_x, train_y, test_y = train_test_split(features, label, test_size=0.4, random_state=i)

        # First do a binary classification
        train_name = "train_y_" + str(idx)
        train_name = (train_y == idx)  # it is a 3 or not a 3
        classifier = SGDClassifier(
                            max_iter=1000,
                            penalty='l2',
                            verbose=0,
        )
        classifier.fit(train_x, train_name.ravel())
        mp_train = compute_scores("SGD Train", classifier, train_x, train_name, label)
        mp_train_tag = pd.DataFrame({'random_state': [i]})
        mp_train_combined = pd.concat([mp_train, mp_train_tag], axis=1)
        data = data.append(mp_train_combined, ignore_index=True)

        test_name = "test_y_" + str(idx)
        test_name= (test_y == idx)
        mp_test = compute_scores("SGD Test", classifier, test_x, test_name, label)
        mp_test_tag = pd.DataFrame({'random_state': [i]})
        mp_test_combined = pd.concat([mp_test, mp_test_tag], axis=1)
        data = data.append(mp_test_combined, ignore_index=True)
    return data

def random_forest_classifier(features, label, idx):
    data = pd.DataFrame(columns = ['tag', 'total', 'total positive', 'total negative', 'tp', 'tn', 'fp', 'fn', 'tpr', 'tnr', 'fpr', 'fnr', 'accuracy', 'recall', 'precision', 'f1-score', 'random_state'])
    for i in [40]:
        train_x, test_x, train_y, test_y = train_test_split(features, label, test_size=0.4, random_state=i)

        # First do a binary classification
        train_name = "train_y" + str(idx)
        train_name = (train_y == idx) # it is a 3 or not a 3
        classifier = RandomForestClassifier(
                                   n_estimators=100,
                                   max_leaf_nodes=1000,
                                   n_jobs=-1,
                                   verbose=1,
        )
        classifier.fit(train_x, train_name.ravel())
        mp_train = compute_scores("Random Forest Train", classifier, train_x, train_name, label)
        mp_train_tag = pd.DataFrame({'random_state': [i]})
        mp_train_combined = pd.concat([mp_train, mp_train_tag], axis=1)
        data = data.append(mp_train_combined, ignore_index=True)

        test_name = "test_y_" + str(idx)
        test_name= (test_y == idx)
        mp_test = compute_scores("Random Forest Test", classifier, test_x, test_name, label)
        mp_test_tag = pd.DataFrame({'random_state': [i]})
        mp_test_combined = pd.concat([mp_test, mp_test_tag], axis=1)
        data = data.append(mp_test_combined, ignore_index=True)
        
        predicted_probas = classifier.predict_proba(test_x)
        skplt.metrics.plot_roc_curve(test_y, predicted_probas)
        plt.title('Random Forest Classifier ROC Curve')
        plt.show()
    return data

def adaboost_classifier(features, label, idx):
    data = pd.DataFrame(columns = ['tag', 'total', 'total positive', 'total negative', 'tp', 'tn', 'fp', 'fn', 'tpr', 'tnr', 'fpr', 'fnr', 'accuracy', 'recall', 'precision', 'f1-score', 'random_state'])
    for i in [40]:
        train_x, test_x, train_y, test_y = train_test_split(features, label, test_size=0.4, random_state=40)

        # First do a binary classification
        train_name = "train_y" + str(idx)
        train_name = (train_y == idx) # it is a 3 or not a 3
        seed = 96
        classifier = AdaBoostClassifier(
                                random_state=seed,
                                #base_estimator=RandomForestClassifier(random_state=seed),
                                n_estimators=100,
                                learning_rate=0.005,
        )
        classifier.fit(train_x, train_name.ravel())
        mp_train = compute_scores("Adaptive Boost Train", classifier, train_x, train_name, label)
        mp_train_tag = pd.DataFrame({'random_state': [i]})
        mp_train_combined = pd.concat([mp_train, mp_train_tag], axis=1)
        data = data.append(mp_train_combined, ignore_index=True)

        test_name = "test_y_" + str(idx)
        test_name= (test_y == idx)
        mp_test = compute_scores("Adaptive Boost Test", classifier, test_x, test_name, label)
        mp_test_tag = pd.DataFrame({'random_state': [i]})
        mp_test_combined = pd.concat([mp_test, mp_test_tag], axis=1)
        data = data.append(mp_test_combined, ignore_index=True)

        predicted_probas = classifier.predict_proba(test_x)
        skplt.metrics.plot_roc_curve(test_y, predicted_probas)
        plt.title('Adaptive Boost Classifier ROC Curve')
        plt.show()
    return data


def run_sgd_classifier():
    final_features, final_labels, label_to_identify = read_feature_label_file()
    data = sgd_classifier(final_features, final_labels, label_to_identify)              # 0= 中間賺錢嗰堆, 1= Monalisa level
    data.to_csv('op_final_result.csv', index=False)                                     # 0= low txn freq, 1= high txn freq

def run_random_forest_classifier():
    final_features, final_labels, label_to_identify = read_feature_label_file()
    data = random_forest_classifier(final_features, final_labels, label_to_identify)    
    data.to_csv('op_final_result.csv', index=False)

def run_adaboost_classifier():
    final_features, final_labels, label_to_identify = read_feature_label_file()
    data = adaboost_classifier(final_features, final_labels, label_to_identify)
    data.to_csv('op_final_result.csv', index=False)


run_sgd_classifier()
run_random_forest_classifier()
run_adaboost_classifier()