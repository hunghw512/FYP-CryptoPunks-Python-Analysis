import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import multilabel_confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt

# this reads the dataset
def read_feature_label_file():
    #
    features_df = pd.read_csv('mp_final_features.csv')
    features_df.drop(features_df.columns[[0,1]], axis=1, inplace=True)
    #
    label_df = pd.read_csv('mp_final_labels_combined.csv')
    label_df.drop(label_df.columns[[0]], axis=1, inplace=True)
    return features_df.to_numpy(), label_df.to_numpy()

def compute_average_scores(tag, classifier, x, y, label):
    
    predict_y = classifier.predict(x)
    accuracy = classifier.score(x, y)
    
    l = label.copy()
    print("occurence in label 0: {}".format((l == 0).sum()))
    print("occurence in label 1: {}".format((l == 1).sum()))
    print("occurence in label 2: {}".format((l == 2).sum()))
    print("occurence in label 3: {}".format((l == 3).sum()))
    print("occurence in predict 0: {}".format((predict_y == 0).sum()))
    print("occurence in predict 1: {}".format((predict_y == 1).sum()))
    print("occurence in predict 2: {}".format((predict_y == 2).sum()))
    print("occurence in predict 3: {}".format((predict_y == 3).sum()))

    cm = multilabel_confusion_matrix(y, predict_y)

    # computing tn, fp, fn, tp
    num_classifier = cm.shape[0]

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # computing for each idx
    data = []
    for idx in range(num_classifier):
        tp += cm[idx][1][1]  #Values that are actually positive and predicted positive
        tn += cm[idx][0][0]  #Values that are actually negative and predicted negative
        fp += cm[idx][0][1]  #Values that are actually negative and predicted positive
        fn += cm[idx][1][0]  #Values that are actually positive and predicted negative

        total = cm[idx][1][1] + cm[idx][0][0] + cm[idx][0][1] + cm[idx][1][0]
        total_positive = cm[idx][1][1] + cm[idx][1][0]
        total_negative = cm[idx][0][0] + cm[idx][0][1]

        tpr = cm[idx][1][1] / (cm[idx][1][1] + cm[idx][1][0])
        tnr = cm[idx][0][0] / (cm[idx][0][1] + cm[idx][0][0]) 
        fpr = cm[idx][0][1] / (cm[idx][0][1] + cm[idx][0][0])
        fnr = cm[idx][1][0] / (cm[idx][1][1] + cm[idx][1][0]) 

        recall = cm[idx][1][1] / (cm[idx][1][1] + cm[idx][1][0])
        precision = cm[idx][1][1] / (cm[idx][1][1] + cm[idx][0][1]) #Among all predicted positive, how often is it correct(true positive).
        f1 = 2.0 * precision * recall / (precision + recall)

        # save result to list
        data_list = [total, total_positive, total_negative, cm[idx][1][1], cm[idx][0][0], cm[idx][0][1], cm[idx][1][0], tpr, tnr, fpr, fnr, recall, precision, f1]
        data.append(data_list)

    mp = pd.DataFrame(data, columns=['total', 'total positive', 'total negative', 'tp', 'tn', 'fp', 'fn', 'tpr', 'tnr', 'fpr', 'fnr', 'recall', 'precision', 'f1-score'])
    
    # computing for sum of idx
    tpr = tp / (tp + fn)
    tnr = tn / (fp + tn)
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)

    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    f1 = 2.0 * precision * recall / (precision + recall)

    print("{} Confusion matrix: tp={}, tn={}, fp={}, fn={}".format(tag, tp, tn, fp, fn))
    print("{} Accuracy = {:.6f}".format(tag, accuracy))
    print("{} Precision: {:.6f}".format(tag, precision))
    print("{} Recall: {:.6f}".format(tag, recall))
    print("{} F1 score: {:.6f}".format(tag, f1))
    print(" True positive rate (tpr) = {:.6f} [among all truth, model predicts as truth correctly, i.e., recall]".format(tpr))
    print(" True negative rate (tnr) = {:.6f} [among all false, model predicts as false correctly]".format(tnr))
    print("False psoitive rate (fpr) = {:.6f} [among all false, model predicts as truth incorrectly]".format(fpr))
    print("False negative rate (fnr) = {:.6f} [among all truth, model predicts as false incorrectly]".format(fnr))

    #computing for weighted
    n = 0.25
    w_tp = round(tp * n, 0)
    w_tn = round(tn * n, 0)
    w_fp = round(fp * n, 0)
    w_fn = round(fn * n, 0)

    w_total = w_tp + w_tn + w_fp + w_fn
    w_positive = w_tp + w_fn
    w_negative = w_tn + w_fp

    w_tpr = w_tp / (w_tp + w_fn)
    w_tnr = w_tn / (w_fp + w_tn)
    w_fpr = w_fp / (w_fp + w_tn)
    w_fnr = w_fn / (w_tp + w_fn)

    w_recall = w_tp / (w_tp + w_fn)
    w_precision = w_tp / (w_tp + w_fp)
    w_f1 = 2.0 * w_precision * w_recall / (w_precision + w_recall)

    weighted_list = [["weighted", w_total, w_positive, w_negative, w_tp, w_tn, w_fp, w_fn, w_tpr, w_tnr, w_fpr, w_fnr, w_recall, w_precision, w_f1]]
    w_mp = pd.DataFrame(weighted_list, columns=['', 'total', 'total positive', 'total negative', 'tp', 'tn', 'fp', 'fn', 'tpr', 'tnr', 'fpr', 'fnr', 'recall', 'precision', 'f1-score'])
    w_mp.set_index("", inplace=True)
    output = mp.append(w_mp)
    print(output, "\n")

    return output

def run_random_forest_classifier(features, label):
    train_x, test_x, train_y, test_y = train_test_split(features, label, test_size=0.4, random_state=40)

    for n in [100]:
        for m in [1000]:
            classifier = RandomForestClassifier(
                n_estimators=n,
                max_leaf_nodes=m,
                n_jobs=-1,
                verbose=1,
            )
            classifier.fit(train_x, train_y)
            
            output = compute_average_scores("Random Forest Train", classifier, train_x, train_y, label)
            output = compute_average_scores("Random Forest Test", classifier, test_x, test_y, label)
            output.to_csv('111.csv')
            if ((output.iloc[2,3]>0) and (output.iloc[3,3]>0)):
                print("break!")
                print("The n_estimators is {}".format(n))
                print("The max_leaf_nodes is {}".format(m))
                break
            elif ((n==2000) and (m==2000)):
                print("No result, the loop ended.")

    predicted_probas = classifier.predict_proba(test_x)
    skplt.metrics.plot_roc_curve(test_y, predicted_probas)
    plt.title('F2L Random Forest Classifier ROC Curve')
    plt.show()
    return

def run_adaboost_classifier(features, label):
    train_x, test_x, train_y, test_y = train_test_split(features, label, test_size=0.4, random_state=40)

    seed = 96
    classifier = AdaBoostClassifier(
        random_state=seed,
        base_estimator=RandomForestClassifier(random_state=seed),
        n_estimators=100,
        learning_rate=0.005,
    )
    classifier.fit(train_x, train_y.ravel())
    
    compute_average_scores("Adaptive Boost Train", classifier, train_x, train_y, label)
    compute_average_scores("Adaptive Boost Test", classifier, test_x, test_y, label)

    predicted_probas = classifier.predict_proba(test_x)
    skplt.metrics.plot_roc_curve(test_y, predicted_probas)
    plt.title('F2L Adaptive Boost Classifier ROC Curve')
    plt.show()
    return

def run_random_forest():
    final_features, final_labels = read_feature_label_file()
    run_random_forest_classifier(final_features, final_labels)

def run_adaboost():
    final_features, final_labels = read_feature_label_file()
    run_adaboost_classifier(final_features, final_labels)

run_random_forest()
run_adaboost()

'''def run_sgd_classifier(features, label):
    train_x, test_x, train_y, test_y = train_test_split(features, label, test_size=0.3, random_state=40)

    classifier = SGDClassifier(
                        max_iter=1000,
                        penalty='l2',
                        verbose=0,
    )
    classifier.fit(train_x, train_y)

    compute_average_scores("SGD Train", classifier, train_x, train_y, label)
    compute_average_scores("SGD Test", classifier, test_x, test_y, label)

    return

def run_sgd():
    final_features, final_labels = read_feature_label_file()
    run_sgd_classifier(final_features, final_labels)

#run_sgd()'''