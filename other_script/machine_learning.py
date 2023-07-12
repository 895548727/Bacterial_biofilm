from sklearn import svm
from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import joblib
from xgboost import XGBClassifier as XGB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
# import calculate_prediction
# from machinelearning import *
from sklearn.model_selection import cross_val_score
# 用于特征降维的机器学习算法
import math
def calculate_metrics(labels, scores, cutoff=0.5, po_label=1):
    my_metrics = {
        'Sensitivity': 'NA',
        'Specificity': 'NA',
        'Accuracy': 'NA',
        'MCC': 'NA',
        'Recall': 'NA',
        'Precision': 'NA',
        'F1-score': 'NA',
        'Cutoff': cutoff,
    }
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(scores)):
        if labels[i] == po_label:
            if scores[i] >= cutoff:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if scores[i] < cutoff:
                tn = tn + 1
            else:
                fp = fp + 1

    my_metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) != 0 else 'NA'
    my_metrics['Specificity'] = tn / (fp + tn) if (fp + tn) != 0 else 'NA'
    my_metrics['Accuracy'] = (tp + tn) / (tp + fn + tn + fp)
    my_metrics['MCC'] = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (
        tp + fn) * (tn + fp) * (tn + fn) != 0 else 'NA'
    my_metrics['Precision'] = tp / (tp + fp) if (tp + fp) != 0 else 'NA'
    my_metrics['Recall'] = my_metrics['Sensitivity']
    my_metrics['F1-score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 'NA'
    # print(my_metrics)
    return my_metrics
def ml_ten(data_pd, types, b=1, n=1):
    acc = []
    metrics = []
    X,y = data_pd.iloc[n:, 2:],data_pd.iloc[n:, 1].astype("int")
    # print(X)
    X, y = np.array(X),np.array(y)
    cv = StratifiedKFold(n_splits=10)
    # parameter_space = {
    #     "n_estimators":[10,15,20],
    #     "criterion": ["gini","entropy"],
    #     "min_samples_leaf": [2,4,6]
    # }
    # scores = ['roc_auc']
    for train_index, test_index in cv.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier(n_estimators=100
                                     ,random_state=0,criterion='gini',min_samples_leaf=4)
        # grid = GridSearchCV(clf,parameter_space)
        # clf = svm.SVC()
        clf.fit(X_train, y_train)
        joblib.dump(clf,"./"+ types+".pkl")
        metrics.append(calculate_metrics(y_test.tolist(),clf.predict(X_test)))
    # score = cross_val_score(clf, X, y, cv=cv).mean();
        acc.append(accuracy_score(y_train, clf.predict(X_train)))
	# eva['Sensitivity']
	# eva['Specificity']
	# eva['Accuracy']
	# eva['MCC']
	# eva['Recall']
	# eva['Precision']
	# eva['F1-score']
	# print ('测试集准确率：', accuracy_score(y_test, clf.predict(X_test)));
    # auc_value = roc_auc_score(y_test, clf.predict(X_test).tolist());
    # data_plot = pd.merge(y_test, clf.predict(X_test))
    # plot_roc_cv()
    # print(auc_value)
    importance = 0;
    if b == 0:
        importance = clf.feature_importances_;
    # print(acc)
    my_metrics = []
    for i in metrics[0].keys():
        sum = 0
        for j in range(len(metrics)):
            sum += metrics[j][i]
        my_metrics.append(sum/10)
    print(my_metrics)
    # joblib.dump(clf, types + '.pkl');
    # h = clf.predict_proba(X)
    return clf.predict(X), importance, X_test, y_test, clf.predict(X_test),y

def ml(data_pd, types, b=1, n=1):
    acc = []
    metrics = []
    X,y = data_pd.iloc[n:, 2:], data_pd.iloc[n:, 1].astype("int")
    # print(X)
    X, y = np.array(X),np.array(y)
    cv = StratifiedKFold(n_splits=10)
    # parameter_space = {
    #     "n_estimators":[10,15,20],
    #     "criterion": ["gini","entropy"],
    #     "min_samples_leaf": [2,4,6]
    # }
    # scores = ['roc_auc']
    for train_index, test_index in cv.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier(n_estimators=100
                                     ,random_state=0,criterion='gini',min_samples_leaf=4)
        # grid = GridSearchCV(clf,parameter_space)
        # clf = svm.SVC()
        clf.fit(X_train, y_train)
        metrics.append(calculate_metrics(y_test.tolist(),clf.predict(X_test)))
    # score = cross_val_score(clf, X, y, cv=cv).mean();
        acc.append(accuracy_score(y_train, clf.predict(X_train)))
	# eva['Sensitivity']
	# eva['Specificity']
	# eva['Accuracy']
	# eva['MCC']
	# eva['Recall']
	# eva['Precision']
	# eva['F1-score']
	# print ('测试集准确率：', accuracy_score(y_test, clf.predict(X_test)));
    # auc_value = roc_auc_score(y_test, clf.predict(X_test).tolist());
    # data_plot = pd.merge(y_test, clf.predict(X_test))
    # plot_roc_cv()
    # print(auc_value)
    importance = 0;
    if b == 0:
        importance = clf.feature_importances_;
    # print(acc)
    my_metrics = []
    for i in metrics[0].keys():
        sum = 0
        for j in range(len(metrics)):
            sum += metrics[j][i]
        my_metrics.append(sum/10)
    print(my_metrics)
    # joblib.dump(clf, types + '.pkl');
    # h = clf.predict_proba(X)
    return clf.predict(X), importance, X_test, y_test, clf.predict(X_test),y
def xgboost(data_pd, types, b=1, n=1):
    acc = []
    metrics = []
    X,y = data_pd.iloc[n:, 2:], data_pd.iloc[n:, 1].astype("int")
    # print(X)
    X, y = np.array(X),np.array(y)
    cv = StratifiedKFold(n_splits=10)
    # parameter_space = {
    #     "n_estimators":[10,15,20],
    #     "criterion": ["gini","entropy"],
    #     "min_samples_leaf": [2,4,6]
    # }
    # scores = ['roc_auc']
    for train_index, test_index in cv.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = XGB(n_estimators=100)
        # grid = GridSearchCV(clf,parameter_space)
        # clf = svm.SVC()
        clf.fit(X_train, y_train)
        metrics.append(calculate_metrics(y_test.tolist(),clf.predict(X_test)))
    # score = cross_val_score(clf, X, y, cv=cv).mean();
        acc.append(accuracy_score(y_train, clf.predict(X_train)))
	# eva['Sensitivity']
	# eva['Specificity']
	# eva['Accuracy']
	# eva['MCC']
	# eva['Recall']
	# eva['Precision']
	# eva['F1-score']
	# print ('测试集准确率：', accuracy_score(y_test, clf.predict(X_test)));
    # auc_value = roc_auc_score(y_test, clf.predict(X_test).tolist());
    # data_plot = pd.merge(y_test, clf.predict(X_test))
    # plot_roc_cv()
    # print(auc_value)
    importance = 0;
    if b == 0:
        importance = clf.feature_importances_;
    # print(acc)
    my_metrics = []
    for i in metrics[0].keys():
        sum = 0
        for j in range(len(metrics)):
            sum += metrics[j][i]
        my_metrics.append(sum/10)
    print(my_metrics)
    # joblib.dump(clf, types + '.pkl');
    # h = clf.predict_proba(X)
    return clf.predict(X), importance, X_test, y_test, clf.predict(X_test),y

def svm_self(data_pd, types, b=1, n=1):
    acc = []
    metrics = []
    X,y = data_pd.iloc[n:, 2:], data_pd.iloc[n:, 1].astype("int")
    # print(X)
    X, y = np.array(X),np.array(y)
    cv = StratifiedKFold(n_splits=10)
    # parameter_space = {
    #     "n_estimators":[10,15,20],
    #     "criterion": ["gini","entropy"],
    #     "min_samples_leaf": [2,4,6]
    # }
    # scores = ['roc_auc']
    for train_index, test_index in cv.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # clf = XGB(n_estimators=100)
        # grid = GridSearchCV(clf,parameter_space)
        # clf = svm.SVC(kernel='rbf', degree=3)
        # clf = LogisticRegression()
        clf = RandomForestClassifier(n_estimators=100
                                     , random_state=0, criterion='gini', min_samples_leaf=4)
        clf.fit(X_train, y_train)
        joblib.dump(clf, "./svm_model/" + types + ".pkl")
        metrics.append(calculate_metrics(y_test.tolist(),clf.predict(X_test)))
    # score = cross_val_score(clf, X, y, cv=cv).mean();
        acc.append(accuracy_score(y_train, clf.predict(X_train)))
	# eva['Sensitivity']
	# eva['Specificity']
	# eva['Accuracy']
	# eva['MCC']
	# eva['Recall']
	# eva['Precision']
	# eva['F1-score']
	# print ('测试集准确率：', accuracy_score(y_test, clf.predict(X_test)));
    # auc_value = roc_auc_score(y_test, clf.predict(X_test).tolist());
    # data_plot = pd.merge(y_test, clf.predict(X_test))
    # plot_roc_cv()
    # print(auc_value)
    importance = 0;
    if b == 0:
        importance = clf.feature_importances_;
    # print(acc)
    my_metrics = []
    for i in metrics[0].keys():
        sum = 0
        for j in range(len(metrics)):
            if (metrics[j][i] == 'NA'):
                sum += 0
            else:
                sum += metrics[j][i]
        my_metrics.append(sum/10)
    print(my_metrics)
    # joblib.dump(clf, types + '.pkl');
    # h = clf.predict_proba(X)
    return clf.predict(X), importance, X_test, y_test, clf.predict(X_test),y