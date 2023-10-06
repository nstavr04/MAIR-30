# This file contains additional functions to evaluate the trained models

from Baseline_Classifiers import baseline_predict
from Baseline_Classifiers import baseline_majority_predict
from ML_Classifiers import get_decision_tree_classifier
from ML_Classifiers import get_logistic_regression_classifier

import numpy as np

from Data_Preparation import get_data

x_data_dup,y_data_dup,x_deduplicated,y_deduplicated = get_data(shuffel=True)
x_data_dup = x_data_dup.values
y_data_dup = y_data_dup.values
x_deduplicated = x_deduplicated.values
y_deduplicated = y_deduplicated.values

def find_difficult_instances(dublicates= False):
    if dublicates:
        x_data = x_data_dup
        y_data = y_data_dup
    else:
        x_data = x_deduplicated
        y_data = y_deduplicated
    splits = [int(len(x_data)/(len(x_data)/10)*step) for step in [1,2,3,4,5,6,7,8,9,10]]
    splits_dd = [int(len(x_deduplicated) / (len(x_deduplicated) / 10) * step) for step in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    prev = 0
    for i in range(10):
        x_test = x_data[prev:splits[i]]
        y_test = y_data[prev:splits[i]]
        #x_test_dd = x_deduplicated[prev:splits_dd[i]]
        #y_test_dd = y_deduplicated[prev:splits_dd[i]]
        x_train = []
        y_train = []
        for j in range(len(x_data)):
            if j < prev or j >= splits[i]:
                x_train.append(x_data[j])
                y_train.append(y_data[j])

        #x_train_dd = [x for x in x_deduplicated if x not in x_test_dd ]
        #y_train_dd = [x for x in y_deduplicated if x not in y_test_dd]

        y_bl_maj = baseline_majority_predict(x_test)
        y_bl_rules = baseline_predict(x_test)
        dt, vec_dt = get_decision_tree_classifier(x_train,y_train)
        y_dt = dt.predict(vec_dt.transform(x_test))
        lr, vec_dt = get_logistic_regression_classifier(x_train,y_train)
        y_lr = lr.predict(vec_dt.transform(x_test))
        for k,target in enumerate(y_test):
            if target != y_bl_rules[k] and (target != y_dt[k] and target != y_lr[k]):
                print('missclassified utterances')
                print('Uttrance: ',x_test[k])
                print('True label: ',y_test[k])
                print()

        prev = splits[i]

if __name__ == '__main__':
    find_difficult_instances(False)




