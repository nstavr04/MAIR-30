from BaselineClassifiers import baseline_predict
from BaselineClassifiers import baseline_majority_predict
from ML_Classifiers import get_decision_tree_classifier
from ML_Classifiers import get_logistic_regression_classifier

import numpy as np

from DataPreparation import get_data

x_data,y_data,x_deduplicated,y_deduplicated = get_data(shuffel=True)
x_data = x_data.values
y_data = y_data.values
x_deduplicated = x_deduplicated.values
y_deduplicated = y_deduplicated.values

def find_difficult_instances():
    splits = [int(len(x_data)/(len(x_data)/10)*step) for step in [1,2,3,4,5,6,7,8,9,10]]
    splits_dd = [int(len(x_deduplicated) / (len(x_deduplicated) / 10) * step) for step in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    prev = 0
    for i in range(10):
        x_test = x_data[prev:splits[i]]
        y_test = y_data[prev:splits[i]]
        print(len(x_test) == len(y_test))
        #x_test_dd = x_deduplicated[prev:splits_dd[i]]
        #y_test_dd = y_deduplicated[prev:splits_dd[i]]
        x_train = []
        y_train = []
        for j in range(len(x_data)):
            if j < prev or j >= splits[i]:
                x_train.append(x_data[i])
                y_train.append(y_data[i])

        #x_train_dd = [x for x in x_deduplicated if x not in x_test_dd ]
        #y_train_dd = [x for x in y_deduplicated if x not in y_test_dd]
        print(x_test[1])
        print(x_train[1])

        y_bl_maj = baseline_majority_predict(x_test)
        y_bl_rules = baseline_predict(x_test)
        print(type(x_test))
        dt = get_decision_tree_classifier(x_train,y_train)
        print(dt.predict(np.array(["hello",'great'])))
        y_dt = dt.predict(x_test)
        lr = get_logistic_regression_classifier(x_train,y_train)
        y_lr = lr.predict(x_test)
        for i,target in enumerate(y_test):
            if target != y_bl_maj[i] and target != y_bl_rules[i] and target != y_dt[i] and target != y_lr[i]:
                print('missclassified utterances')
                print(x_test[i])
                print(y_test[i])

        prev = splits[i]

if __name__ == '__main__':
    find_difficult_instances()




