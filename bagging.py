import numpy as np
import Decision_Tree.decision_tree as decision_tree
import copy
import random
import argparse
import pandas as pd
from Data_Loader.car_data_loader import datalabel, feature2id, id2feature, load_data

class Bagging(object):

    def __init__(self, base_model, model_count=10):
        self.base_model = base_model
        self.base_model_list = []
        self.model_count = model_count

    def fit(self, X_train, Y_train, label_num, rate=0.632):
        # Generate decision tree
        for i in range(self.model_count):
            base_model = copy.deepcopy(self.base_model)
            # Bagging data
            n, m = X_train.shape
            sample_idx = np.random.permutation(n)[:int(n * rate)]
            X_t_, Y_t_ = X_train[sample_idx, :], Y_train[sample_idx]
            # Train
            base_model.fit(X_t_, Y_t_, label_num)
            self.base_model_list.append(base_model)
            print('=' * 10 + ' %r/%r base model trained ' % (i + 1, self.model_count) + '=' * 10)
            # print(dt_CART.visualization())

    def predict(self, X):
        output_matrix = np.zeros((self.model_count, X.shape[0]))
        output_label = np.zeros(X.shape[0])
        for i, base_model in enumerate(self.base_model_list):
            output_matrix[i, :] = base_model.predict(X)
        for col in range(output_matrix.shape[1]):
            output_label[col] = np.argmax(np.bincount(output_matrix[:, col].astype(int)))
        return output_label.astype(int)

def count_acc(gt_y, pred_y):
    return sum(gt_y==pred_y)/ len(pred_y)

if __name__ == '__main__':    
    random.seed(112)
    np.random.seed(112)
    parser = argparse.ArgumentParser(description='Decision Tree')
    parser.add_argument(
            "--draw_trees",
            action="store_true",
            help="Whether to draw a decision tree with a small data set",
        )
    parser.add_argument(
            "--sample_name",
            type=int, default=40,
            help="How many data would like to use for drawing the trees",
        )
    parser.add_argument(
            "--alpha",
            type=float, default=2.0,
            help="weight factor for original prune",
        )
    parser.add_argument(
            "--train_test_frac",
            type=float, default=0.8,
            help="The data ratio of the training set to the test set",
        )

    args = parser.parse_args()

    print(args)
    print(datalabel)
    data_sets = load_data("/home/yangzhixian/ML/data/car.data", ',')
    row_, col_ = data_sets.shape

    train_test_frac = args.train_test_frac
    train_row = int(row_ * train_test_frac)
    print("train row is :", train_row)
    print("test row is :", row_ - train_row)
    train_sets = data_sets[:train_row, :]
    test_sets = data_sets[train_row : , :]
    train_sets_encode = np.array([[feature2id[j][train_sets[i, j]] for j in range(col_)] for i in range(train_row)])
    test_sets_encode = np.array([[feature2id[j][test_sets[i, j]] for j in range(col_)] for i in range(row_-train_row)])
    
    train_X_t, train_Y_t = train_sets_encode[:, :-1], train_sets_encode[:, -1]
    test_X_t, test_Y_t = test_sets_encode[:, :-1], test_sets_encode[:, -1]

    base_model = decision_tree.DTreeCART()
    Bagging_model = Bagging(base_model)
    Bagging_model.fit(train_X_t, train_Y_t, len(feature2id[-1].values()))
    Bagging_pred_y = Bagging_model.predict(test_X_t)
    print("Bagging model in Test set acc : ", count_acc(test_Y_t, Bagging_pred_y))

    CART_model = decision_tree.DTreeCART()
    CART_model.fit(train_X_t, train_Y_t, len(feature2id[-1].values()))
    CART_pred_y = CART_model.predict(test_X_t).astype(int)
    print("CART Test set acc : ", count_acc(test_Y_t, CART_pred_y))
