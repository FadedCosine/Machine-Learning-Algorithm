import numpy as np
import random
import queue
import copy
import argparse
import pandas as pd
import Decision_Tree.draw_tree as draw_tree
from Data_Loader.car_data_loader import datalabel, feature2id, id2feature, load_data
import time

class Node(object):

    def __init__(self, parent=None, pre_split_feature_x=None):
        self.split_feature = None
        # self.split_feature_name = ""
        self.cart_split_feature_x = None
        # self.cart_split_feature_x_name = ""
        self.pre_split_feature_x = pre_split_feature_x
        # self.pre_split_feature_x_name = pre_split_feature_x_name
        self.s = None  # Number, 作为连续性特征的分裂点取值
        self.child = []
        self.y = None
        self.hidden_y_for_prune = None
        self.data = None
        self.exp_entropy = None
        self.data_len = None
        self.pivot_data_num = None
        self.parent = parent

    def append(self, child):
        self.child.append(child)

    def predict_classification(self, features, kind="NotCART"):
        if self.y is not None:
            return self.y
        for child in self.child:
            if kind == "CART":
                if (1-child.pre_split_feature_x) == (features[self.split_feature] == self.cart_split_feature_x):
                    return child.predict_classification(features, kind)
            else:
                if child.pre_split_feature_x == features[self.split_feature]:
                    return child.predict_classification(features)
        return self.child[0].predict_classification(features)

    def predict_regression(self, features): # for continuous features, in CART tree
        if self.y is not None:
            return self.y
        child_idx = 0 if features[self.split_feature] <= self.s else 1
        return self.child[child_idx].predict_regression(features)

    def is_leaf(self):#判断当前节点是否是叶节点
        return len(self.child) == 0

    def if_children_leaf(self):#判断当前节点的所有子节点是否都是叶节点
        if len(self.child) == 0: #一定是内部节点，如果是叶子节点不可以
            return False
        else:
            all_children_leaf = True
            for child in self.child:
                all_children_leaf = (all_children_leaf and child.is_leaf())
            return all_children_leaf
    
    


class DTreeID3(object):

    def __init__(self, epsilon=1e-5):
        self.tree = Node()
        self.epsilon = epsilon
        self.label_num = None
      

    def fit(self, X_train, Y_train, label_num, K=None):
        feature_ids = np.arange(X_train.shape[1])
        self.label_num = label_num
        self._train(X_train, Y_train, self.tree, feature_ids, K)

    def predict(self, X):
        n = X.shape[0]
        Y = np.zeros(n)
        for i in range(n):
            Y[i] = self.tree.predict_classification(X[i, :])
        return Y

    
    def _train(self, X_train, Y_train, node, feature_ids, K=None):
        # 注意 X_train，也就yes样本在特征空间的取值， Y_train yes 样本的label
        # 计算当前结点的经验熵
        prob = self._cal_prob(Y_train)
        prob = np.array([a if 0 < a <= 1 else 1 for a in prob])
        node.exp_entropy = -np.sum(prob * np.log2(prob))
        node.data_len = len(Y_train)
        node.hidden_y_for_prune = np.argmax(np.bincount(Y_train))
        node.pivot_data_num = np.bincount(Y_train)[node.hidden_y_for_prune]
        # 1. 结束条件：若 D 中所有实例属于同一类，决策树成单节点树，直接返回
        if np.any(np.bincount(Y_train) == len(Y_train)):
            node.y = Y_train[0]
            return
        # 2. 结束条件：若 A 为空，则返回单结点树 T，标记类别为样本默认输出最多的类别
        # 有可能特征完全相同但标签不同
        if feature_ids.size == 0:
            node.y = np.argmax(np.bincount(Y_train))
            return
        # 3. 计算特征集 A 中各特征对 D 的信息增益，选择信息增益最大的特征 A_g
        max_info_gain, g = self._feature_choose_standard(X_train, Y_train, node.exp_entropy)
        # 4. 结束条件：如果 A_g 的信息增益小于阈值 epsilon，决策树成单节点树，直接返回
        if max_info_gain <= self.epsilon:
            node.y = np.argmax(np.bincount(Y_train))
            return
        # 5. 对于 A_g 的每一可能值 a_i，依据 A_g = a_i 将 D 分割为若干非空子集 D_i，将当前结点的标记设为样本数最大的 D_i 对应
            # 的类别，即对第 i 个子节点，以 D_i 为训练集，以 A - {A_g} 为特征集，递归调用以上步骤，得到子树 T_i，返回 T_i
        node.split_feature = feature_ids[g]
        # node.split_feature_name = feature_names[g]
      
        a_cls = np.bincount(X_train[:, g])
        new_X_train, feature_ids = np.hstack((X_train[:, 0:g], X_train[:, g+1:])), np.hstack((feature_ids[0:g], feature_ids[g+1:]))
        for k in range(len(a_cls)):
            a_row_idxs = np.argwhere(X_train[:, g] == k).T[0].T
            # 非空子集
            if len(a_row_idxs) > 0:
                child = Node(node, pre_split_feature_x=k) #, feature_x_names[g][k])
                node.append(child)
                X_train_child, Y_train_child = new_X_train[a_row_idxs, :], Y_train[a_row_idxs]
                self._train(X_train_child, Y_train_child, child, feature_ids)

    def _feature_choose_standard(self, X_train, Y_train, entropy):
        row, col = X_train.shape
        # prob = self._cal_prob(Y_train)
        # prob = np.array([a if 0 < a <= 1 else 1 for a in prob])
        # entropy = -np.sum(prob * np.log2(prob))

        max_info_gain_ratio = None
        g = None
        for j in range(col):
            a_cls = np.bincount(X_train[:, j])
            condition_entropy = 0
            for k in range(len(a_cls)):
                a_row_idxs = np.argwhere(X_train[:, j] == k)
                # H(D)
                prob = self._cal_prob(Y_train[a_row_idxs].T[0])
                prob = np.array([a if 0 < a <= 1 else 1 for a in prob])
                H_D = -np.sum(prob * np.log2(prob))
                # H(D|A)=SUM(p_i * H(D|A=a_i))
                condition_entropy += a_cls[k] / np.sum(a_cls) * H_D
            feature_choose_std = entropy - condition_entropy
            if max_info_gain_ratio is None or max_info_gain_ratio < feature_choose_std:
                max_info_gain_ratio = feature_choose_std
                g = j
        return max_info_gain_ratio, g

    def _cal_prob(self, D):
        statistic = np.bincount(D)
        prob = statistic / np.sum(statistic)
        return prob
    def _visualization_dfs(self, node, layer=0):
        prefix = '\n' if layer else ''
        output_str = [prefix + ' ' * 4 * layer, '%r+%r ' % (node.y, node.split_feature)]
        if not node.child:
            return ''.join(output_str)
        for child in node.child:
            output_str.append(self._visualization_dfs(child, layer=layer + 1))
        return ''.join(output_str)
    def visualization(self):
        return self._visualization_dfs(self.tree)

class DTreeC45(DTreeID3):

    def _feature_choose_standard(self, X_train, Y_train, entropy):
        row, col = X_train.shape
        prob = self._cal_prob(Y_train)
        prob = np.array([a if 0 < a <= 1 else 1 for a in prob])
        entropy = -np.sum(prob * np.log2(prob))
        max_info_gain_ratio = None
        g = None
        for j in range(col):
            a_cls = np.bincount(X_train[:, j])
            condition_entropy = 0
            for k in range(len(a_cls)):
                a_row_idxs = np.argwhere(X_train[:, j] == k)
                # H(D) = -SUM(p_i * log(p_i))
                prob = self._cal_prob(Y_train[a_row_idxs].T[0]) # 转置之后，label到了[0]的位置
                prob = np.array([a if 0 < a <= 1 else 1 for a in prob])
                H_D = -np.sum(prob * np.log2(prob))
                # H(D|A)=SUM(p_i * H(D|A=a_i))
                condition_entropy += a_cls[k] / np.sum(a_cls) * H_D
            a_prob = self._cal_prob(X_train[:, j])
            a_prob = np.array([a if 0 < a <= 1 else 1 for a in a_prob])
            Split_info = -np.sum(a_prob * np.log2(a_prob))
            feature_choose_std = (entropy - condition_entropy) / Split_info
            if max_info_gain_ratio is None or max_info_gain_ratio < feature_choose_std:
                max_info_gain_ratio = feature_choose_std
                g = j
        return max_info_gain_ratio, g
    
    def get_prune_node_que(self, RootNode):
        prune_node_que = queue.Queue()
        traverse_node_que = queue.Queue()
        traverse_node_que.put(RootNode)
        while not traverse_node_que.empty():
            cur_node = traverse_node_que.get()
            if cur_node.if_children_leaf():
                prune_node_que.put(cur_node)
            else:
                for child in cur_node.child:
                    traverse_node_que.put(child)
        return prune_node_que
    def prune(self, alpha=0.2, kind=None):
        if kind is None:
            self._prune(self.tree, alpha)
        elif kind == "MEP":
            self._MEP_prune(self.tree)

    def _prune(self, RootNode, alpha=0.2):
        prune_node_que = self.get_prune_node_que(RootNode) #可能剪枝结点的队列
        while not prune_node_que.empty():
            prune_node = prune_node_que.get()
            C_cur = prune_node.data_len * prune_node.exp_entropy + 1 * alpha
            # print("C_cur is ", C_cur)
            C_children = 0.0
            for child in prune_node.child:
                C_children += (child.data_len * child.exp_entropy + 1 * alpha)
            # print("C_children is ", C_children)
            if C_children >= C_cur:
                # print("prune node with split feature :" , datalabel[prune_node.split_feature])
                prune_node.child = [] #把当前节点的儿子剪掉，使得当前结点成为叶子结点
                prune_node.y = prune_node.hidden_y_for_prune #当前节点成为了新的叶子节点，要给当前结点的y赋值
                if prune_node.parent is not None and prune_node.parent.if_children_leaf(): # 剪枝之后，如果当前结点的父节点成了可能剪枝的结点（所有的子节点都是叶子节点），则把当前节点的父亲加入可能剪枝结点的队列
                    prune_node_que.put(prune_node.parent)
    def _MEP_prune(self, RootNode):
        prune_node_que = self.get_prune_node_que(RootNode) #可能剪枝结点的队列
        while not prune_node_que.empty():
            prune_node = prune_node_que.get()
            C_cur = (prune_node.data_len - prune_node.pivot_data_num + self.label_num - 1) / (prune_node.data_len + self.label_num)
            # print("C_cur is ", C_cur)
            C_children = 0.0
            for child in prune_node.child:
                C_children += (child.data_len - child.pivot_data_num + self.label_num - 1) / (child.data_len + self.label_num) * child.data_len / prune_node.data_len
            if C_children >= C_cur:
                # print("prune node with split feature :" , datalabel[prune_node.split_feature])
                prune_node.child = [] #把当前节点的儿子剪掉，使得当前结点成为叶子结点
                prune_node.y = prune_node.hidden_y_for_prune #当前节点成为了新的叶子节点，要给当前结点的y赋值
                if prune_node.parent is not None and prune_node.parent.if_children_leaf(): # 剪枝之后，如果当前结点的父节点成了可能剪枝的结点（所有的子节点都是叶子节点），则把当前节点的父亲加入可能剪枝结点的队列
                    prune_node_que.put(prune_node.parent)


class DTreeCART(DTreeID3):
    def _train(self, X_train, Y_train, node, feature_ids, K=None):
        self._train_helper(X_train, Y_train, node, feature_ids, K)

    def _train_helper(self, X_train, Y_train, node, feature_ids, K=None):
        # 1. 结束条件：若 D 中所有实例属于同一类，决策树成单节点树，直接返回
        if np.any(np.bincount(Y_train) == len(Y_train)):
            node.y = Y_train[0]
            return
        # 2. 与 ID3, C4.5 不一样, 不会直接去掉 A
        if X_train.size == 0:
            node.y = np.argmax(np.bincount(Y_train))
            return
        # 3. 与 ID3, C4.5 不一样, 不仅要确定最优切分特征，还要确定最优切分值
        if K is None:
            max_info_gain, g, v, left_idx, right_idx = self._feature_choose_standard(X_train, Y_train)
        else:
            max_info_gain, g, v, left_idx, right_idx = self._feature_choose_standard_randk(X_train, Y_train, K)
        # 即可能存在特征一样，但label不同的数据，此时max_info_gain为None
        if max_info_gain is None:
            node.y = np.argmax(np.bincount(Y_train))
            return

        # 4. 结束条件：如果 A_g 的基尼指数小于阈值 epsilon，决策树成单节点树，直接返回
        # if max_info_gain <= self.epsilon:
        #     print("finish less epsilon")
        #     print("max_info_gain is ", max_info_gain)
        #     print("X train is ")
        #     print(X_train)
        #     print("Y train is ")
        #     print(Y_train)
        #     node.y = np.argmax(np.bincount(Y_train))
        #     return
        # 5. 与 ID3, C4.5 不一样, 不yes len(a_cls) 叉树，而yes二叉树
        node.split_feature = feature_ids[g]
        node.cart_split_feature_x = v
      
        # left_idx 和 right_idx 分别表示根据最优划分点 分割出的左右子树的数据集
        idx_list = left_idx, right_idx
        for child_idx, row_idx in enumerate(idx_list):
            row_idx = row_idx.T[0].T
            # 因为CARTyes二分，而这里的k的取值也yes0、1，1表示no，0表示yes
            child = Node(node, pre_split_feature_x=child_idx) #, "yes" if k == 0 else 'no')
            node.append(child)
            X_train_child, Y_train_child = X_train[row_idx, :], Y_train[row_idx]
            self._train_helper(X_train_child, Y_train_child, child, feature_ids, K)

    def _feature_choose_standard(self, X_train, Y_train):
        row, col = X_train.shape
        min_gini, g, v, left_idx, right_idx = None, None, None, None, None
        for j in range(col):
            a_cls = np.bincount(X_train[:, j]) # [j]特征在整个数据集中不同的取值
            # 与 ID3, C4.5 不一样,不仅要确定最优切分特征，还要确定最优切分值
            for k in range(len(a_cls)):
                left_row_idxs, right_row_idxs = np.argwhere(X_train[:, j] == k), np.argwhere(X_train[:, j] != k)
                # 根据切分值划为两类，且必须要化成了两类
                if len(left_row_idxs) != 0 and len(right_row_idxs) != 0:
                # H(D) = -SUM(p_i * log(p_i))
                    left_prob, right_prob = self._cal_prob(Y_train[left_row_idxs].T[0]), self._cal_prob(Y_train[right_row_idxs].T[0])
                    left_gini, right_gini = 1 - np.sum(left_prob * left_prob), 1 - np.sum(right_prob * right_prob)
                    # H(D|A)=SUM(p_i * H(D|A=a_i))
                    
                    gini_DA = a_cls[k] / np.sum(a_cls) * left_gini + (1 - a_cls[k] / np.sum(a_cls)) * right_gini
                    # if gini_DA == 0.0:

                    if min_gini is None or min_gini > gini_DA:
                        min_gini, g, v, left_idx, right_idx = gini_DA, j, k, left_row_idxs, right_row_idxs

        return min_gini, g, v, left_idx, right_idx
    def _feature_choose_standard_randk(self, X_train, Y_train, K):
        row, col = X_train.shape
        min_gini, g, v, left_idx, right_idx = None, None, None, None, None
        for j in range(col):
            a_cls = np.bincount(X_train[:, j]) # [j]特征在整个数据集中不同的取值
            # 与 ID3, C4.5 不一样,不仅要确定最优切分特征，还要确定最优切分值
            sample_feature_idx = np.random.permutation(len(a_cls))[:int(len(a_cls) * K)]
            # sample_feature_idx = list(range(len(a_cls)))[:int(len(a_cls) * K)]
            for cur_feature in sample_feature_idx:
                left_row_idxs, right_row_idxs = np.argwhere(X_train[:, j] == cur_feature), np.argwhere(X_train[:, j] != cur_feature)
                # 根据切分值划为两类，且必须要化成了两类
                if len(left_row_idxs) != 0 and len(right_row_idxs) != 0:
                # H(D) = -SUM(p_i * log(p_i))
                    left_prob, right_prob = self._cal_prob(Y_train[left_row_idxs].T[0]), self._cal_prob(Y_train[right_row_idxs].T[0])
                    left_gini, right_gini = 1 - np.sum(left_prob * left_prob), 1 - np.sum(right_prob * right_prob)
                    # H(D|A)=SUM(p_i * H(D|A=a_i))
                    
                    gini_DA = a_cls[cur_feature] / np.sum(a_cls) * left_gini + (1 - a_cls[cur_feature] / np.sum(a_cls)) * right_gini
                    # if gini_DA == 0.0:

                    if min_gini is None or min_gini > gini_DA:
                        min_gini, g, v, left_idx, right_idx = gini_DA, j, cur_feature, left_row_idxs, right_row_idxs

        return min_gini, g, v, left_idx, right_idx
        

    def predict(self, X):
        n = X.shape[0]
        Y = np.zeros(n)
        for i in range(n):
            Y[i] = self.tree.predict_classification(X[i, :], kind="CART")
        return Y

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
    if args.draw_trees:
        draw_row = args.sample_name
        for_draw_sets = data_sets[:draw_row, :]
        for_draw_sets_encode = np.array([[feature2id[j][for_draw_sets[i, j]] for j in range(col_)] for i in range(draw_row)])
        for_draw_X_t, for_draw_Y_t = for_draw_sets_encode[:, :-1], for_draw_sets_encode[:, -1]
        ID3_model_for_draw = DTreeID3()
        ID3_model_for_draw.fit(for_draw_X_t, for_draw_Y_t, len(feature2id[-1].values()))
        print('=' * 20 + ID3_model_for_draw.__class__.__name__ + '=' * 20)   
        print('\n<Label Groundtruth>')
        print(for_draw_Y_t)
        print('\n<Label Output>')
        ID3_pred_y = ID3_model_for_draw.predict(for_draw_X_t).astype(int)
        print(ID3_pred_y)
        print("ID3 Train set acc : ", count_acc(for_draw_Y_t, ID3_pred_y))
        draw_tree.ID3_Tree(ID3_model_for_draw.tree, "ID3-Tree.png")

        C45_model_for_draw = DTreeC45()
        C45_model_for_draw.fit(for_draw_X_t, for_draw_Y_t, len(feature2id[-1].values()))
        print('=' * 20 + C45_model_for_draw.__class__.__name__ + '=' * 20)   
        print('\n<Label Groundtruth>')
        print(for_draw_Y_t)
        print('\n<Label Output>')
        C45_pred_y = C45_model_for_draw.predict(for_draw_X_t).astype(int)
        print(C45_pred_y)
        draw_tree.C45_Tree(C45_model_for_draw.tree, "C4.5-Tree.png")
        C45_model_with_MEP_for_draw = copy.deepcopy(C45_model_for_draw)
        print("Before original prune!")
        print("C4.5 Train set acc : ", count_acc(for_draw_Y_t, C45_pred_y))
        C45_model_for_draw.prune(alpha=args.alpha)
        print("after original prune, C4.5 Train set acc : ", count_acc(for_draw_Y_t, C45_model_for_draw.predict(for_draw_X_t).astype(int)))
        C45_model_with_MEP_for_draw.prune(kind="MEP")
        print("after MEP prune, C4.5 Train set acc : ", count_acc(for_draw_Y_t, C45_model_with_MEP_for_draw.predict(for_draw_X_t).astype(int)))
        draw_tree.C45_Tree(C45_model_for_draw.tree, "C4.5-Tree after original prune.png")
        draw_tree.C45_Tree(C45_model_with_MEP_for_draw.tree, "C4.5-Tree after MEP prune.png")

        #CART
        CART_model_for_draw = DTreeCART()
        CART_model_for_draw.fit(for_draw_X_t, for_draw_Y_t, len(feature2id[-1].values()))
        print('=' * 20 + CART_model_for_draw.__class__.__name__ + '=' * 20)   
        print('\n<Label Groundtruth>')
        print(for_draw_Y_t)
        print('\n<Label Output>')
        CART_pred_y = CART_model_for_draw.predict(for_draw_X_t).astype(int)
        print(CART_pred_y)
        print("CART Train set acc : ", count_acc(for_draw_Y_t, CART_pred_y))
        draw_tree.CART_Tree(CART_model_for_draw.tree, "CART-Tree.png")


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
    # ID3
    ID3_model = DTreeID3()
    ID3_start_time = time.time()
    ID3_model.fit(train_X_t, train_Y_t, len(feature2id[-1].values()))
    ID3_end_time = time.time()
    print('=' * 20 + ID3_model.__class__.__name__ + '=' * 20)   
    print("ID3 Build time: ", ID3_end_time - ID3_start_time)
    print("ID3 Train set acc : ", count_acc(train_Y_t, ID3_model.predict(train_X_t).astype(int)))
    ID3_start_time = time.time()
    ID3_pred_y = ID3_model.predict(test_X_t).astype(int)
    ID3_end_time = time.time()
    print("ID3 Test time: ", ID3_end_time - ID3_start_time)
    print("ID3 Test set acc : ", count_acc(test_Y_t, ID3_pred_y))
    
    #C4.5
    C45_model = DTreeC45()
    C45_start_time = time.time()
    C45_model.fit(train_X_t, train_Y_t, len(feature2id[-1].values()))
    C45_end_time = time.time()
    print('=' * 20 + C45_model.__class__.__name__ + '=' * 20)   
    print("C4.5 Build time: ", C45_end_time - C45_start_time)
    C45_model_for_MEP = copy.deepcopy(C45_model)
    # print("Before original prune, C4.5 Train set acc : ", count_acc(train_Y_t, C45_model.predict(train_X_t).astype(int)))
    C45_start_time = time.time()
    C45_pred_y = C45_model.predict(test_X_t).astype(int)
    C45_end_time = time.time()
    print("C4.5 Test time: ", C45_end_time - C45_start_time)
    print("Before original prune, C4.5 Test set acc : ", count_acc(test_Y_t, C45_pred_y))
    C45_model.prune(alpha=args.alpha)
    # print("After original prune, C4.5 Train set acc : ", count_acc(train_Y_t, C45_model.predict(train_X_t).astype(int)))
    print("After original prune, C4.5 Test set acc : ", count_acc(test_Y_t, C45_model.predict(test_X_t).astype(int)))

    C45_model_for_MEP.prune(kind="MEP")
    # print("After MEP prune, C4.5 Train set acc : ", count_acc(train_Y_t, C45_model_for_MEP.predict(train_X_t).astype(int)))
    print("After MEP prune, C4.5 Test set acc : ", count_acc(test_Y_t, C45_model_for_MEP.predict(test_X_t).astype(int)))

    # CART
    CART_model = DTreeCART()
    CART_start_time = time.time()
    CART_model.fit(train_X_t, train_Y_t, len(feature2id[-1].values()))
    CART_end_time = time.time()
    print('=' * 20 + CART_model.__class__.__name__ + '=' * 20) 
    print("CART Build time: ", CART_end_time - CART_start_time)  
    print("CART Train set acc : ", count_acc(train_Y_t, CART_model.predict(train_X_t).astype(int)))
    CART_start_time = time.time()
    CART_pred_y = CART_model.predict(test_X_t).astype(int)
    CART_end_time = time.time()
    print("CART Test time: ", CART_end_time - CART_start_time) 
    print("CART Test set acc : ", count_acc(test_Y_t, CART_pred_y))


