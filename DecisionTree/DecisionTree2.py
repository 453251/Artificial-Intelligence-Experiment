import pandas as pd
from collections import Counter
from math import log2
import random
from graphviz import Digraph
import copy


class DecisionTree:
    def __init__(self, filepath, validation_ratio=0.4, test_ratio=0.0):
        self.filepath = filepath
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.tree = None
        self.train_data = None
        self.validate_data = None
        self.test_data = None
        self.data = None

    # 读取数据集并离散化
    def read_dataset(self):
        df = pd.read_csv(self.filepath)
        df.drop(columns=['编号'], inplace=True)
        self.data = df

        # # 离散化连续值
        # for col in ['密度', '含糖率']:
        #     median = df[col].median()
        #     df[col] = df[col].apply(lambda x: f"高于{median:.3f}" if x >= median else f"低于{median:.3f}")

        return df

    # 分割数据集
    def train_test_split(self, df):
        indices = list(df.index)
        # random.seed(7)
        random.seed(4)
        random.shuffle(indices)

        split_point_test = int(len(df) * (1 - self.test_ratio))
        split_point_validate = int(len(df) * (1 - self.test_ratio - self.validation_ratio))
        train_indices = indices[:split_point_validate]
        validate_indices = indices[split_point_validate:split_point_test]
        test_indices = indices[split_point_test:]

        self.train_data = df.loc[train_indices].reset_index(drop=True)
        self.validate_data = df.loc[validate_indices].reset_index(drop=True)
        self.test_data = df.loc[test_indices].reset_index(drop=True)

    # 熵
    @staticmethod
    def entropy(y):
        freq = Counter(y)
        total = len(y)
        return -sum((count / total) * log2((count / total)) for count in freq.values())

    # 条件熵
    def conditional_entropy(self, x, y):
        total = len(y)
        return sum((x == val).sum() / total * self.entropy(y[x == val]) for val in set(x))

    # 信息增益
    def info_gain(self, x, y):
        return self.entropy(y) - self.conditional_entropy(x, y)

    def info_gain_continuous(self, x, y):
        """利用C4.5中的离散化技术（二分法）来进行离散化"""
        df = pd.DataFrame({'x': x, 'y': y})
        df = df.sort_values('x').reset_index(drop=True)

        best_gain = -1
        best_threshold = None

        for i in range(len(df) - 1):
            """将连续值进行排序，对排序后的连续值两两计算平均值，找出信息增益最大的平均值作为二分的阈值"""
            if df.loc[i, 'y'] != df.loc[i + 1, 'y']:
                t = (df.loc[i, 'x'] + df.loc[i + 1, 'x']) / 2
                y_left = df[df['x'] <= t]['y']
                y_right = df[df['x'] > t]['y']

                p_left = len(y_left) / len(y)
                p_right = len(y_right) / len(y)

                cond_entropy = p_left * self.entropy(y_left) + p_right * self.entropy(y_right)
                gain = self.entropy(y) - cond_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_threshold = t

        return best_gain, best_threshold

    def id3(self, data, attributes, target):
        """带预剪枝的id3"""
        # 取出标签列
        y = data[target]

        # 如果只有一个类别（全是好瓜或者全都不是）
        if len(set(y)) == 1:
            return '好瓜' if y.iloc[0] == '是' else '坏瓜'
        # 如果没有剩余属性了（即属性都用完了），还没判定出来，则选取一个出现的最多的
        if not attributes:
            return '好瓜' if y.mode()[0] == '是' else '坏瓜'
        best_attr = None
        best_gain = -1
        best_threshold = None
        is_continuous = False

        for attr in attributes:
            """找出最优的属性值（离散的或者连续的）"""
            if pd.api.types.is_numeric_dtype(data[attr]):
                gain, threshold = self.info_gain_continuous(data[attr], y)
                if gain > best_gain:
                    best_gain = gain
                    best_attr = attr
                    best_threshold = threshold
                    is_continuous = True
            else:
                gain = self.info_gain(data[attr], y)
                if gain > best_gain:
                    best_gain = gain
                    best_attr = attr
                    best_threshold = None
                    is_continuous = False

        if best_attr is None:
            return '好瓜' if y.mode()[0] == '是' else '坏瓜'

        tree = {}

        if is_continuous:
            # 离散的，根据二分阈值进行离散化
            tree[f"{best_attr} <= {best_threshold:.3f}"] = {}
            left_data = data[data[best_attr] <= best_threshold]
            right_data = data[data[best_attr] > best_threshold]

            for subset, label in [(left_data, "是"), (right_data, "否")]:
                if subset.empty:
                    tree[f"{best_attr} <= {best_threshold:.3f}"][label] = y.mode()[0]
                else:
                    sub_tree = self.id3(subset, [attr for attr in attributes if attr != best_attr], target)
                    # 预剪枝判断
                    temp_tree = copy.deepcopy(tree)
                    temp_tree[f"{best_attr} <= {best_threshold:.3f}"][label] = sub_tree
                    # 判断
                    y_true = self.validate_data[target]
                    y_true = y_true.apply(lambda x: '好瓜' if x == '是' else '坏瓜')

                    y_pred_tree = self.validate_data.apply(lambda row: self.predict_one(row, temp_tree), axis=1)
                    acc_tree = (y_true == y_pred_tree).mean()

                    majority = '好瓜' if y.mode()[0] == '是' else '坏瓜'
                    acc_leaf = (y_true == majority).mean()
                    if acc_leaf >= acc_tree:
                        tree[f"{best_attr} <= {best_threshold:.3f}"][label] = '好瓜' if y.mode()[0] == '是' else '坏瓜'
                    else:
                        tree[f"{best_attr} <= {best_threshold:.3f}"][label] = sub_tree
        else:
            # 否则以离散的best_attribute为根节点构建决策树（子树），需要包含最佳属性（根结点），边（属性值），子树
            tree[best_attr] = {}
            for val in self.data[best_attr].unique():
                """判断该属性是否在best_attribute中存在，如果存在，则构建子树，如果不存在，则在当前数据集中挑选一个最常见的标签"""
                if val in data[best_attr].unique():
                    sub_data = data[data[best_attr] == val].drop(columns=[best_attr])
                    sub_tree = self.id3(sub_data, [attribute for attribute in attributes if attribute != best_attr],
                                        target)
                    if val in self.validate_data[best_attr].unique():
                        # 预剪枝判断
                        temp_tree = copy.deepcopy(tree)
                        temp_tree[best_attr][val] = sub_tree
                        # 判断
                        y_true = self.validate_data[target]
                        y_true = y_true.apply(lambda x: '好瓜' if x == '是' else '坏瓜')

                        y_pred_tree = self.validate_data.apply(lambda row: self.predict_one(row, temp_tree), axis=1)
                        acc_tree = (y_true == y_pred_tree).mean()

                        majority = '好瓜' if y.mode()[0] == '是' else '坏瓜'
                        acc_leaf = (y_true == majority).mean()
                        if acc_leaf >= acc_tree:
                            tree[best_attr][val] = '好瓜' if y.mode()[0] == '是' else '坏瓜'
                        else:
                            tree[best_attr][val] = sub_tree
                    else:
                        tree[best_attr][val] = sub_tree
                else:
                    tree[best_attr][val] = '好瓜' if y.mode()[0] == '是' else '坏瓜'
        return tree

    def fit(self):
        df = self.read_dataset()
        self.train_test_split(df)

        attributes = self.train_data.columns[:-1].tolist()
        target = self.train_data.columns[-1]
        self.tree = self.id3(self.train_data, attributes, target)

    def visualize(self, filename):
        dot = Digraph(comment='决策树')
        dot.attr(fontname="SimHei")  # 设置全局字体（支持中文）

        node_id = [0]

        def add_nodes_edges(tree, parent=None, edge_label=""):
            # 判断是否为叶子结点
            if isinstance(tree, dict):
                # node为当前树（子树）的根结点，也就是当前最佳属性，然后包含若干条边（属性值）和子树
                for node, subtree in tree.items():
                    curr_id = str(node_id[0])
                    dot.node(curr_id, node,
                             shape="box",
                             style="filled",
                             fillcolor="lightyellow",
                             fontname="SimHei")
                    # 如果有父结点，就画父结点到当前node的边，参数从父结点获取，因为显然是当子结点画出来之后，才能画父结点到子结点的边
                    if parent is not None:
                        dot.edge(parent, curr_id, label=edge_label, fontname="SimHei")
                    node_id[0] += 1
                    # 画子树，子树的父结点为当前结点（即curr_id）
                    for val, child in subtree.items():
                        add_nodes_edges(child, curr_id, str(val))
            else:
                leaf_id = str(node_id[0])
                dot.node(leaf_id, str(tree),
                         shape="ellipse",
                         style="filled",
                         fillcolor="palegreen",
                         fontname="SimHei")
                # 叶子结点
                if parent is not None:
                    dot.edge(parent, leaf_id, label=edge_label, fontname="SimHei")
                node_id[0] += 1

        add_nodes_edges(self.tree)
        dot.render(filename, view=True, format='png')

    def post_prune(self, tree, data, target):
        # 如果是叶子结点，直接返回
        if not isinstance(tree, dict):
            return tree

        root = next(iter(tree))
        branches = tree[root]

        # 连续属性格式处理
        if "<=" in root:
            attr, threshold = root.split(" <= ")
            threshold = float(threshold)
            data_left = data[data[attr] <= threshold]
            data_right = data[data[attr] > threshold]

            for label, subset in zip(["是", "否"], [data_left, data_right]):
                # 分支是个子树，进行剪枝判断
                if isinstance(branches[label], dict) and not subset.empty:
                    branches[label] = self.post_prune(branches[label], subset, target)

            # 剪枝判断
            y_true = data[target]
            y_true = y_true.apply(lambda x: '好瓜' if x == '是' else '坏瓜')

            y_pred_tree = data.apply(lambda row: self.predict_one(row, tree), axis=1)
            acc_tree = (y_true == y_pred_tree).mean()

            majority = y_true.mode()[0]
            acc_leaf = (y_true == majority).mean()

            if acc_leaf >= acc_tree:
                return majority
            else:
                return tree

        else:  # 离散属性
            for val in list(branches.keys()):
                subset = data[data[root] == val]
                if isinstance(branches[val], dict) and not subset.empty:
                    branches[val] = self.post_prune(branches[val], subset, target)

            y_true = data[target]
            y_true = y_true.apply(lambda x: '好瓜' if x == '是' else '坏瓜')
            y_pred_tree = data.apply(lambda row: self.predict_one(row, tree), axis=1)
            acc_tree = (y_true == y_pred_tree).mean()

            # 剪枝后的“叶子”准确率（用当前数据多数类来代替）
            majority = y_true.mode()[0]
            acc_leaf = (y_true == majority).mean()

            if acc_leaf >= acc_tree:
                return majority
            else:
                return tree

    def predict_one(self, row, tree):
        while isinstance(tree, dict):
            node = next(iter(tree))
            branches = tree[node]
            if "<=" in node:
                attr, threshold = node.split(" <= ")
                threshold = float(threshold)
                tree = branches["是"] if row[attr] <= threshold else branches["否"]
            else:
                # 取出当前最佳属性在row中的属性值
                val = row[node]
                # tree = branches.get(val, Counter(self.train_data.iloc[:, -1]).most_common(1)[0][0])
                # 容错处理：如果 val 不在 branches 里，用训练集中的多数类作为预测结果
                if val not in branches:
                    return Counter(self.train_data.iloc[:, -1]).most_common(1)[0][0]
                tree = branches[val]

        return tree

    def predict(self, x, y):
        y_pred = x.apply(lambda row: self.predict_one(row, self.tree), axis=1)
        y = y.apply(lambda x: '好瓜' if x == '是' else '坏瓜')
        return (y_pred == y).mean()


clf = DecisionTree("watermelon_3.csv")
clf.fit()
acc_train = clf.predict(clf.train_data, clf.train_data.iloc[:, -1])
acc_valid = clf.predict(clf.validate_data, clf.validate_data.iloc[:, -1])
# acc_test = clf.predict(clf.test_data, clf.test_data.iloc[:, -1])
print('决策树(预剪枝)')
print(f"训练集准确率: {acc_train:.2%}")
print(f"验证集准确率: {acc_valid:.2%}")
# print(f"测试集准确率: {acc_test:.2%}")
clf.visualize('decision_tree_with_pre_prune')
