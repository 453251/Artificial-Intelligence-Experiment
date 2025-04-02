import pandas as pd
from collections import Counter
from math import log2
import random
from graphviz import Digraph


class DecisionTree:
    def __init__(self, filepath, test_ratio=0):
        self.filepath = filepath
        self.test_ratio = test_ratio
        self.tree = None
        self.train_data = None
        self.test_data = None

    # 读取数据集并离散化
    def read_dataset(self):
        df = pd.read_csv(self.filepath)
        df.drop(columns=['编号'], inplace=True)

        # 离散化连续值
        for col in ['密度', '含糖率']:
            median = df[col].median()
            df[col] = df[col].apply(lambda x: f"高于{median:.3f}" if x >= median else f"低于{median:.3f}")

        return df

    # 分割数据集
    def train_test_split(self, df):
        indices = list(df.index)
        random.shuffle(indices)

        split_point = int(len(df) * (1 - self.test_ratio))
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        self.train_data = df.loc[train_indices].reset_index(drop=True)
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

    def id3(self, data, attributes, target):
        # 取出标签列
        y = data[target]

        # 如果只有一个类别（全是好瓜或者全都不是）
        if len(set(y)) == 1:
            return '好瓜' if y.iloc[0] == '是' else '坏瓜'
        # 如果没有剩余属性了（即属性都用完了），则选取一个出现的最多的
        if not attributes:
            return '好瓜' if y.mode()[0] == '是' else '坏瓜'

        # 计算信息增益
        gains = {attribute: self.info_gain(data[attribute], y) for attribute in attributes}
        # 选取信息增益最大的
        best_attribute = max(gains, key=gains.get)

        # 以best_attribute为根节点构建决策树（子树），需要包含最佳属性（根结点），边（属性值），子树
        tree = {best_attribute: {}}
        for val in self.train_data[best_attribute].unique():
            """判断该属性是否在best_attribute中存在，如果存在，则构建子树，如果不存在，则在当前数据集中挑选一个最常见的标签"""
            if val in data[best_attribute].unique():
                sub_data = data[data[best_attribute] == val].drop(columns=[best_attribute])
                sub_tree = self.id3(sub_data, [attribute for attribute in attributes if attribute != best_attribute],
                                    target)
                tree[best_attribute][val] = sub_tree
            else:
                tree[best_attribute][val] = '好瓜' if y.mode()[0] == '是' else '坏瓜'
        return tree

    def fit(self):
        df = self.read_dataset()
        self.train_test_split(df)

        attributes = self.train_data.columns[:-1].tolist()
        target = self.train_data.columns[-1]
        self.tree = self.id3(self.train_data, attributes, target)

    def visualize(self):
        dot = Digraph(comment='决策树')
        dot.attr(fontname="SimHei")  # 设置全局字体（支持中文）

        node_id = [0]

        def add_nodes_edges(tree, parent=None, edge_label=""):
            if isinstance(tree, dict):
                for node, subtree in tree.items():
                    curr_id = str(node_id[0])
                    dot.node(curr_id, node,
                             shape="box",
                             style="filled",
                             fillcolor="lightyellow",
                             fontname="SimHei")
                    if parent is not None:
                        dot.edge(parent, curr_id, label=edge_label, fontname="SimHei")
                    node_id[0] += 1
                    for val, child in subtree.items():
                        add_nodes_edges(child, curr_id, str(val))
            else:
                leaf_id = str(node_id[0])
                dot.node(leaf_id, str(tree),
                         shape="ellipse",
                         style="filled",
                         fillcolor="palegreen",
                         fontname="SimHei")
                dot.edge(parent, leaf_id, label=edge_label, fontname="SimHei")
                node_id[0] += 1

        add_nodes_edges(self.tree)
        dot.render('decision_tree_violently', view=True, format='png')


clf = DecisionTree("watermelon_3.csv")
clf.fit()
clf.visualize()
