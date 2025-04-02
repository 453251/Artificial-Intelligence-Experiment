from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv
import graphviz

Dtree = open('watermelon_data3.0.csv', 'r')
reader = csv.reader(Dtree)
"""
色泽 1-3代表 浅白 青绿 乌黑 
根蒂 1-3代表 稍蜷 蜷缩 硬挺 
敲声 1-3代表 清脆 浊响 沉闷 
纹理 1-3代表 清晰 稍糊 模糊 
脐部 1-3代表 平坦 稍凹 凹陷 
触感 1-2代表 硬滑 软粘
好瓜 1代表 是 0 代表 不是
"""

# 获取第一行数据
headers = reader.__next__()
print(headers)
# 特征和标签列表
featureList = []
labelList = []

for row in reader:
    labelList.append(row[-1])
    rowDict = {}
    for i in range(1, len(row) - 3):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
print(featureList)
# 将特征列表转换为01表示
vec = DictVectorizer()
x_data = vec.fit_transform(featureList).toarray()
print("x_data: " + str(x_data))
# 将标签列表转换为01表示
lb = preprocessing.LabelBinarizer()
y_data = lb.fit_transform(labelList)
print("y_data: " + str(y_data))

# 创建决策树模型
model = tree.DecisionTreeClassifier(criterion='entropy')
# 输入数据建立模型
model.fit(x_data, y_data)

# 测试
x_test = x_data[0]
predict = model.predict(x_test.reshape(1, -1))
print("predict: " + str(predict))

# 导出决策树
dot_data = tree.export_graphviz(model,
                                out_file=None,
                                feature_names=vec.get_feature_names_out(),
                                class_names=lb.classes_,
                                filled=True,
                                rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('Tree')