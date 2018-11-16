# -*- coding: utf-8 -*-
import pandas as pd 
from sklearn import metrics
import numpy as np
data_raw = pd.read_csv('data_2_0.txt',index_col = 0)
#data_df = data_raw.apply(preprocessing.LabelEncoder().fit_transform)
data_df = data_raw
#%%
class decision_tree():
    def __init__(self):
        self.modle = {}
        self.x = []
        self.y = []
        self.data_df = []
        self.use_round = True
        self.feature_names = []
        self.accuracy = 0
    def fit(self, x,y,use_round = True):
        self.x = x
        self.y = y
        self.data_df = x.join(y)
        self.use_round = use_round
        self.feature_names = self.data_df.columns.tolist()
        self.modle = self.create_tree(dataset = self.data_df , 
                                 optional_feature_names= self.feature_names
                                 ,use_round = use_round)
        _,self.accuracy = self.predicte(self.x,self.y)
#     创建树   
    def create_tree(self,dataset,optional_feature_names,use_round = True):
        label = dataset.iloc[:,-1]
        if len(dataset)==0:
            return '是'
        if len(label.value_counts())==1:
            # 若全为同一标签，则返回
            return label.iloc[0]
        if len(optional_feature_names)== 0 or data_df.apply(lambda x: (x==x[1]).all()).all():   
            # 若没有可选属性, 或者dataset在可选属性集上取值相同，则返回最多取值的标签
            return label.value_counts(sort = True).iloc[0]        
        bestFeat,gain = self.choose_best_feature(dataset,use_round = use_round)  #选择最优特征    
        myTree = {bestFeat:{}}	
        optional_feature_names.remove(bestFeat)  #  删除已经使用特征标签
        featValues = data_df[bestFeat].unique()
        for value in featValues:									#遍历特征，创建决策树。	
            myTree[bestFeat][value] = self.create_tree(
                    dataset[ dataset[bestFeat ]==value ][optional_feature_names],
                    optional_feature_names.copy())
        return myTree    
     # 计算信息熵
    def calculate_Ent(self,counts,use_round = False):
        all_sum = counts.sum().sum()
        column_sum = counts.sum(axis=0)
        counts_p = counts.apply(lambda x:x/column_sum if isinstance(x,int) else x/x.sum())
        counts_Ent = counts_p.apply(lambda x: -x*np.log2(x))
        Ent = counts_Ent.sum(axis=0) * (column_sum/all_sum)
        if use_round:
            Ent = np.round(Ent.sum(),3)
        else:
            Ent = Ent.sum()
        return Ent
    def choose_best_feature(self,dataset,use_round = True):
        label_name = dataset.columns[-1]
        label_counts = dataset.iloc[:,-1].value_counts()
        cur_Ent = self.calculate_Ent(label_counts,use_round = use_round)
        gain = []
        for columni in dataset.columns[:-1]:
            feature_label_counts = dataset[label_name].groupby(
                    [dataset[label_name],dataset[columni]]).count()
            feature_label_counts = feature_label_counts.fillna(1e-6).unstack()
            new_Ent = self.calculate_Ent(feature_label_counts,use_round = use_round)
            gain.append(cur_Ent-new_Ent)
        if use_round:
            gain = np.round(gain,3) 
        return dataset.columns[np.argmax(gain)],gain
    def predicte(self,x,y = None):
        pre_labels = []
        accuracy = 0
        for i in range(len(x)):
            optional_feature_names = x.columns.tolist()
            xi = x.iloc[i,:]
            desiced_model = self.modle
            while len(optional_feature_names)>0:
                desiced_feature = list(desiced_model.keys())[0] 
                desiced_feature_value = xi[desiced_feature]
                optional_feature_names.remove(desiced_feature)
                desiced_model = desiced_model[desiced_feature][desiced_feature_value]
                if not isinstance(desiced_model,dict):
                    pre_labels.append(desiced_model)
                    break
        if not y is None:
            accuracy = metrics.accuracy_score(y, pre_labels)
        return pre_labels,accuracy
#%%  模型使用
        
decision_tree = decision_tree()
decision_tree.fit(data_df.iloc[:,:-1] ,data_df.iloc[:,-1])
complete_modle_tree = decision_tree.modle
print("===========================================\n")
print("使用全部数据训练:\n得到模型为:\n{:}\n".format(complete_modle_tree))
#%%  绘制决策树

import treePlotter
#[treePlotter] Source ：https://github.com/WordZzzz/ML/blob/master/Ch03/treePlotter.py
fig = treePlotter.createPlot(complete_modle_tree)
fig.savefig("tree.png")




#%%  模型实际使用测试
#from sklearn.model_selection import train_test_split
#print("===========================================\n")
#print("使用 70% 数据进行训练:\n")
#X_train,X_test,y_train,y_test = train_test_split( data_df.iloc[:,:-1] ,
#                   data_df.iloc[:,-1],test_size=0.3, random_state = 1234)
#decision_tree.fit(x= X_train, y = y_train)
#trainTest_modle_tree = decision_tree.modle
#acc_train = decision_tree.accuracy
#_,acc_test = decision_tree.predicte(x= X_test, y = y_test)
#print("训练集准确率: {0:.3f}  测试准确率:{1:.3f}".format( acc_train,acc_test))
#print("模型为:\n{:}".format(trainTest_modle_tree))
#treePlotter.createPlot(trainTest_modle_tree)