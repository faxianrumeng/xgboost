'''Xgboost回归算法'''
import numpy as np
import pandas as pd
import threading
#import logging
#logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
import  os
import math
import time
import random

# 构造一个关于cart树的节点的类
class   Ctft():
    def __init__(self, data, parent, left, right, array_data, best_attri=None, weight=None):
        self.array_data = array_data     #
        self.data = data                 # 生成左右子树的分裂值
        self.best_attri = best_attri     #
        self.parent = parent             #父节点
        self.leftson = left              #左子树
        self.rightson = right            #右子树
        self.weight = weight             #每个节点上的权重

# Xgboost类
class Xgboost():
   def __init__(self, Train_set, Validation_set, Lambda):   #Train_set 是监督集，是一个nparray对象，每行最后一列是对应的标签数据
       self.Lambda = Lambda
       self.Gamma = 0
       self.train_set = Train_set
       self.validation_set = Validation_set
       self.subset = []
       self.psy = self.train_set      #每次迭代后生成为下次迭代产生的新的训练数据
       self.col_length = len(self.train_set[0,:])-1     #属性个数
       self.tree_height = 6           #每棵树的最大高度，一般可以默认int(np.log(len(self.train_set)))
       self.segma = 0.1
       self.tree_num = 0
       self.weight = []
       for i in range(len(self.psy)):
           self.psy[i, -1] = -1*self.psy[i, -1]
       self.ctftlist = {}        #xgboost模型构建
       self.tree_func(1)
       self.MSE, self.validation_set = self.xgboost_vertification(self.validation_set, i+1)     #计算均方差

   # 生成每个cart决策树的分类集合及叶子权重，分裂点
   def tree_func(self,paraname):###para is the atrribution name
       self.ctftlist[str(paraname)] = []
       self.ctftlist[str(paraname)].append(Ctft(
                                                data = None,
                                                parent=None,
                                                left=None,
                                                right=None,
                                                array_data=self.psy
                                                ))
       self.psy = None
       for tr in range(0, self.tree_height-1):
           ctftlist_len = len(self.ctftlist[str(paraname)])
           sampling_attri = np.random.choice(self.col_length,
                            int(np.log(self.col_length) / np.log(2.0))) ###随机选择
           split, best_attri = self.cart_classify(
                                self.ctftlist[str(paraname)][0+2*tr].array_data,
                                                                 sampling_attri)
           sort_array = self.sort_attribution(
                                self.ctftlist[str(paraname)][0+2*tr].array_data,
                                                                     best_attri)
           Right, Left = self.son_set(sort_array, best_attri, split)
           del sort_array
           if len(Left) != 0 and tr < self.tree_height - 2:
               self.ctftlist[str(paraname)][0+2*tr].data = split
               self.ctftlist[str(paraname)][0+2*tr].leftson = ctftlist_len+1
               self.ctftlist[str(paraname)][0+2*tr].rightson = ctftlist_len
               self.ctftlist[str(paraname)][0 + 2 * tr].best_attri =best_attri
               Weight= self.grad_compute(Right, best_attri)
               self.ctftlist[str(paraname)].append(Ctft(data=None,
                                                        parent=0 + 2 * tr,
                                                        left=None,
                                                        right=None,
                                                        array_data=None,
                                                        best_attri=best_attri,
                                                        weight=Weight))
               self.ctftlist[str(paraname)].append(Ctft(data=None,
                                                        parent=0+2*tr,
                                                        left=None,
                                                        right=None,
                                                        best_attri=best_attri,
                                                        array_data=Left))
               del self.ctftlist[str(paraname)][0 + 2 * tr].array_data
           elif len(Left) == 0:
               self.ctftlist[str(paraname)][0 + 2 * tr]. best_attri = best_attri
               self.ctftlist[str(paraname)][0+2*tr].weight = self.grad_compute(Right, best_attri)
               del self.ctftlist[str(paraname)][0+2*tr].array_data
               break
           elif len(Left) != 0 and tr == self.tree_height - 2:
               self.ctftlist[str(paraname)][0 + 2 * tr].data = split
               self.ctftlist[str(paraname)][0 + 2 * tr].leftson = ctftlist_len
               self.ctftlist[str(paraname)][0 + 2 * tr].rightson = ctftlist_len + 1
               self.ctftlist[str(paraname)][0 + 2 * tr].best_attri = best_attri
               Weight = self.grad_compute(Left, best_attri)
               self.ctftlist[str(paraname)].append(Ctft(data=None,
                                                        parent=0+2*tr,
                                                        left=None,
                                                        right=None,
                                                        array_data=None,
                                                        weight=Weight,
                                                        best_attri=best_attri))
               Weight = self.grad_compute(Right, best_attri)
               self.ctftlist[str(paraname)].append(Ctft(data=None,
                                                        parent=0+2*tr,
                                                        left=None,
                                                        right=None,
                                                        array_data=None,
                                                        weight=Weight,
                                                        best_attri=best_attri))
               del self.ctftlist[str(paraname)][0 + 2 * tr].array_data
   # 寻找cart树结点的分裂点
   def cart_classify(self, Set, key_word_set):     #key_word_set is int
       Max=-1000000000.00000
       Split=0.0
       best_attri=0
       for i in key_word_set:
           Sort_array = self.sort_attribution(Set, i)
           Temp_list = self.histogram_data(Sort_array, i)
           tmp,TMP_Max = self.split_finding(Temp_list, Sort_array, i)
           if TMP_Max>Max:
               Max = TMP_Max
               Split = Temp_list[tmp]
               best_attri = i
       return  Split, best_attri
   # 判断预测值的权重
   def judge_func(self, obj, location, key_word, address, f):
       ctftlist_data = self.ctftlist[str(key_word)][address].data
       obj_best_attri = obj[location,self.ctftlist[str(key_word)][address].best_attri]
       ctftlist_weight = self.ctftlist[str(key_word)][address].weight
       if ctftlist_data is not None and ctftlist_data <= obj_best_attri and ctftlist_weight is None:
          f=self.judge_func(obj,location,key_word,self.ctftlist[str(key_word)][address].leftson,f)
       elif ctftlist_data is not None and ctftlist_data > obj_best_attri and ctftlist_weight is None:
          f=self.judge_func(obj, location,key_word, self.ctftlist[str(key_word)][address].leftson,f)
       elif ctftlist_weight is not None:
          f=ctftlist_weight * obj_best_attri
       return f
   # 验证函数
   def xgboost_vertification(self, input_set, Tree_num):##input_set输入集
       predict_record=np.zeros(len(input_set))
       temp_data=0
       for i in range(0,len(input_set)):
             temp_data=self.judge_func(input_set, i,1, 0, temp_data)
             predict_record[i]=predict_record[i]+temp_data
       mse=np.sqrt(sum((input_set[:,-1]-predict_record)**2))
       for k in range(0,len(input_set)):
           input_set[k,-1]=input_set[k,-1]-predict_record[k]
       return mse,input_set
# 按照属性大小排序
   def sort_attribution(self, Narray, keyword):
       sorted_array = np.array(sorted(Narray, key=lambda x:x[keyword]))
       return  sorted_array
# 构造近似属性直方图数据
   def histogram_data(self, Narray, Keyword):
       temp_list = []
       for i in range(0, 102, 2):
           temp_list.append(np.percentile(Narray[:, Keyword], i))
       temp_list = np.array(temp_list)
       return  temp_list
# 寻找分裂点
   def split_finding(self, Candidate_split, sort_array, Key_word):
       Grad = []
       laplace = []
       temp_G = 0
       temp_H = 0
       begin_point = 0
       gain = []
       for i in range(1, len(Candidate_split)):
           for j in range(begin_point, len(sort_array)):
               if sort_array[j, Key_word]<Candidate_split[i] and sort_array[j, Key_word]>=Candidate_split[i-1]:
                   temp_G += sort_array[j, -1]
                   temp_H += 2
               else:
                   begin_point = j
                   Grad.append(temp_G)
                   laplace.append(temp_H)
                   temp_G = 0
                   temp_H = 0
                   break
       for k in range(len(Grad)):
           temp1 = sum(Grad[:k+1])**2 / (sum(laplace[:k+1]) + self.Lambda)
           temp2= sum(Grad[1+k:])**2 / (sum(laplace[k+1:]) + self.Lambda)
           temp3=-1.0*(sum(Grad)**2) / (sum(laplace) + self.Lambda)
           gain.append(temp1 + temp2 + temp3)
       split=gain.index(max(gain)) + 1
       return split, max(gain)
# 获得子树集
   def son_set(self, Sort_array, Key_word, splitpoint):
       leftSon_tree = []
       rightSon_tree = []
       for i in range(0, len(Sort_array)):
           if Sort_array[i, Key_word] <= splitpoint:
               rightSon_tree.append(Sort_array[i, :])
           elif Sort_array[i, Key_word] > splitpoint:
               leftSon_tree.append(Sort_array[i, :])
       rightSon_tree = np.array(rightSon_tree)
       leftSon_tree = np.array(leftSon_tree)
       return  rightSon_tree, leftSon_tree
# 计算梯度生成训练下一棵树的数据集
   def grad_compute(self,son_array, Best_attri):
       weight = -1 *self.segma* 2.0*sum(son_array[:, -1]) / (2.0 * len(son_array) + self.Lambda)
       self.weight.append(weight)
       for i in range(0, len(son_array)):
           son_array[i, -1] = weight * son_array[i, Best_attri] + son_array[i, -1]
       if self.psy == None:
           self.psy = np.array(son_array)
       else:
           self.psy = np.concatenate((self.psy, son_array), axis=0)
       return weight

# 预测
   def xgboost_prediction(self,input_set):##input_set输入集
       predict_record = np.zeros(len(input_set))
       temp_data = 0
       for i in range(0,len(input_set)):
         for j in range(0,self.tree_num):
             temp_data = self.judge_func(input_set,i,j,0,temp_data)
             predict_record[i] = predict_record[i] + temp_data
       return predict_record

# 构造线程中的函数
def single_thread_xgboost(threadnum, Lambda, Set):
    myMSE = 1000000000.0
    tree_num = 50
    mse_list = list(np.zeros(threadnum))
    rand_index = np.random.choice(len(Set), 0.8*len(Set))
    Train_set = Set[rand_index]
    validation_set = Set[np.array(list(set(range(len(Set))) - set(rand_index)))]
    local_tset = Train_set
    local_vset = validation_set
    test_xgboost = Xgboost(Train_set, validation_set, Lambda)
    test_xgboost.ctftlist = {}
    for j in range(0,tree_num):
        para = list(np.zeros(threadnum))
        for i in range(0,threadnum):
            para[i] = Xgboost(local_tset,local_vset,Lambda=Lambda)
            mse_list[i] = para[i].MSE
        if myMSE >min(mse_list):
            myMSE = min(mse_list)
            local_tset = para[mse_list.index(myMSE)].psy
            local_vset = para[mse_list.index(myMSE)].validation_set
            for i in range(0, len(para[mse_list.index(myMSE)].psy)):
                local_tset[i, -1] = -1.0 * local_tset[i, -1]
            test_xgboost.ctftlist[str(j)] = para[mse_list.index(myMSE)].ctftlist['1']
            print(para[mse_list.index(myMSE)].weight)
        else:
            test_xgboost.tree_num = j
            break
    return test_xgboost

def multi_thread_xgboost(threadnum, Lambda, Set):  # lambad是一个超参数
    myMSE = 1000000000.0
    tree_num = 50
    mse_list = list(np.zeros(threadnum))
    Train_set = Set[0:int(len(Set) * 0.8), :]          #构造训练集
    validation_set = Set[int(len(Set) * 0.8):, :]      #构造验证集
    test_xgboost = Xgboost( Set[0:int(len(Set) * 0.8), :], Set[int(len(Set) * 0.8):, :], Lambda)
    temp_dic = {}
    for j in range(0, tree_num):
        print(j)
        temp_th = []
        para = list(np.zeros(threadnum))
        for i in range(0, threadnum):
            tem = MyThread(func = thread_fuc,args = (Train_set, validation_set, Lambda,i))
            temp_th.append(tem)
            tem.start()
        for t in range(0,threadnum):
             temp_th[t].join()
             para[t] = temp_th[t].get_result()
             mse_list[t] = para[t].MSE
        if myMSE >= min(mse_list):
            myMSE = min(mse_list)
            Train_set = para[mse_list.index(myMSE)].psy
            validation_set = para[mse_list.index(myMSE)].validation_set
            for i in range(0, len(para[mse_list.index(myMSE)].psy)):
                Train_set[i, -1] = -1.0 * Train_set[i, -1]
            temp_dic[str(j)] = para[mse_list.index(myMSE)].ctftlist['1']
        else:
            test_xgboost.tree_num = j
            test_xgboost.ctftlist = temp_dic
            break
    return test_xgboost







