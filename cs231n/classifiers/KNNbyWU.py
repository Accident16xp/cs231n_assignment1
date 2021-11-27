from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class Knearest(object):
    

    def __init__(self):
        pass

    #训练部分，KNN的训练较为简单，直接记住数据即可
    def train(self, X, y):
        
        self.X_train = X
        self.y_train = y
        
    #预测函数
    def predict(self, X, k=1, num_loops=0):
      
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        

        return self.predict_labels(dists, k=k)



    
    #使用双循环计算两个向量之间的距离
    def compute_distances_two_loops(self, X):
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j]=np.sqrt(np.sum(np.square(X[i]-self.X_train[j])))

        return dists

    
#使用单循环计算两个向量之间的距离
    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i,:]=np.sqrt((np.sum(np.square(X[i,:]-self.X_train),axis=1)))
        return dists

#不适用循环计算两个向量之间的距离，直接采用向量的方式计算
    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
       
        dists = np.sqrt(-2*np.dot(X, self.X_train.T) + np.sum(np.square(self.X_train), axis = 1) + np.transpose([np.sum(np.square(X), axis = 1)]))

        return dists

    
#预测标签
    def predict_labels(self, dists, k=1):
        
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pred
