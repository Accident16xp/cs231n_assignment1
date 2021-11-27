from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    结构化的SVM损失函数，利用循环来实现
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    输入有维度D，有C个分类，然后在N个例子的小数据集上进行操作

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    W：一个(D,C）包含权重的数组
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    X：一个（N，D)包含小数据集的数组
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    y即标签集
    - reg: (float) regularization strength
    reg为正则项

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero
    #梯度初始为0
    # compute the loss and the gradient
    num_classes = W.shape[1]
    #类别数，就是W的第二个维度，也即C
    num_train = X.shape[0]
    #数据数量为X的第一个维度，也即N
    loss = 0.0
    #初始化损失为0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        #得分为X的第i个行向量点乘W即为 （1，D）*（D,C)
        #这里的.dot函数即为乘积的意思
        #socres即一个图片在所有类上的得分向量
        correct_class_score = scores[y[i]]
        #正确类处的得分如上
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1，1的作用更像一个阈值
            #差距为分数-正确类的评分，再加上一项1
            if margin > 0:
                #如果margin>0,则视为误差，求和
                loss += margin
                dW[:,y[i]]+=-X[i,:]
                #对y[i]的每一项处的倒数均为-X[i,]
                dW[:,j]+=X[i]
                

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW/=num_train
    #总体的误差，为求平均

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW+=reg*W
    #np.sum（W*W)，意为对W的整体的每一个位置的平方数求和，或者说，这是一个L2范数的正则项
    #+reg*||W||^2

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the  #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.              
    #好的建议是一边计算顺势，一边计算导数，即修改上面的代码
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


#下面是向量化的实现
def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) 
    # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #实现一个向量化的表示损失函数的方法
    num_classes=W.shape[1]
    num_train=X.shape[0]
    scores=X.dot(W)
    
    scores_correct=scores[np.arange(num_train),y]
    scores_correct=np.reshape(scores_correct,(num_train,-1))
    #转置一下
    margins=scores-scores_correct+1
    margins=np.maximum(0,margins)
    margins[np.arange(num_train),y]=0
    loss=np.sum(margins)/num_train+0.5*reg*np.sum(W*W)

    
    
                              
                              
                              
                      

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the 
    #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #梯度计算
    coeff_mat = np.zeros((num_train,num_classes))
    coeff_mat[margins>0]=1
    coeff_mat[range(num_train),list(y)]=0
    coeff_mat[range(num_train),list(y)]=-np.sum(coeff_mat,axis=1)
     
    dW=(X.T).dot(coeff_mat)
    dW=dW/num_train+reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
