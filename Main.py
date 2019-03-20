#encoding:utf-8
from sklearn import datasets as ds
import numpy as np

def main():        
    print("-main-")
    
def loss(X,Y,theta):#计算X在h函数的结果y中的全部损失
    num=np.size(Y)
    h=np.dot(X,theta)
    loss=np.dot((h-Y).transpose(),h-Y)/(2*num)
    return loss[0][0]

def der(X,Y,theta):
    num=np.size(Y)
    temp=np.dot(X,theta)-Y
    return np.dot(X.transpose(),temp)/num
    

if __name__=='__main__':
    print("---start---")
    main()
    boston = ds.load_boston() # 导入数据集
    X = boston.data # 获得其特征向量,506个数据，13个属性
    Y = boston.target # 获得样本label，506个房价值
    Y=Y.reshape(506,1)#左列为1的增广，所以是450*14（第一列为1）
    xx=np.ones((506,1))
    X = np.c_[xx,X]
    
    #对数据进行特征缩放
    for i in range(1,14):
        maxn=X[:,i].max()
        minn=X[:,i].min()
        meann=X[:,i].mean()
        X[:,i]=(X[:,i]-meann)/(maxn-minn)  

    
    #设置初始参数
    theta=np.zeros((14,1))#定义14个参数
    learnRatio=0.001#定义学习率
    times=1700#定义循环次数
    
    for i in range(0,times):
        if i%10==0:
            print("After %d:\tloss in train is: %.7f\tloss in validation is %.7f"%(i,loss(X[0:400],Y[0:400],theta),loss(X[400:450],Y[400:450],theta)))
            #print("After "+str(i)+" loss in train is: %.2f"+"\t\tloss in Validation is: "+str(loss(X[400:450],Y[400:450],theta)),%loss(X[0:400],Y[0:400],theta))
        theta=theta-learnRatio*der(X[0:400],Y[0:400],theta)
    
    #对训练的结果进行预测
    PX=X[450:506]
    PY=Y[450:506]
    pY=np.dot(PX,theta)
    for i in range(0,PY.size):
        print("predict: %.1f\treal: %.1f"%(pY[i,0],PY[i,0]))
    print("loss in predict is:"+str(loss(X[450:506],Y[450:506],theta)))
    
    
    print("---end---")
