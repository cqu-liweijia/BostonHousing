#encoding:utf-8
from sklearn import datasets as ds
import numpy as np
from sklearn.linear_model import LinearRegression

if __name__=='__main__':
    print("---start---")
    boston = ds.load_boston() # 导入数据集
    X = boston.data # 获得其特征向量,506个数据，13个属性
    Y = boston.target # 获得样本label，506个房价值
    
    #定义线性模型
    linreg = LinearRegression()
    #进行训练
    linreg.fit(X[0:450], Y[0:450])
    
    #获取线性回归参数t为theta t0为偏置项
    t=linreg.coef_
    t=t.reshape(13,1)
    t0=linreg.intercept_
    
    #预测集
    PX=X[450:506]
    PY=Y[450:506]
    PY=PY.reshape(56,)
    
    #进行预测
    pY=np.dot(PX,t)+t0
    pY=pY.reshape(56,)
    for i in range(0,np.size(PY)):
        print("Predict: "+str(pY[i])+" real: "+str(PY[i]))
    print("---end---")
