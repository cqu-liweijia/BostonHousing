#encoding:utf-8
from sklearn import datasets as ds
import numpy as np
from sklearn.linear_model import LinearRegression

if __name__=='__main__':
    print("---start---")
    boston = ds.load_boston() # �������ݼ�
    X = boston.data # �������������,506�����ݣ�13������
    Y = boston.target # �������label��506������ֵ
    
    #��������ģ��
    linreg = LinearRegression()
    #����ѵ��
    linreg.fit(X[0:450], Y[0:450])
    
    #��ȡ���Իع����tΪtheta t0Ϊƫ����
    t=linreg.coef_
    t=t.reshape(13,1)
    t0=linreg.intercept_
    
    #Ԥ�⼯
    PX=X[450:506]
    PY=Y[450:506]
    PY=PY.reshape(56,)
    
    #����Ԥ��
    pY=np.dot(PX,t)+t0
    pY=pY.reshape(56,)
    for i in range(0,np.size(PY)):
        print("Predict: "+str(pY[i])+" real: "+str(PY[i]))
    print("---end---")
