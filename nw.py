# -*- coding: utf-8 -*- 
print("--------------------------start---------------------")

import tensorflow as tf
from sklearn import datasets as ds

#神经网络的配置参数
dimensionOfX=13
dimensionOfY=1
trainset_size=400
learning_rate=0.1
batch_size=7#定义窗口大小
STEPS=70000  
lamda=0.95

#随机生成训练集
boston = ds.load_boston() # 导入数据集
X = boston.data # 获得其特征向量,506个数据，13个属性
Y = boston.target # 获得样本label，506个房价值
Y=Y.reshape(506,1)
#对数据进行特征缩放
for i in range(0,13):
    maxn=X[:,i].max()
    minn=X[:,i].min()
    meann=X[:,i].mean()
    X[:,i]=(X[:,i]-meann)/(maxn-minn) 

#训练集的输入格式
x=tf.placeholder(tf.float32,shape=(None,dimensionOfX),name='x-input')
y_=tf.placeholder(tf.float32,shape=(None,dimensionOfY),name='y-input')

#定义并初始化神经网络的参数值
hidden_num_1=7
hidden_num_2=9

w1=tf.get_variable("w1",[13,hidden_num_1], initializer=tf.truncated_normal_initializer(stddev=0.1))
b1=tf.get_variable("b1", [1,hidden_num_1], initializer=tf.constant_initializer(0.0))

w2=tf.get_variable("w2",[hidden_num_1,hidden_num_2], initializer=tf.truncated_normal_initializer(stddev=0.1))
b2=tf.get_variable("b2", [1,hidden_num_2], initializer=tf.constant_initializer(0.0))

w3=tf.get_variable("w3",[hidden_num_2,1], initializer=tf.truncated_normal_initializer(stddev=0.1))
b3=tf.get_variable("b3", [1,1], initializer=tf.constant_initializer(0.0))


#模拟神经网络的传输
a1=tf.nn.relu(tf.matmul(x,w1)+b1)
a2=tf.nn.relu(tf.matmul(a1,w2)+b2)
y=tf.nn.relu(tf.matmul(a2,w3)+b3)

#定义输出损失函数和反向传播参数,加入正则化
data_loss=tf.reduce_mean(tf.multiply(y-y_,y-y_))
tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamda)(w1))
tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamda)(w2))
tf.add_to_collection('losses',data_loss)
loss=tf.add_n(tf.get_collection('losses'))

#定义反向传播参数
train_step=tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    #初始化参数
    init_op=tf.global_variables_initializer()
    sess.run(init_op)

    #开始训练
    for i in range(STEPS):
        start=(i*batch_size)%trainset_size
        end=min(start+batch_size,trainset_size)
        
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%2000==0:
            total_loss_train=sess.run(loss,feed_dict={x:X[0:trainset_size],y_:Y[0:trainset_size]})
            total_loss_Validation=sess.run(data_loss,feed_dict={x:X[400:450],y_:Y[400:450]})
            print("After %d training steps,loss_train %g ,\tloss_Validation %g"%(i,total_loss_train,total_loss_Validation))

    pre=sess.run(y,feed_dict={x:X[450:506],y_:Y[450:506]})   
    for i in range(450,506):
        print("myPredict:"+str(pre[i-450][0])+"   real:"+str(Y[i][0]))  
    print("loss in predict is "+str(sess.run(data_loss,feed_dict={x:X[450:506],y_:Y[450:506]})))
    sess.close()

print("--------------------------end-----------------------")
