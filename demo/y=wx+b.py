import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

#设置生成的图像尺寸和去除警告
os.environ['TF_CPU_MIN_LOG_LEVEL'] = '2'
plt.rcParams["figure.figsize"] = (14,8)

#随机生成一个线性的数据，当然你可以换成读取对应的数据集
n_observations = 100
xs = np.linspace(-3,3,n_observations)
ys = 0.8 * xs + 0.1 + np.random.uniform(-0.5,0.5,n_observations)
plt.scatter(xs,ys)
plt.show()

#准备好占位符placeholder
X = tf.placeholder(tf.float32,name='X')
Y = tf.placeholder(tf.float32,name='Y')

#初始化参数/权重
W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name='bias')

#计算预测结果
Y_pred = tf.add(tf.multiply(X,W),b)

#前向传播：计算损失值函数
loss = tf.square(Y-Y_pred,name='loss')

#初始化优化器optimizer
learning_rate = 0.01 #学习率
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#指定迭代次数，并在session里执行graph
n_samples = xs.shape[0]
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)#初始化参数
    wirter = tf.summary.FileWriter('./graphs/linear_regression',sess.graph)#写入日志
    for i in range(1000):
        total_loss = 0
        for x,y in zip(xs,ys):
            # 通过feed_dic把数据灌进去
            _,loss_value = sess.run([optimizer,loss],feed_dict={X:x,Y:y})#反向传播：更新权重和参数
            total_loss += loss_value
        if i % 5 == 0:
            print('Epoch {0}: {1}'.format(i,total_loss/n_samples))

    # 关闭writer
    wirter.close()

    # 取出w和b的值
    W,b =sess.run([W,b])

#打印最后更新的w、b的值
print(W,b)
print('W:'+str(W[0]))
print('b:'+str(b[0]))

#画出线性回归线
plt.plot(xs, ys, 'bo', label='Real data')
plt.plot(xs, xs * W + b, 'r', label='Predicted data')
plt.legend()
plt.show()