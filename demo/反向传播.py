import numpy as np

def sigmod(x):
    return 1/(1+np.exp(-x))

def sigmodDerivative(x):
    #求sigmodd 偏导
    return np.multiply(x,np.subtract(1,x))

def forward(weightsA,weightsB,bias):
    #前向传播    隐层
    neth1 = inputX[0] * weightsA[0][0] + inputX[1] * weightsA[0][1] + bias[0]
    outh1 = sigmod(neth1)
    print('隐层第一个神经元:neth:{},outh{}'.format(neth1,outh1))

    neth2 = inputX[0] * weightsA[1][0] + inputX[1] * weightsA[1][1] + bias[1]
    outh2 = sigmod(neth2)
    print('隐层第二个神经元:neth:{},outh{}'.format(neth2,outh2))
    #输出层
    neto1 = outh1 * weightsB[0][0] + outh2 * weightsB[0][1] + bias[2]
    outo1 = sigmod(neto1)
    print('输出层第一个神经元:neto:{},outo{}'.format(neto1,outo1))

    neto2 = outh1 * weightsB[1][0] + outh2 * weightsB[1][1] + bias[3]
    outo2 = sigmod(neto2)
    print('输出层第二个神经元:neto:{},outo{}'.format(neto2, outo2))

    #向量化
    outA = np.array([outh1,outh2]) #隐层
    outB = np.array([outo1,outo2])  #输出层

    Etotal = 1 / 2 * np.subtract(y, outB) ** 2
    print('误差值:',Etotal)

    return outA,outB

def backpagration(outA,outB):
    #反向传播
    deltaB = np.multiply(np.subtract(outB,y),sigmodDerivative(outB))   #δ(输出层)
    print('deltab:',deltaB)

    deltaA = np.multiply(np.matmul(np.transpose(weightsB),deltaB),sigmodDerivative(outA))   #δ(隐层)
    print('deltaA',deltaA)

    deltaWB = np.matmul(deltaB.reshape(2,1),outA.reshape(1,2))    #∂Etotal/∂w(输出层)
    print('deltaWB',deltaWB)

    deltaWA = np.matmul(deltaA.reshape(2, 1), inputX.reshape(1, 2))    #∂Etotal/∂w(隐层)
    print('deltaWA', deltaWA)

    # 权重参数更新
    weightsB_new = np.subtract(weightsB, deltaWB)
    print('weightsB_new',weightsB_new)

    bias[3] = np.subtract(bias[3],deltaB[1])
    print('biasB',bias[3])
    bias[2] = np.subtract(bias[2],deltaB[0])
    print('biasB',bias[2])

    weightsA_new = np.subtract(weightsA,deltaWA)
    print('weightsA_new',weightsA_new)

    bias[1] = np.subtract(bias[1], deltaA[1])
    print("biasA", bias[1])
    bias[0] = np.subtract(bias[0], deltaA[0])
    print("biasA", bias[0])
    print("all bias", bias)

    return weightsA_new, weightsB_new, bias


if __name__ == "__main__":
    # 初始化数据
    # 权重参数
    bias = np.array([0.5, 0.5, 1.0, 1.0])
    weightsA = np.array([[0.1, 0.3], [0.2, 0.4]])
    weightsB = np.array([[0.6, 0.8], [0.7, 0.9]])
    # 期望值
    y = np.array([1, 0])
    # 输入层
    inputX = np.array([0.5, 1.0])

    print("第1次前向传播")
    outA, outB = forward(weightsA, weightsB, bias)
    print("反向传播-参数更新")
    weightsA_new, weightsB_new, bias = backpagration(outA, outB)
    # 更新完毕
    # 验证权重参数--第n次前向传播
    for i in range(2,1001):
        print("第{}次前向传播".format(i))
        outA, outB = forward(weightsA_new, weightsB_new, bias)

        print("反向传播-参数更新")
        weightsA_new, weightsB_new, bias = backpagration(outA, outB)

