import pandas as pd
import numpy as np
import math
# 决策函数
def decision(beta, X_val, value): # X_val为自变量集, value为阈值
    f, Y_pred = [], []
    z = X_val @ np.array(beta)
    for i in z:
        f.append(1 / (1 + math.exp(-i)))

    for i in f:
        if i > value:
            Y_pred.append(1)
        else:
            Y_pred.append(0)
    return Y_pred
# 将预测结果写入csv文件的函数
def writer(Y_pred, value): # value取0时为使用闭式解得到的结果，取1时为使用牛顿法得到的结果
    dataframe = pd.DataFrame({"class" : Y_pred})
    if value == 0:
        dataframe.to_csv("58121124_0.csv",index=False,sep=',')
    else:
        dataframe.to_csv("58121124_1.csv", index=False, sep=',')
# 用于计算准确率，查准率，查全率的函数
def discriminate(Y_pred, Y_real):
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(160):
        if Y_pred[i] == 1 and Y_real[i] == 1:
            TP = TP + 1
        elif Y_pred[i] == 1 and Y_real[i] == 0:
            FP = FP + 1
        elif Y_pred[i] == 0 and Y_real[i] == 1:
            FN = FN + 1
        else:
            TN = TN + 1
    accuracy = (TP + TN) / 160 # 准确率
    P = TP / (TP + FP) # 查准率
    R = TP / (TP + FN) # 查全率
    return accuracy, P, R

# 数据处理
X_train = pd.read_csv("train_features.csv")
Y_train = pd.read_csv("train_target.csv")
X_val = pd.read_csv("val_features.csv")
Y_val = pd.read_csv("val_target.csv")
X_test = pd.read_csv("test_feature.csv")

X_train["add_column"] = 1
X_val["add_column"] = 1
X_test["add_column"] = 1

X_train = np.array(X_train)
Y_train =np.array(Y_train)
X_val = np.array(X_val)
Y_val = np.array(Y_val)
X_test = np.array(X_test)

# 求解闭式解
beta1 = np.linalg.pinv(np.transpose(X_train) @ X_train) @ np.transpose(X_train) @ Y_train
print(" 闭式解求得的β值为:\n%s" % beta1)

# 进行决策
ThresholdValue = 0.5 #阈值

Y_pred1 = decision(beta1, X_val, ThresholdValue)
accuracy1, P1, R1 = discriminate(Y_pred1, Y_val)
print("闭式解方法且阈值为0.5时，准确率为：%s， 查准率为：%s， 查全率为：%s" % (accuracy1, P1, R1))

ThresholdValue = 0.52 # 将阈值设置为0.52,此时在验证集上准确率，查准率，查全率都达到1
Y_pred1 = decision(beta1, X_test, ThresholdValue)
writer(Y_pred1, 0)