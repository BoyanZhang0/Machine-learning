import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
import matplotlib.pyplot as plt

df = pd.read_csv("song_data.csv").drop('song_name', axis=1)

# 离散属性
DiscreteValue = ['key', 'time_signature', 'audio_mode']

# 连续属性
ContinuousValue = []
for i in df.columns:
    if i not in DiscreteValue:
        ContinuousValue.append(i)
ContinuousValue.remove('song_popularity')

# 对离散变量进行独热编码
df = pd.get_dummies(df, columns=DiscreteValue[0:2])

# 对连续型变量进行归一化
df[ContinuousValue] = MinMaxScaler().fit_transform(df[ContinuousValue])

# 定义线性模型的自变量X和因变量Y
X = df.drop('song_popularity', axis=1)
Y = df['song_popularity']

# 进行多重共线性检测和处理
vif_info = pd.DataFrame()
vif_info['X'] = X.columns
vif_info['VIF'] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print(vif_info)

count = 0

for i in vif_info["VIF"]:
    if i > 5:
        count = count + 1

if count > 0:
    pca = PCA(n_components=0.95)
    X = pd.DataFrame(pca.fit_transform(X))
    print(pca.explained_variance_ratio_)

#进行模型构建
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20, shuffle=True)
Regressor = LinearRegression()
Regressor.fit(X_train, Y_train)

# 进行实验结果分析
PredictYOnTrain = Regressor.predict(X_train)
TrainError = mean_squared_error(Y_train, PredictYOnTrain)
print("训练误差为：", TrainError)
PredictYOnText = Regressor.predict(X_test)
TextError = mean_squared_error(Y_test, PredictYOnText)
print("测试误差为：", TextError)

# 数据可视化

#由于自变量维度过高故选取最重要的几个进行可视化
tempLabel = []
for i in df.columns:
    if i != 'song_popularity' and i != 'song_name':
        tempLabel.append(i)
print(tempLabel)

for i in tempLabel:
    data = DataFrame({i: df[i], 'y': Y})
    print(data.corr(), "\n")

#此处根据相关系数选取acousticness, danceability, instrumentalness, loudness
fig, ax = plt.subplots(2, 4)
# 训练集

# 标签acousticness
ax[0][0].set_xlabel('acousticness')
ax[0][0].set_ylabel('song_popularity')
ax[0][0].set_title('train')
ax[0][0].scatter(X_train[1], Y_train, label="realValue")
ax[0][0].scatter(X_train[1], PredictYOnTrain, label="predictValue")
ax[0][0].legend()
# 标签danceability
ax[0][1].set_xlabel('danceability')
ax[0][1].set_ylabel('song_popularity')
ax[0][1].set_title('train')
ax[0][1].scatter(X_train[2], Y_train, label="realValue")
ax[0][1].scatter(X_train[2], PredictYOnTrain, label="predictValue")
ax[0][1].legend()
# 标签instrumentalness
ax[0][2].set_xlabel('instrumentalness')
ax[0][2].set_ylabel('song_popularity')
ax[0][2].set_title('train')
ax[0][2].scatter(X_train[4], Y_train, label="realValue")
ax[0][2].scatter(X_train[4], PredictYOnTrain, label="predictValue")
ax[0][2].legend()
#标签loudness
ax[0][3].set_xlabel('loudness')
ax[0][3].set_ylabel('song_popularity')
ax[0][3].set_title('train')
ax[0][3].scatter(X_train[6], Y_train, label="realValue")
ax[0][3].scatter(X_train[6], PredictYOnTrain, label="predictValue")
ax[0][3].legend()

# 测试集

# 标签acousticness
ax[1][0].set_xlabel('acousticness')
ax[1][0].set_ylabel('song_popularity')
ax[1][0].set_title('test')
ax[1][0].scatter(X_test[1], Y_test, label="realValue")
ax[1][0].scatter(X_test[1], PredictYOnText, label="predictValue")
ax[1][0].legend()
# 标签danceability
ax[1][1].set_xlabel('danceability')
ax[1][1].set_ylabel('song_popularity')
ax[1][1].set_title('test')
ax[1][1].scatter(X_test[2], Y_test, label="realValue")
ax[1][1].scatter(X_test[2], PredictYOnText, label="predictValue")
ax[1][1].legend()
# 标签instrumentalness
ax[1][2].set_xlabel('instrumentalness')
ax[1][2].set_ylabel('song_popularity')
ax[1][2].set_title('test')
ax[1][2].scatter(X_test[4], Y_test, label="realValue")
ax[1][2].scatter(X_test[4], PredictYOnText, label="predictValue")
ax[1][2].legend()
#标签loudness
ax[1][3].set_xlabel('loudness')
ax[1][3].set_ylabel('song_popularity')
ax[1][3].set_title('test')
ax[1][3].scatter(X_test[6], Y_test, label="realValue")
ax[1][3].scatter(X_test[6], PredictYOnText, label="predictValue")
ax[1][3].legend()

plt.show()