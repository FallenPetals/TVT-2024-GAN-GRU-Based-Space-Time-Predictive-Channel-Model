import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import openpyxl
import tensorflow as tf
import keras.backend as K

# 定义 mae损失函数
def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))



# 定义 huber 损失
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    condition = K.abs(error) < delta
    squared_loss = 0.5 * K.square(error)
    linear_loss = delta * (K.abs(error) - 0.5 * delta)
    return K.mean(K.switch(condition, squared_loss, linear_loss))

# log cosh 损失
def logcosh(y_true, y_pred):
    log_loss = tf.math.log(tf.math.cosh(y_pred - y_true))
    return tf.reduce_mean(log_loss)

# 空时自相关函数损失
# 定义权重计算函数
def calculate_weights(rx_positions, look_back):
    weights = []
    for i in range(len(rx_positions) - look_back):
        x = sum(rx_positions[i:i+look_back] == rx_positions[i])
        weight = x / (2**0.5) + (look_back - x)
        weights.append(weight)
        print('weight',weight)
    return tf.convert_to_tensor(weights, dtype=tf.float32)

# 定义Wlog损失函数
def weighted_logcosh(y_true, y_pred, weights):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    log_loss = tf.math.log(tf.math.cosh(y_pred - y_true))

    # 调整 weights 的形状以匹配 log_loss 的形状
    weights = tf.expand_dims(weights, axis=-1)
    weights = tf.expand_dims(weights, axis=-1)
    weights = tf.tile(weights, [1, tf.shape(log_loss)[1], 1])
    # 调整 log_loss 的形状以匹配 weights 的形状
    log_loss = tf.expand_dims(log_loss, axis=1)
    log_loss = tf.tile(log_loss, [1, tf.shape(weights)[1], 1])

    weighted_loss = log_loss * weights
    return K.mean(weighted_loss)

def weighted_logcosh_wrapper(rx_positions, look_back):
    weights = calculate_weights(rx_positions, look_back)
    def weighted_logcosh_inner(y_true, y_pred):
        return weighted_logcosh(y_true, y_pred, weights)
    return weighted_logcosh_inner

# 定义 Weighted mse 损失
def weighted_mse(y_true, y_pred, weights):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mse_loss = tf.square(y_pred - y_true)

    # 调整 weights 的形状以匹配 mse_loss 的形状
    weights = tf.expand_dims(weights, axis=-1)
    weights = tf.expand_dims(weights, axis=-1)
    weights = tf.tile(weights, [1, tf.shape(mse_loss)[1], 1])
    # 调整 mse_loss 的形状以匹配 weights 的形状
    mse_loss = tf.expand_dims(mse_loss, axis=1)
    mse_loss = tf.tile(mse_loss, [1, tf.shape(weights)[1], 1])

    weighted_loss = mse_loss * weights
    return K.mean(weighted_loss)

def weighted_mse_wrapper(rx_positions, look_back):
    weights = calculate_weights(rx_positions, look_back)
    def weighted_mse_inner(y_true, y_pred):
        return weighted_mse(y_true, y_pred, weights)
    return weighted_mse_inner



# 有num_rx个Rx位置，每个Rx位置有20个样本
num_rx = 15
samples_per_rx = 20
# 生成 rx_positions 数组
rx_positions = np.repeat(np.arange(1, num_rx + 1), samples_per_rx)
print(rx_positions)  # 输出: [1, 1, 1, ..., 15, 15, 15] 总共300个元素
print(len(rx_positions))  # 输出: 300

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

# 定义数据集构造函数
def create_dataset(dataset, look_back=21):
	dataX, dataY = [], []
	for i in range(len(dataset)-2*look_back+2):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(dataset[i + 2*look_back-2, :])
	return numpy.array(dataX), numpy.array(dataY)

# 定义随机种子，以便重现结果
numpy.random.seed(7)

# 加载数据
dataframe = read_csv('generated_data.csv', usecols=np.arange(0,300), engine='python')
#dataframe = read_csv('NLOS_mea.csv', usecols=np.arange(0,300), engine='python')
#dataframe = read_csv('pdp_imag.csv', usecols=np.arange(0,300), engine='python')

#
# 加载真实测量数据
# real_data = read_csv('LOS_mea.csv', usecols=np.arange(0,300), engine='python')

# 打印数据集
input_len = 300
dataset = dataframe.values
dataset = dataset.astype('float32')
print('original_data',dataset)
print(dataset.shape)
# 缩放数据
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print('scaler',dataset)
print(dataset.shape)

# 划分训练集和测试集
train_size = 200# 训练集占总数据集的80%
test_size = 100# 测试集占总数据集的20%
train, test = dataset[0:train_size,:], dataset[len(dataset)-test_size:len(dataset),:]# 训练集和测试集
print('train',train)
print('train',train.shape)
print('test',test)
print('test',test.shape)
print('length',len(test))
#print(train[0:3,0])
# 预测数据步长为3,三个预测一个，3->1
look_back = 21
# 构造训练集和测试集数据
trainX, trainY = create_dataset(train, look_back)
print('trainX',trainX)
print('trainX',trainX.shape)
print('trainY',trainY)
print('trainY',trainY.shape)
testX, testY = create_dataset(test, look_back)
print('testX',testX)
print('testY',testY)
print('testY',testY.shape)
validX, validY = create_dataset(dataset, look_back)
# 重构输入数据格式 [samples, time steps, features] = [93,1,3]
trainX = numpy.reshape(trainX, (trainX.shape[0], input_len, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], input_len, testX.shape[1]))
validX = numpy.reshape(validX, (validX.shape[0], input_len, validX.shape[1]))
print('trainX',trainX.shape)
print('testX',testX.shape)

#print('test',testX)

# 构建 GRU 网络
model = Sequential()
model.add(GRU(4, input_shape=(input_len, look_back)))
model.add(Dense(input_len))

model.compile(loss='mean_squared_error', optimizer='adam') # 均方误差
# model.compile(loss=huber_loss, optimizer='adam') # huber_loss误差
# model.compile(loss=logcosh, optimizer='adam') # logcosh误差
# model.compile(loss=weighted_mse_wrapper(rx_positions, look_back), optimizer='adam')
# model.compile(loss=weighted_logcosh_wrapper(rx_positions, look_back), optimizer='adam') # 带权重的logcosh误差
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)



# 对训练数据的Y进行预测
trainPredict = model.predict(trainX)
print('predict',trainPredict.shape)
print('trainPredict',trainPredict)
# 对测试数据的Y进行预测
testPredict = model.predict(testX)
# 对所有数据的Y进行预测
validPredict = model.predict(validX)
# 对数据进行逆缩放
trainPredict = scaler.inverse_transform(trainPredict)
print('predict',trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)
print('test',testPredict.shape)
validPredict = scaler.inverse_transform(validPredict)
validY = scaler.inverse_transform(validY)
print('valid',validPredict)
print(validPredict.shape)

# 计算均方误差
testscoremse = mean_squared_error(testY, testPredict)
print('test Score: mse %.2f' % (testscoremse))

# 计算RMSE误差
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
validScore = math.sqrt(mean_squared_error(validY, validPredict))
print('Valid Score: %.2f RMSE' % (validScore))

# 训练集预测结果可视化
trainPredictPlot = numpy.empty_like(dataset)
print(trainPredictPlot.shape)
# 用nan填充数组
trainPredictPlot[:, :] = numpy.nan
# 将训练集预测的Y添加进数组
trainPredictPlot[2*look_back-1:len(trainPredict)+2*look_back-1, :] = trainPredict
print(trainPredictPlot.shape)

# 测试集预测结果可视化
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
# 将测试集预测的Y添加进数组，从第94+4位到最后，共44行
testPredictPlot[len(dataset)-len(testPredict):len(dataset), :] = testPredict
original_data = scaler.inverse_transform(dataset)

# # 验证集预测结果可视化
# validPredictPlot = numpy.empty_like(dataset)
# validPredictPlot[:, :] = numpy.nan
# # 将测试集预测的Y添加进数组，从第94+4位到最后，共44行
# validPredictPlot[(look_back):len(dataset), :] = validPredict


# # 保存原始数据
# original_data = np.array(original_data)
# data = pd.DataFrame(original_data)
# writer = pd.ExcelWriter('original_data.xlsx')		# 写入Excel文件
# data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# writer._save()
# writer.close()

#保存训练集预测数据
trainPredictPlot = np.array(trainPredictPlot)
data = pd.DataFrame(trainPredictPlot)
writer = pd.ExcelWriter('trainPredictPlot_LOS.xlsx')		# 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer._save()
writer.close()
#保存测试集预测数据
testPredictPlot = np.array(testPredictPlot)
data = pd.DataFrame(testPredictPlot)
writer = pd.ExcelWriter('testPredictPlot_LOS.xlsx')		# 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer._save()
writer.close()
# #保存验证集预测数据
# validPredictPlot = np.array(validPredictPlot)
# data = pd.DataFrame(validPredictPlot)
# writer = pd.ExcelWriter('LOS.xlsx')		# 写入Excel文件
# data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# writer._save()
# writer.close()

# # 保存训练集预测数据
# testPredict = np.array(testPredict)
# data = pd.DataFrame(testPredict)
# writer = pd.ExcelWriter('testPredict_imag.xlsx')		# 写入Excel文件
# data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# writer._save()
# writer.close()

# 画图
# plt.plot(original_data)
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
# plt.show()



