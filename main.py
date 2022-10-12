import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import tensorflow as tf
import keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Flatten
from sklearn.metrics import mean_squared_error
import statsmodels as stm
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('BBK01.WX0082.csv', delimiter = ';')

df.rename(columns= {'Unnamed: 0':'Date', 'BBK01.WX0082':'Value', 'BBK01.WX0082_FLAGS': 'bin'}, inplace = True)

del df['bin']

df = df.drop(labels = range(0,6), axis =0)

df['Value'] = [v.replace(',','.') for v in df['Value']]
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = [d.year for d in df['Date']]
#df['Date'] = [d.strftime('%m') for d in df.index]

df = df.astype({'Value':'float'})
df.set_index('Date', inplace = True)

df.head()

#print(df.dtypes)


plt.plot(df['Value'])
plt.show()

plt.figure(figsize=(13,6))
bp = sns.boxplot(x = 'Year', y= 'Value', data = df)


decomposed = seasonal_decompose(df['Value'], model = 'additive')

trend = decomposed.trend
seasonal = decomposed.seasonal
residual = decomposed.resid

fig = decomposed.plot()
fig.set_size_inches((13, 13))
plt.show()

dataset = df.values
dataset = dataset.astype('float32')

seq_size = 5

train_size = int(len(dataset)*0.5)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size-seq_size:len(dataset),:]

def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        window = dataset[i:(i+seq_size),0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
    return np.array(x), np.array(y)


trainX, trainY = to_sequences(train, seq_size)
testX, testY = to_sequences(test, seq_size)

print("Shape of training set: {}".format(trainX.shape))
print("Shape of test set: {}".format(testX.shape))

trainX = np.reshape(trainX,(trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX,(testX.shape[0], 1, testX.shape[1]))


print("Shape of training set: {}".format(trainX.shape))
print("Shape of test set: {}".format(testX.shape))

print("Single LSTM with hidden Dense...")
model = Sequential()
model.add(LSTM(64, input_shape=(None, seq_size)))
model.add(Dense(32))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

from keras.callbacks import EarlyStopping
monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = 20, verbose = 1, mode = 'auto', restore_best_weights = True)
model.summary()

print('Train...')

model.fit(trainX,trainY, validation_data = (testX, testY), verbose = 2, epochs = 100)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))


testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(seq_size*2)+1-seq_size:len(dataset)-1, :] = testPredict

x = df.index
trainPredictY = [t[0] for t in trainPredictPlot]
testPredictY = [t[0] for t in testPredictPlot]

x = df.index
plt.figure(figsize=(13,6))
actual, = plt.plot(x,df['Value'])
actual.set_label('Actual')
trainPrediction, = plt.plot(x,trainPredictY)
trainPrediction.set_label('TrainPrediction')
testPrediction, = plt.plot(x,testPredictY)
testPrediction.set_label('TestPrediction')
#plt.vlines(x[train_size], np.min(trainPrediction,testPrediction), np.max(trainPrediction,testPrediction))
plt.legend()
plt.show()

