
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv1D, MaxPooling1D, Dropout, Activation, Input, ReLU, LSTM
import numpy as np
from keras.models import load_model
from keras.losses import mean_squared_error
import os
import cv2
import mp3
from keras import metrics
from tensorflow import concat
from preprocessing import *
import threading
import time
import math

def non_zero_accuracy(y_true, y_pred):
    y_true = y_true * (y_true != 0) 
    y_pred = y_pred * (y_true != 0)

    error = metrics.mean_squared_error(y_true, y_pred)
    return error

def LossFunction(t, p):
    return mean_squared_error(t, p)

def InitializeModel():
  model = Sequential()
  model.add(Conv1D(32, 3, input_shape=(1, 900)))
  model.add(LSTM(512, dropout = 0.3, recurrent_dropout = 0.3, activation="relu"))
  model.add(Dense(88, activation="tanh"))

  model.compile(
    optimizer='adam',
    loss=LossFunction,
    metrics=["accuracy"],
  )
  
  return model

def FitSave(model, i, o):
  history = model.fit(
    i,
    o,
    epochs=1,
    batch_size=64,
    verbose=1)
  model.save('model.h5', True)
  return history.history["accuracy"][-1]

def TestModel(modelFile, i):
  model = load_model("model.h5", custom_objects={"LossFunction":LossFunction})
  return model.predict(i)


data = []
def GenerateData(mp3Dir, midDir, appendConverted=True):
  for f in os.listdir(midDir):
    while len(data) > 5:
      time.sleep(1)

    if ".mid" in f:
      try:
        if appendConverted:
          i, o = GetDataSignal(midDir+"/"+f, mp3Dir+"/"+f.replace(".mid", "-converted.mp3"))
        else:
          i, o = GetDataSignal(midDir+"/"+f, mp3Dir+"/"+f.replace(".mid", ".mp3"))
        #i = np.reshape(i, (1, i.shape[0], i.shape[1]))
        data.append((i, o, f))
      except Exception as e:
        print(e)
        raise e
        continue

def Train(model):
  while True:
    if data != []:
      try:
        i, o = data.pop(0)
        FitSave(model, i, o)
      except Exception as e:
        print(e)
        continue

def FitData(model, mp3Dir, midDir, appendConverted=True):
  dataThread = threading.Thread(target=GenerateData, args=(mp3Dir, midDir))
  dataThread.start()
  t = time.time()
  accuracies = []
  while True:
    if data != []:
      try:
        i, o, f = data.pop(0)
        acc = str(round(FitSave(model, i, o)*100, 2))
        print(acc+"% - "+str(round(time.time()-t, 1))+"s"+" - "+f)
        open("accuracy.json", "a").write('["'+f+'", '+acc+'],\n')
        accuracies.append(float(acc))
        t = time.time()
      except Exception as e:
        print(e)
        continue
    elif not dataThread.isAlive():
      return sum(accuracies)
    time.sleep(1)

def OptimizeModel(soundDir, midiDir):
  layerSizes = [4096, 2048, 1024, 512, 256]
  
  for x in reversed(layerSizes):
    for y in reversed(layerSizes):
      model = Sequential()
      model.add(Dense(x, activation="tanh", input_shape=(88,)))
      model.add(Dense(y, activation="tanh"))
      model.add(Dropout(.1))
      model.add(Dense(88, activation='tanh'))
      model.compile(optimizer='adam', loss=LossFunction, metrics=["accuracy"])
      open("accuracy.json", "a").write('\n--------START MODEL L1={} L2={}----------\n'.format(x, y))
      performance = FitData(model, soundDir, midiDir)
      open("accuracy.json", "a").write('\n--------END MODEL L1={} L2={}----------\n--------RES={}--------'.format(x, y, performance))
      open("modelPerformance.json", "a").write('MODEL L1={} L2={} RES={}\n'.format(x, y, performance))

#OptimizeModel(r"data\sound_data", r"data\midi_data")
model = InitializeModel()
for x in range(0, 10):
  FitData(model, r"data\sound_data", r"data\midi_data")
#i, o = main.GetData("data/invent1.mid", "data/invent1.mp3")
# i = main.GetSound("data/invent1.mp3")
# cv2.imwrite("predict_data.bmp", TestModel("model.h5", i)*255)
