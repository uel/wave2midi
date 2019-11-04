import mp3
import numpy as np
import midi
import math
import os
import cv2
import pylab

#sound accuracy data/s
sAcc = 512
#midi accuracy 
mAcc = 16

limits = []
noteFrequencies = []
for x in range(1, 90):
    x = x-0.5
    limits.append(math.pow(2, (x-49)/12)*440)
    noteFrequencies.append(math.pow(2, (x+0.5-49)/12)*440)

xAxis = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]*8
xAxis = xAxis[:-8]

def FrequencyDistribution(signal, rate=44100):
    fft = np.fft.fft(signal, norm="ortho")
    freq = np.fft.fftfreq(len(fft), 1/rate)

    notes = []
    for x in range(0, 88):
        r = np.where((freq>limits[x])*(freq<=limits[x+1]))
        r = abs(fft[r])
        val = max(r, default=0)/rate
        notes.append(val)

    # pylab.plot(range(1, 89), notes)
    # pylab.show()

    return np.array([notes])

def GetSound(soundFile):
    rate, data = mp3.read(soundFile)
    data = data.sum(axis=1) / 2

    freqDist = np.array([[0]*88])

    for x in range(0, math.floor((len(data)/rate))*mAcc):
        dist = FrequencyDistribution(data[int((rate/mAcc)*x):int((rate/mAcc)*(x+1))])
        freqDist = np.concatenate((freqDist, dist), axis=0)
    
    return freqDist[1:]

def GetSoundSignal(soundFile):
    rate, data = mp3.read(soundFile)
    data = data.sum(axis=1) / 2
    data+=abs(min(data))
    data/=max(data)
    # freqDist = np.array([[0]*88])

    # for x in range(0, math.floor((len(data)/rate))*mAcc):
    #     dist = data[int((rate/mAcc)*x):int((rate/mAcc)*(x+1))]
    #     freqDist = np.concatenate((freqDist, dist), axis=0)
    
    return data

def GetData(midiFile, soundFile):
    freqDist = GetSound(soundFile)
    midiArray = midi.Midi2Array(midiFile, mAcc)

    m = midiArray*2
    r = np.concatenate((freqDist[:m.shape[0]]*256, m[:freqDist.shape[0]]), axis=1)
    cv2.imwrite("current_data.bmp", r)

    return freqDist[:midiArray.shape[0]], midiArray[:freqDist.shape[0]]/128

def f(l, d):
    while l>0:
        if l % d == 0:
            return l
        else:
            l-=1
    return 0
def GetDataSignal(midiFile, soundFile):
    freqDist = GetSoundSignal(soundFile)
    midiArray = midi.Midi2Array(midiFile, 44100/900)

    # m = midiArray*2
    # r = np.concatenate((freqDist[:m.shape[0]]*256, m[:freqDist.shape[0]]))
    # cv2.imwrite("current_data.bmp", r)

    d = f(len(freqDist), 900)
    freqDist = np.array_split(freqDist[:d], int(d/900))
    #freqDist = np.stack(freqDist)
    return freqDist[:midiArray.shape[0]], midiArray[:len(freqDist)]/128