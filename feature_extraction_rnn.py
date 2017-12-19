import numpy as np
import librosa
import glob
import os

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)

def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 20, frames = 41, output=""):
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = []

    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip,s = librosa.load(fn)
            label = fn.split('\\')[3].split('-')[1]
            # UrbanSound8K/audio/fold1/7061-6-0-0.wav

            for (start,end) in windows(sound_clip,window_size):
                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                    mfccs.append(mfcc)
                    labels.append(label)
    features = np.asarray(mfccs).reshape(len(mfccs),frames,bands)

    print(features.shape)

    np.savez("Extraction/audio/" + output ,features=features,labels=labels)
    return np.array(features), np.array(labels, dtype=np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1

    return one_hot_encode

parent_dir = "UrbanSound8K\\audio"

for i in range(11):
    sub_dirs= ['fold'+str(i)]
    features,labels = extract_features(parent_dir,sub_dirs,output="mfcc_50_"+str(i))
    labels = one_hot_encode(labels)