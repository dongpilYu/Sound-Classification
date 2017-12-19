import numpy as np
from hmmlearn import hmm
import glob
import os
import librosa

def extract_mfcc(parent_dir, sub_dirs, file_ext="*.wav",output=""):

    labels = []
    features = []
    for l, sub_dir in enumerate(sub_dirs):

        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):

            sound_clip, s = librosa.load(fn)
            label = fn.split('\\')[3].split('-')[1]
            # UrbanSound8K/audio/fold1/7061-6-0-0.wav
            y_harmonic, y_percussive = librosa.effects.hpss(sound_clip)
            tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=s)
            mfcc = librosa.feature.mfcc(y=sound_clip,sr=s,hop_length=512,n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc)
            beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),beat_frames)

            labels.append(label)
            features.append(beat_mfcc_delta)


    np.savez("Extraction/audio" + output, features=np.ndarray(features), labels=np.ndarray(labels))
    return np.array(features), np.array(labels)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    # (720,2)
    one_hot_encode[np.arange(n_labels), labels.astype('int64')] = 1
    return one_hot_encode

parent_dir = "UrbanSound8K\\audio"

for i in range(11):
    sub_dirs= ['fold'+str(i)]
    features,labels = extract_mfcc(parent_dir,sub_dirs,output="/mfcc_50_"+str(i))
    labels = one_hot_encode(labels)

