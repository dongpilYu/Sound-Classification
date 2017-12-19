import numpy as np
import librosa
import glob
import os

def windows(data, window_size):
    start = 0
    while(start < len(data)):
        yield int(start), int(start + window_size)
        start += (window_size / 10)

def extract_features(parent_dir, sub_dirs, file_ext="*.wav",bands=60, frames=101, output=""):
    window_size = 512 * (frames-1)
    log_specgrams = []
    labels = []

    # 90%
    """
    (0, 60, 101, 2)
    (0, 60, 101, 2)
    (0, 60, 101, 2)
    (0, 60, 101, 2)
    (0, 60, 101, 2)
    (0, 60, 101, 2)
    (0, 60, 101, 2)
    (0, 60, 101, 2)
    (0, 60, 101, 2)
    (0, 60, 101, 2)
    (0, 60, 101, 2)
    """

    # 50%
    """
    
    (0, 60, 41, 2)
    (13173, 60, 41, 2)
    (13021, 60, 41, 2)
    (14168, 60, 41, 2)
    (14606, 60, 41, 2)
    (13727, 60, 41, 2)
    (12279, 60, 41, 2)
    (12769, 60, 41, 2)
    (11955, 60, 41, 2)
    (12371, 60, 41, 2)
    (12610, 60, 41, 2)
    """

    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip, s = librosa.load(fn)
            label = fn.split('\\')[3].split('-')[1]
            # UrbanSound8K/audio/fold1/7061-6-0-0.wav

            for (start, end) in windows(sound_clip, window_size):
                if (len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                    logspec = librosa.logamplitude(melspec)
                    logspec = logspec.T.flatten()[:, np.newaxis].T

                    # 같은 배열에 대해 차원만 증가시키는 경우 [:, np.newaxis]를 사용한다.
                    # logspec = (60,41)
                    # logspec.T.flatten() = (41,60) -> (2460,) -> (2460,1) -> (1, 2460)

                    log_specgrams.append(logspec)
                    labels.append(label)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)

    # features
    # (5446,60,41,2)

    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    print(features.shape)

    np.savez("Extraction/audio" + output ,features=features,labels=labels)
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
    features,labels = extract_features(parent_dir,sub_dirs,output="_extraction_90_"+str(i))
    labels = one_hot_encode(labels)
