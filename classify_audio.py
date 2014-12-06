#!/usr/bin/env python2

from __future__ import division

from subprocess import check_output
from contextlib import contextmanager
from os import path
import tempfile
import shutil

from funcy import pluck, partial, compose
from numpy import hstack, ones
from scipy.io.wavfile import read as wav_read
from scipy.io import savemat
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from features import mfcc


# Because this isn't python3 :(
@contextmanager
def TemporaryDirectory():
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        shutil.rmtree(name)


def dense_data_and_labels(data_and_labels):
    X = hstack(pluck(0, data_and_labels)).T
    dense_labels = hstack([label*ones(data.shape[1]) for data, label in
                           data_and_labels])
    return X, dense_labels


def train(data_and_labels, cls):
    X, dense_labels = dense_data_and_labels(data_and_labels)
    classifier = cls()
    return classifier.fit(X, dense_labels)


def experiment(training_wavs, test_wavs, load_bucket, is_voice_cls, which_cls):
    get_data = partial(map, load_bucket)
    training_data = get_data(training_wavs)
    test_data = get_data(test_wavs)
    is_voice = train(zip(training_data, [0, 1, 1]), is_voice_cls)
    which_speaker = train(zip(training_data[1:], [1, 2]), which_cls)

    activity_data, activity_labels = dense_data_and_labels(zip(test_data,
                                                               [0, 1, 1]))
    score1 = is_voice.score(activity_data, activity_labels)

    whois_data, whois_labels = dense_data_and_labels(zip(test_data[1:],
                                                         [1, 2]))
    score2 = which_speaker.score(whois_data, whois_labels)
    return score1, score2


def train_classifiers(training_wavs, test_wavs, load_bucket, is_voice_cls, which_cls):
    get_data = partial(map, load_bucket)
    training_data = get_data(training_wavs)
    test_data = get_data(test_wavs)
    is_voice = train(zip(training_data, [0, 1, 1]), is_voice_cls)
    which_speaker = train(zip(training_data[1:], [1, 2]), which_cls)

    def classify(data):
        return is_voice.predict(data)*which_speaker.predict(data)
    
    return classify


def to_wav(mp4_path):
    with TemporaryDirectory() as d:
        wav_path = path.join(d, "out.wav")
        check_output(["ffmpeg", "-i", mp4_path, wav_path])
        return wav_read(wav_path)


FPS = 30    
WINLEN = 1/FPS
WINSTEP = WINLEN
    
def features(rate_sig):
    rate, sig = rate_sig
    return mfcc(sig, rate, winlen=WINLEN, winstep=WINSTEP).T


wav_to_features = compose(features, wav_read)
mp4_to_features = compose(features, to_wav)


def batch(arr, fps, winlen=WINLEN):
    return array(map(median, array_split(arr, len(arr)//n)))


# TODO: take mp4 as argument
# TODO: return mat file
def main():
    print(experiment(["silence1.wav", "marcell1.wav", "faraz1.wav"],
                     ["silence2.wav", "marcell2.wav", "faraz2.wav"],
                     wav_to_features, LinearSVC, GaussianNB))
    
    classify = train_classifiers(["silence1.wav", "marcell1.wav", "faraz1.wav"],
                                 ["silence2.wav", "marcell2.wav", "faraz2.wav"],
                                 wav_to_features, LinearSVC, GaussianNB)
    
    savemat("out.mat", {"x": classify(mp4_to_features("videos/1_stationary_single.mp4").T)})

if __name__ == '__main__':
    main()
