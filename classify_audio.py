#!/usr/bin/env python2

from __future__ import division

from subprocess import check_call
from contextlib import contextmanager
from os import path
import tempfile
import shutil

import click
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


def train_classifiers(training_wavs, load_bucket, is_voice_cls, which_cls):
    get_data = partial(map, load_bucket)
    training_data = get_data(training_wavs)
    is_voice = train(zip(training_data, [0, 1, 1]), is_voice_cls)
    which_speaker = train(zip(training_data[1:], [1, 2]), which_cls)

    def classify(data):
        return is_voice.predict(data)*which_speaker.predict(data)
    
    return classify


def to_wav(mp4_path):
    with TemporaryDirectory() as d:
        wav_path = path.join(d, "out.wav")
        check_call(["ffmpeg", "-v", "0", "-i", mp4_path, wav_path])
        return wav_read(wav_path)


FPS = 30    
WINLEN = 1/FPS
    
def features(rate_sig, frames, features):
    rate, sig = rate_sig
    win_len = frames*WINLEN
    return mfcc(sig, rate, winlen=win_len, winstep=win_len, numcep=features).T


def batch(arr, fps, winlen=WINLEN):
    return array(map(median, array_split(arr, len(arr)//n)))


@click.command()
@click.argument('input-mp4')
@click.argument('output-mat')
@click.option("--silence")
@click.option("--speaker1")
@click.option("--speaker2")
@click.option("--num-features", default=13)
@click.option("--num-frames", default=1)
def main(input_mp4, output_mat, silence, speaker1, speaker2, num_features,
         num_frames):
    my_features = partial(features, frames=num_frames, features=num_features)
    wav_to_features = compose(my_features, wav_read)
    mp4_to_features = compose(my_features, to_wav)

    classify = train_classifiers([silence, speaker1, speaker2], 
                                 wav_to_features, LinearSVC, GaussianNB)
    
    savemat(output_mat, {"x": classify(mp4_to_features(input_mp4).T)})

if __name__ == '__main__':
    main()
