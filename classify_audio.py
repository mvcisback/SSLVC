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
from numpy.random import shuffle
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


def split_data(data):
    map(shuffle, data)
    partitioned = [(x[:, :-100], x[:, -100:]) for x in data]
    train, test = zip(*partitioned)
    return train, test


def train(data_and_labels, cls):
    X, dense_labels = dense_data_and_labels(data_and_labels)
    classifier = cls()
    return classifier.fit(X, dense_labels)


def train_classifiers(training_wavs, load_bucket, is_voice_cls, which_cls,
                      verbose=False):
    data = map(load_bucket, training_wavs)
    training, test = split_data(data)
    is_voice = train(zip(training, [0, 1, 1]), is_voice_cls)
    which_speaker = train(zip(training[1:], [1, 2]), which_cls)

    X2, labels2 = dense_data_and_labels(zip(test, [0, 1, 1]))
    score1 = is_voice.score(X2, labels2)

    X3, labels3 = dense_data_and_labels(zip(test[1:], [1, 2]))
    score2 = which_speaker.score(X3, labels3)

    if verbose:
        print(score1, score2)

    def classify(data):
        return is_voice.predict(data)*which_speaker.predict(data)

    return classify


def to_wav(mp4_path):
    with TemporaryDirectory() as d:
        wav_path = path.join(d, "out.wav")
        check_call(["ffmpeg", "-v", "0", "-i", mp4_path, wav_path])
        return wav_read(wav_path)


def features(rate_sig, frames, features, fps):
    rate, sig = rate_sig
    win_len = frames*(1/fps)
    return mfcc(sig, rate, winlen=win_len, winstep=win_len, numcep=features).T


@click.command()
@click.argument('input-mp4', type=click.Path(exists=True))
@click.argument('output-mat', type=click.Path())
@click.option("--noise", type=click.Path(exists=True), required=True)
@click.option("--speaker", type=click.Path(exists=True), required=True, 
              multiple=True)
@click.option("--num-features", default=13, type=click.IntRange(min=1))
@click.option("--num-frames", default=1, type=click.IntRange(min=1))
@click.option("--verbose/--silent", default=False)
@click.option("--fps", default=30, type=click.IntRange(min=1))
def main(input_mp4, output_mat, noise, speaker, num_features,
         num_frames, verbose, fps):
    my_features = partial(features, frames=num_frames, features=num_features,
                          fps=fps)
    wav_to_features = compose(my_features, wav_read)
    mp4_to_features = compose(my_features, to_wav)

    classify = train_classifiers([noise] + list(speaker),
                                 wav_to_features, LinearSVC, GaussianNB,
                                 verbose)

    savemat(output_mat, {"x": classify(mp4_to_features(input_mp4).T)})

if __name__ == '__main__':
    main()
