#!/usr/bin/env python2

from __future__ import division

from operator import itemgetter, mul
from subprocess import check_call
from contextlib import contextmanager
from os import path
import tempfile
import shutil

import click
import numpy as np
from matplotlib.pyplot import savefig, scatter, legend, title
from funcy import pluck, partial, compose
from numpy import hstack, ones, log, transpose
from numpy.fft import fft
from numpy.random import shuffle
from scipy.io.wavfile import read as wav_read
from scipy.io import savemat
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from features import mfcc


WIN_SIZE = 44100
HOP = int((1-3/4) * WIN_SIZE)

def windows(x, win_size=WIN_SIZE, hop=HOP):
    num_hops = int(np.floor((len(x)-win_size)/hop))
    steps = (i*hop for i in range(num_hops))
    return [x[step:win_size+step] for step in steps]

def log_spectrogram(x, win_size=WIN_SIZE, hop=HOP):
    to_freq = compose(np.transpose, np.matrix, np.fft.fft,
                      partial(mul, np.hamming(win_size)))

    return log(np.abs(np.array(np.hstack(map(to_freq, windows(x, win_size, hop)))))[win_size/2:,:])


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
    partitioned = [(x[:,:-10], x[:,-10:]) for x in data]
    train, test = zip(*partitioned)
    return train, test


def train(data_and_labels, cls):
    X, dense_labels = dense_data_and_labels(data_and_labels)
    classifier = cls()
    return classifier.fit(X, dense_labels)


def gen_stats(data, training, test, n_speakers, is_voice, which_speaker):
    pca = PCA(n_components=2)
    pca.fit(hstack(data).T)
    pca_data = map(compose(transpose, pca.transform, transpose), data)

    for color, sp in zip(["red", "blue", "green"], pca_data):
        scatter(sp[0], sp[1], color=color)
        legend(["silence", "speaker1", "speaker2"], loc='lower left',)
        title("PCA 2 Components of Log Spectrogram")
    savefig("pca_2_comp_log_spec.svg")
    
    X2, labels2 = dense_data_and_labels(zip(test, [0] + n_speakers*[1]))
    score1 = is_voice.score(X2, labels2)
    
    X3, labels3 = dense_data_and_labels(zip(test[1:], range(1, n_speakers+1)))
    score2 = which_speaker.score(X3, labels3)

    print(score1, score2)

    savemat("data.mat", {"sp" + str(i): elem for i, elem in enumerate(data)})



def train_classifiers(training_wavs, load_bucket, is_voice_cls, which_cls,
                      verbose=False):
    data = map(load_bucket, training_wavs)
    training, test = split_data(data)
    n_speakers = len(training_wavs)-1
    is_voice = train(zip(training, [0] + n_speakers*[1]), is_voice_cls)
    which_speaker = train(zip(training[1:], range(1, n_speakers+1)), which_cls)

    if verbose:
        gen_stats(data, training, test, n_speakers, is_voice, which_speaker)
        

    def classify(data):
        return is_voice.predict(data), which_speaker.predict_log_proba(data)

    return classify


def to_wav(mp4_path):
    with TemporaryDirectory() as d:
        wav_path = path.join(d, "out.wav")
        check_call(["ffmpeg", "-v", "0", "-i", mp4_path, wav_path])
        return wav_read(wav_path)


def features(rate_sig, frames, features, fps, use_mfcc=False):
    rate, sig = rate_sig
    win_len = frames*(1/fps)
    if use_mfcc:
        y = mfcc(sig[:, 0], rate, winlen=win_len, winstep=win_len, numcep=features).T
    else:
        win_len = int(win_len*rate)
        y = log_spectrogram(sig[:, 0], win_size=win_len, hop=win_len)
    return y


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
@click.option("--log-spec/--mfcc", default=True)
def main(input_mp4, output_mat, noise, speaker, num_features,
         num_frames, verbose, fps, log_spec):
    my_features = partial(features, frames=num_frames, features=num_features,
                          fps=fps, use_mfcc=not log_spec)
    wav_to_features = compose(my_features, wav_read)
    mp4_to_features = compose(my_features, to_wav)

    classify = train_classifiers([noise] + list(speaker),
                                 wav_to_features, LinearSVC, GaussianNB,
                                 verbose)

    mp4_features = mp4_to_features(input_mp4).T
    
    if verbose:
        savemat("all_features.mat", {"all_features": mp4_features})
    
    voice, person = classify(mp4_features)
    savemat(output_mat, {"voice": voice, "person": person})

if __name__ == '__main__':
    main()
