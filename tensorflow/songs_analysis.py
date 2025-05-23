import io
import json
import subprocess
import pandas as pd
import seaborn as sns
import numpy as np
import os
import librosa as lb
import glob
from pprint import pprint as pp
import scipy
import matplotlib.pyplot as plt
import soundfile as sf
# plt.rcParams["figure.figsize"] = (14,5)

# from more_itertools import unique_everseen
from collections import OrderedDict
# import soundfile as sf
# import difflib
import statistics
import itertools

import ujson
from scipy.stats import skew
from scipy.stats import kurtosis


def get_features_mean(song_data, sample_rate, hop_length, n_fft):
    # try:
    y_harmonic, y_percussive = lb.effects.hpss(song_data)  # split song into harmonic and percussive parts
    stft_harmonic = lb.core.stft(y_harmonic, n_fft=n_fft, hop_length=hop_length)  # Compute power spectrogram.
    stft_percussive = lb.core.stft(y_percussive, n_fft=n_fft, hop_length=hop_length)  # Compute power spectrogram.

    rmsH = np.sqrt(np.mean(np.abs(lb.feature.rms(S=stft_harmonic)) ** 2, axis=0, keepdims=True))
    rmsH_a = np.mean(rmsH)
    rmsH_std = np.std(rmsH)
    rmsH_skew = skew(np.mean(rmsH, axis=0))
    rmsH_kurtosis = kurtosis(np.mean(rmsH, axis=0), fisher=True, bias=True)

    rmsP = np.sqrt(np.mean(np.abs(lb.feature.rms(S=stft_percussive)) ** 2, axis=0, keepdims=True))
    rmsP_a = np.mean(rmsP)
    rmsP_std = np.std(rmsP)
    rmsP_skew = skew(np.mean(rmsP, axis=0))
    rmsP_kurtosis = kurtosis(np.mean(rmsP, axis=0), fisher=True, bias=True)

    centroid = lb.feature.spectral_centroid(y=song_data, sr=sample_rate, n_fft=n_fft,
                                            hop_length=hop_length)  # Compute the spectral centroid.
    centroid_a = np.mean(centroid)
    centroid_std = np.std(centroid)

    bw = lb.feature.spectral_bandwidth(y=song_data, sr=sample_rate, n_fft=n_fft,
                                       hop_length=hop_length)  # Compute p’th-order spectral bandwidth:
    bw_a = np.mean(bw)
    bw_std = np.std(bw)

    contrast = lb.feature.spectral_contrast(y=song_data, sr=sample_rate, n_fft=n_fft,
                                            hop_length=hop_length)  # Compute spectral contrast [R16]
    contrast_a = np.mean(contrast)
    contrast_std = np.std(contrast)

    polyfeat = lb.feature.poly_features(y=y_harmonic, sr=sample_rate, n_fft=n_fft,
                                        hop_length=hop_length)  # Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
    polyfeat_a = np.mean(polyfeat[0])
    polyfeat_std = np.std(polyfeat[0])

    tonnetz = lb.feature.tonnetz(y=lb.effects.harmonic(y_harmonic),
                                 sr=sample_rate)  # Computes the tonal centroid features (tonnetz), following the method of [R17].
    tonnetz_a = np.mean(tonnetz)
    tonnetz_std = np.std(tonnetz)

    zcr = lb.feature.zero_crossing_rate(y=song_data, frame_length=sample_rate, hop_length=hop_length)  # zero crossing rate
    zcr_a = np.mean(zcr)
    zcr_std = np.std(zcr)

    onset_env = lb.onset.onset_strength(y=y_percussive, sr=sample_rate)
    onset_a = np.mean(onset_env)
    onset_std = np.std(onset_env)

    D = lb.stft(song_data)
    times = lb.frames_to_time(
        np.arange(D.shape[1]))  # not returned, but could be if you want to plot things as a time series

    bpm, beats = lb.beat.beat_track(y=y_percussive, sr=sample_rate, onset_envelope=onset_env, units='time')
    beats_a = np.mean(beats)
    beats_std = np.std(beats)

    features_dict = {'rmseP_a': rmsP_a, 'rmseP_std': rmsP_std, 'rmseH_a': rmsH_a, 'rmseH_std': rmsH_std,
                     'centroid_a': centroid_a, 'centroid_std': centroid_std, 'bw_a': bw_a, 'bw_std': bw_std,
                     'contrast_a': contrast_a, 'contrast_std': contrast_std, 'polyfeat_a': polyfeat_a,
                     'polyfeat_std': polyfeat_std, 'tonnetz_a': tonnetz_a, 'tonnetz_std': tonnetz_std,
                     'zcr_a': zcr_a, 'zcr_std': zcr_std, 'onset_a': np.float64(onset_a), 'onset_std': np.float64(onset_std),
                     'bpm': bpm, 'rmseP_skew': rmsP_skew, 'rmseP_kurtosis': rmsP_kurtosis,
                     'rmseH_skew': rmsH_skew, 'rmseH_kurtosis': rmsH_kurtosis, 'beats_a': beats_a,
                     'beats_std': beats_std}

    # combine_features = {**features_dict, **bands_dict}
    return features_dict


if __name__ == "__main__":
    songs = {}
    with open("../data/total.json", "r") as file:
        songs = ujson.load(file)
        file.close()

    files = [f for f in glob.glob("..\\songs\\*.wav", recursive=True)]

    wav_data = {}
    i = 1
    for file in files:
        print(i, "of", len(files))
        i += 1
        song_id = file.split('\\')[-1].split('.')[0]
        y, sr = lb.load(file, sr=44100)
        # wav_data[song_id] = {'y': y, 'sr': sr}
        res = get_features_mean(y, sr, hop_length=512, n_fft=2048)
        songs[song_id]['audio_analysis'] = res
        print("Song", songs[song_id]['track']['name'], "done")
        if i % 10 == 0:
            with open("../data/total_with_analysis.json", 'w') as file:
                file.write(json.dumps(songs, separators=(',', ':')))
                file.close()
                print("Results saved")

    with open("../data/total_with_analysis.json", 'w') as file:
        file.write(json.dumps(songs, separators=(',', ':')))
        file.close()
    abs(1)
