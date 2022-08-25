import json
import sys
from random import randrange

import keras.optimizers
import ujson as ujson
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
import librosa as lb
from scipy.stats import skew
from scipy.stats import kurtosis
import os
import tensorflow as tf
from tqdm import tqdm
import warnings
import hashlib

BLOCK_SIZE = 65536

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.filterwarnings("ignore")

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

    zcr = lb.feature.zero_crossing_rate(y=song_data, frame_length=sample_rate,
                                        hop_length=hop_length)  # zero crossing rate
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
                     'zcr_a': zcr_a, 'zcr_std': zcr_std, 'onset_a': np.float64(onset_a),
                     'onset_std': np.float64(onset_std),
                     'bpm': bpm, 'rmseP_skew': rmsP_skew, 'rmseP_kurtosis': rmsP_kurtosis,
                     'rmseH_skew': rmsH_skew, 'rmseH_kurtosis': rmsH_kurtosis, 'beats_a': beats_a,
                     'beats_std': beats_std}

    # combine_features = {**features_dict, **bands_dict}
    return features_dict

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    genres = []
    genres_count = {}
    song_features_labels = ['danceability', 'energy', 'key',
                            'loudness', 'mode', 'speechiness',
                            'acousticness', 'instrumentalness', 'liveness',
                            'valence', 'tempo', 'duration_ms',
                            'time_signature', 'explicit', 'popularity']
    song_audio_analysis_labels = ['rmseP_a', 'rmseP_std', 'rmseH_a', 'rmseH_std',
                                  'centroid_a', 'centroid_std', 'bw_a', 'bw_std',
                                  'contrast_a', 'contrast_std', 'polyfeat_a',
                                  'polyfeat_std', 'tonnetz_a', 'tonnetz_std',
                                  'zcr_a', 'zcr_std', 'onset_a', 'onset_std',
                                  'bpm', 'rmseP_skew', 'rmseP_kurtosis',
                                  'rmseH_skew', 'rmseH_kurtosis', 'beats_a',
                                  'beats_std']
    with open("../data/data.json", "r") as file:
        songs = ujson.load(file)
        file.close()
    for song_id in list(songs):
        if songs[song_id]['genre'] not in genres:
            genres.append(songs[song_id]['genre'])
    if len(sys.argv) < 2:
        print(str(len(songs)) + " valid songs have " + str(len(genres)) + " different genres")
        inputs_array = []  # X
        outputs_array = []  # Y
        inputs_array_test = []
        outputs_array_test = []
        names = []
        test_names = []
        one_pop = False
        for song_id in songs.keys():
            song = songs[song_id]
            song_audio_analysis = np.array([songs[song_id]['audio_analysis']['rmseP_a'],
                                            songs[song_id]['audio_analysis']['rmseP_std'],
                                            songs[song_id]['audio_analysis']['rmseH_a'],
                                            songs[song_id]['audio_analysis']['rmseH_std'],
                                            songs[song_id]['audio_analysis']['centroid_a'],
                                            songs[song_id]['audio_analysis']['centroid_std'],
                                            songs[song_id]['audio_analysis']['bw_a'],
                                            songs[song_id]['audio_analysis']['bw_std'],
                                            songs[song_id]['audio_analysis']['contrast_a'],
                                            songs[song_id]['audio_analysis']['contrast_std'],
                                            songs[song_id]['audio_analysis']['polyfeat_a'],
                                            songs[song_id]['audio_analysis']['polyfeat_std'],
                                            songs[song_id]['audio_analysis']['tonnetz_a'],
                                            songs[song_id]['audio_analysis']['tonnetz_std'],
                                            songs[song_id]['audio_analysis']['zcr_a'],
                                            songs[song_id]['audio_analysis']['zcr_std'],
                                            songs[song_id]['audio_analysis']['onset_a'],
                                            songs[song_id]['audio_analysis']['onset_std'],
                                            songs[song_id]['audio_analysis']['bpm'],
                                            songs[song_id]['audio_analysis']['rmseP_skew'],
                                            songs[song_id]['audio_analysis']['rmseP_kurtosis'],
                                            songs[song_id]['audio_analysis']['rmseH_skew'],
                                            songs[song_id]['audio_analysis']['rmseH_kurtosis'],
                                            songs[song_id]['audio_analysis']['beats_a'],
                                            songs[song_id]['audio_analysis']['beats_std']])
            names.append(song['name'])
            inputs_array.append(song_audio_analysis)
            genre_index = genres.index(songs[song_id]['genre'])
            outputs_array.append(genre_index)

        # for i in range(int(len(inputs_array) * 0.2)):
        #     j = randrange(0, len(inputs_array))
        #     inputs_array_test.append(inputs_array.pop(j))
        #     outputs_array_test.append(outputs_array.pop(j))
        #     test_names.append(names.pop(j))

        np_inputs_array = np.array(inputs_array)  # X
        np_outputs_array = np.array(outputs_array)  # Y
        # np_inputs_array_test = np.array(inputs_array_test)
        # np_outputs_array_test = np.array(outputs_array_test)

        model = Sequential()
        model.add(Dense(200, input_dim=len(song_audio_analysis_labels), activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(len(genres), activation='softmax'))
        optimizer = keras.optimizers.Adam(learning_rate=1e-04)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(np_inputs_array, np_outputs_array, epochs=1500, verbose=1, batch_size=16)
        model.save("./model")
    else:
        test_array = []
        with open('../data/cache.json', 'r') as file:
            cache = json.load(file)
            file.close()

        for i in tqdm(range(1, len(sys.argv))):
            #print("Analysing", sys.argv[i])
            sha = hashlib.sha256()
            with open(sys.argv[i], 'rb') as file:
                file_block = file.read(BLOCK_SIZE)
                while len(file_block) > 0:
                    sha.update(file_block)
                    file_block = file.read(BLOCK_SIZE)

            if sha.hexdigest() in cache:
                res = cache[sha.hexdigest()]
            else:
                y, sr = lb.load(sys.argv[i], sr=44100)
                res = get_features_mean(y, sr, hop_length=512, n_fft=2048)
                cache[sha.hexdigest()] = res
            song_audio_analysis = np.array([res['rmseP_a'], res['rmseP_std'], res['rmseH_a'],
                                            res['rmseH_std'], res['centroid_a'], res['centroid_std'],
                                            res['bw_a'], res['bw_std'], res['contrast_a'],
                                            res['contrast_std'], res['polyfeat_a'], res['polyfeat_std'],
                                            res['tonnetz_a'], res['tonnetz_std'], res['zcr_a'],
                                            res['zcr_std'], res['onset_a'], res['onset_std'],
                                            res['bpm'], res['rmseP_skew'], res['rmseP_kurtosis'],
                                            res['rmseH_skew'], res['rmseH_kurtosis'], res['beats_a'],
                                            res['beats_std']])
            test_array.append(song_audio_analysis)

        with open('../data/cache.json', 'w') as file:
            file.write(json.dumps(cache, separators=(',', ':')))
            file.close()

        np_test_array = np.array(test_array)
        model = load_model("./model")
        predict = model.predict(np_test_array)

        for i in range(0, len(sys.argv) - 1):
            predicted_output = predict[i]
            predicted_genres =[]
            for j in range(len(genres)):
                predicted_genres.append(genres[j] + ": " + str(round(predicted_output[j] * 100, 3)) + "%")
            print("File:", sys.argv[i+1], "prediction:", predicted_genres)

        # model.evaluate(np_inputs_array_test, np_outputs_array_test, verbose=1)
        # size = np_inputs_array_test.shape
        # predict = model.predict(np_inputs_array_test)
    # for i in range(size[0]):
    #     predicted_output = predict[i]
    #     actual_output = np_outputs_array_test[i]
    #     predicted_genres = []
    #     name = test_names[i]
    #     for j in range(len(genres)):
    #         predicted_genres.append(genres[j] + ": " + str(round(predicted_output[j] * 100, 3)) + "%")
    #     actual_genre = genres[int(actual_output)]
    #     print(name, "Prediction", predicted_genres, "Actual", actual_genre)
    exit(0)
