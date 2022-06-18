from random import randrange
import multiprocessing

import keras.optimizers
import ujson as ujson
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


def ai(f1: str, f2: str, t1: int, t2: int, input_size: int, output_size: int, learning_input: np.ndarray,
       learning_output: np.ndarray, testing_input: np.ndarray, testing_output: np.ndarray, genres_list: list):
    print("Hidden: ", f1, " (", t1, ") Output: ", f2, " (", t2, ")")
    model = Sequential()
    model.add(Dense(t1, input_dim=input_size, kernel_initializer='he_uniform', activation=f1))
    model.add(Dropout(0.1))
    model.add(Dense(t2, activation=f2))
    model.add(Dropout(0.1))
    model.add(Dense(int(t2 / 2), activation=f2))
    model.add(Dropout(0.1))
    # model.add(Dense(int(t2 / 4), activation=f2))
    # model.add(Dropout(0.1))
    # model.add(Dense(int(t2 / 8), activation=f2))
    # model.add(Dropout(0.1))

    model.add(Dense(output_size, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics='accuracy')
    model.fit(learning_input, learning_output, epochs=50, verbose=1)
    size = testing_input.shape
    predict = model.predict(testing_input)
    for i in range(size[0]):
        predicted_output = predict[i]
        actual_output = testing_output[i]
        predicted_genres = []
        actual_genres = []
        for j in range(predicted_output.size):
            if predicted_output[j] >= 0.5:
                predicted_genres.append(genres_list[j])
        for j in range(actual_output.size):
            if actual_output[j] == 1:
                actual_genres.append(genres_list[j])
        print("Predicted", predicted_genres, "Actual", actual_genres)
    loss, acc = model.evaluate(testing_input, testing_output, verbose=1)

    results = {

        'hidden_function': [f1],
        'hidden_tensors': [t1],
        'output_function': [f2],
        'epoch': [50],
        'accuracy': [acc]

    }
    df = pd.DataFrame(results)
    df.to_csv('new.csv', mode='a', index=False, header=False)
    print("acc: ", acc)


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
        # song_features = np.array([songs[song_id]['features'][0]['danceability'],
        #                           songs[song_id]['features'][0]['energy'],
        #                           songs[song_id]['features'][0]['key'],
        #                           songs[song_id]['features'][0]['loudness'],
        #                           songs[song_id]['features'][0]['mode'],
        #                           songs[song_id]['features'][0]['speechiness'],
        #                           songs[song_id]['features'][0]['acousticness'],
        #                           songs[song_id]['features'][0]['instrumentalness'],
        #                           songs[song_id]['features'][0]['liveness'],
        #                           songs[song_id]['features'][0]['valence'],
        #                           songs[song_id]['features'][0]['tempo'],
        #                           songs[song_id]['features'][0]['duration_ms'],
        #                           songs[song_id]['features'][0]['time_signature'],
        #                           int(songs[song_id]['track']['explicit'] == True),
        #                           songs[song_id]['track']['popularity']])
        names.append(song['name'])
        inputs_array.append(song_audio_analysis)
        genre_index = genres.index(songs[song_id]['genre'])
        outputs_array.append(genre_index)

    for i in range(int(len(inputs_array) * 0.2)):
        j = randrange(0, len(inputs_array))
        inputs_array_test.append(inputs_array.pop(j))
        outputs_array_test.append(outputs_array.pop(j))
        test_names.append(names.pop(j))

    np_inputs_array = np.array(inputs_array)  # X
    np_outputs_array = np.array(outputs_array)  # Y
    np_inputs_array_test = np.array(inputs_array_test)
    np_outputs_array_test = np.array(outputs_array_test)

    functions = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'elu', 'exponential']
    tensors = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    model = Sequential()
    model.add(Dense(200, input_dim=len(song_audio_analysis_labels), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(500, activation='relu'))

    # model.add(Dense(int(t2 / 4), activation=f2))
    # model.add(Dropout(0.1))
    # model.add(Dense(int(t2 / 8), activation=f2))
    # model.add(Dropout(0.1))

    model.add(Dense(len(genres), activation='softmax'))
    optimizer = keras.optimizers.Adam(learning_rate=1e-04)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(np_inputs_array, np_outputs_array, epochs=1500, verbose=1, batch_size=10)
    model.evaluate(np_inputs_array_test, np_outputs_array_test, verbose=1)
    size = np_inputs_array_test.shape
    predict = model.predict(np_inputs_array_test)
    for i in range(size[0]):
        predicted_output = predict[i]
        actual_output = np_outputs_array_test[i]
        predicted_genres = []
        name = test_names[i]
        for j in range(len(genres)):
            predicted_genres.append(genres[j] + ": " + str(round(predicted_output[j] * 100, 3)) + "%")
        actual_genre = genres[int(actual_output)]
        print(name, "Prediction", predicted_genres, "Actual", actual_genre)

    # kid = {"rmseP_a": 0.05808682972161509, "rmseP_std": 0.03350406150321627,
    #        "rmseH_a": 0.13447812560711736, "rmseH_std": 0.05039466668513202,
    #        "centroid_a": 3344.3060743423994, "centroid_std": 1066.724421117653,
    #        "bw_a": 3433.6940201801767, "bw_std": 678.0659407811223, "contrast_a": 17.193003000911276,
    #        "contrast_std": 6.2279061794874515, "polyfeat_a": -0.0003113733908023503,
    #        "polyfeat_std": 0.00010077385013213794, "tonnetz_a": 0.023218740693761446,
    #        "tonnetz_std": 0.09281101269685489, "zcr_a": 0.0815546400010301, "zcr_std": 0.018440167437094753,
    #        "onset_a": 1.4602575302124023, "onset_std": 1.4455164670944214, "bpm": 126.04801829268293,
    #        "rmseP_skew": 1.2909916612952492, "rmseP_kurtosis": 2.6971630993316795, "rmseH_skew": 0.30953916346091415,
    #        "rmseH_kurtosis": 1.4248659700088107, "beats_a": 87.85231982648132, "beats_std": 50.07328960320892}
    #
    # np_kid = np.array([kid['rmseP_a'],
    #                    kid['rmseP_std'],
    #                    kid['rmseH_a'],
    #                    kid['rmseH_std'],
    #                    kid['centroid_a'],
    #                    kid['centroid_std'],
    #                    kid['bw_a'],
    #                    kid['bw_std'],
    #                    kid['contrast_a'],
    #                    kid['contrast_std'],
    #                    kid['polyfeat_a'],
    #                    kid['polyfeat_std'],
    #                    kid['tonnetz_a'],
    #                    kid['tonnetz_std'],
    #                    kid['zcr_a'],
    #                    kid['zcr_std'],
    #                    kid['onset_a'],
    #                    kid['onset_std'],
    #                    kid['bpm'],
    #                    kid['rmseP_skew'],
    #                    kid['rmseP_kurtosis'],
    #                    kid['rmseH_skew'],
    #                    kid['rmseH_kurtosis'],
    #                    kid['beats_a'],
    #                    kid['beats_std']])
    # print(model.predict(np_kid))

    # p = multiprocessing.Process(target=ai, kwargs={
    #     "f1": 'relu',
    #     "f2": 'relu',
    #     "t1": 5000,
    #     "t2": 2000,
    #     "input_size": len(song_audio_analysis_labels) + len(song_features_labels),
    #     "output_size": len(genres),
    #     "learning_input": np_inputs_array,
    #     "learning_output": np_outputs_array,
    #     "testing_input": np_inputs_array_test,
    #     "testing_output": np_outputs_array_test,
    #     "genres_list": genres
    # })
    # p.start()
    # p.join()
    exit(0)
