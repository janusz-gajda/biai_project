from random import randrange
import multiprocessing
import ujson as ujson
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

def ai(f1: str, f2: str, t1: int, t2: int, input_size: int, output_size: int, learning_input: np.ndarray,
       learning_output: np.ndarray, testing_input: np.ndarray, testing_output: np.ndarray):
    print("Hidden: ", f1, " (", t1, ") Output: ", f2, " (", t2, ")")
    model = Sequential()
    model.add(Dense(t1, input_dim=input_size, kernel_initializer='he_uniform', activation=f1))
    model.add(Dense(t2, activation=f2))
    model.add(Dense(int(t2 / 2), activation=f2))
    # model.add(Dense(int(t2 / 4), activation=f2))
    # model.add(Dense(int(t2 / 8), activation=f2))
    # model.add(Dropout(0.1))
    model.add(Dense(output_size, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics='accuracy')
    model.fit(learning_input, learning_output, epochs=50, verbose=1)
    weights = model.get_weights()
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
    song_labels = ['danceability', 'energy', 'key',
                   'loudness', 'mode', 'speechiness',
                   'acousticness', 'instrumentalness', 'liveness',
                   'valence', 'tempo', 'duration_ms',
                   'time_signature', 'explicit', 'popularity']
    with open("../data/genres_dictionary.json") as file:
        genres_dictionary = ujson.load(file)
        file.close()

    with open("../data/total.json", "r") as file:
        songs = ujson.load(file)
        file.close()

    for song_id in list(songs):
        song_genres = []
        for genre in songs[song_id]['genres']:
            if genre in genres_dictionary:
                for dict_genre in genres_dictionary[genre]:
                    if dict_genre not in song_genres:
                        song_genres.append(dict_genre)
                    if dict_genre not in genres:
                        genres.append(dict_genre)
        if len(song_genres) == 0:
            songs.pop(song_id, None)
        else:
            songs[song_id]['genres'] = song_genres


    print(str(len(songs)) + " valid songs have " + str(len(genres)) + " different genres")
    inputs_array = []  # X
    outputs_array = []  # Y
    inputs_array_test = []
    outputs_array_test = []
    for song_id in songs.keys():
        song = songs[song_id]
        if songs[song_id]['features'][0] is None:
            continue
        song_features = np.array([songs[song_id]['features'][0]['danceability'],
                                  songs[song_id]['features'][0]['energy'],
                                  songs[song_id]['features'][0]['key'],
                                  songs[song_id]['features'][0]['loudness'], songs[song_id]['features'][0]['mode'],
                                  songs[song_id]['features'][0]['speechiness'],
                                  songs[song_id]['features'][0]['acousticness'],
                                  songs[song_id]['features'][0]['instrumentalness'],
                                  songs[song_id]['features'][0]['liveness'],
                                  songs[song_id]['features'][0]['valence'],
                                  songs[song_id]['features'][0]['tempo'],
                                  songs[song_id]['features'][0]['duration_ms'],
                                  songs[song_id]['features'][0]['time_signature'],
                                  int(songs[song_id]['track']['explicit'] == True),
                                  songs[song_id]['track']['popularity']])
        song_genres = np.zeros(len(genres), dtype=int)
        for i in range(len(genres)):
            for genre in songs[song_id]['genres']:
                if genre == genres[i]:
                    song_genres[i] = 1
        inputs_array.append(song_features)
        outputs_array.append(song_genres)

    for i in range(int(len(inputs_array) * 0.2)):
        j = randrange(0, len(inputs_array))
        inputs_array_test.append(inputs_array.pop(j))
        outputs_array_test.append(outputs_array.pop(j))

    np_inputs_array = np.array(inputs_array)  # X
    np_outputs_array = np.array(outputs_array)  # Y
    np_inputs_array_test = np.array(inputs_array_test)
    np_outputs_array_test = np.array(outputs_array_test)

    functions = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'elu', 'exponential']
    tensors = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    p = multiprocessing.Process(target=ai, kwargs={
        "f1": 'relu',
        "f2": 'relu',
        "t1": 500,
        "t2": 200,
        "input_size": len(song_labels),
        "output_size": len(genres),
        "learning_input": np_inputs_array,
        "learning_output": np_outputs_array,
        "testing_input": np_inputs_array_test,
        "testing_output": np_outputs_array_test
    })
    p.start()
    p.join()
    exit(0)

