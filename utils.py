import ujson

import tensorflow as tf
import ujson as ujson
import json
import numpy as np
labels = ["rock", "pop", "hip-hop", "metal", "classic", "rap", "instrumental",
          "indie", "punk", "blues", "reggae", "country", "folk", "jazz", "house", "trap",
          "traditional", "alternative", "funk", "psychedelic", "power", "hardcore", "gospel",
          "orchestra", "soundtrack", "worship", "disco", "r&b", "soul"]


genres_dictionary = {}

all_genres = []
all_songs = {}
genres_in_dictionary = {}
genres_without_translation = []
genres_in_database_without_translation = {}

with open('data/old/genres_working.json', 'r') as file:
    all_genres = ujson.load(file)
    file.close()
count = 0
# for genre in all_genres:
#     translation = []
#     i = 0
#     for label in labels:
#         if label in genre:
#             i += 1
#     if i != 0:
#         count += 1
#     else:
#         genres_without_translation.append(genre)

a = np.float64(3.14159)
b = np.float32(3.14159)
c = np.float64(b)
print(count)
