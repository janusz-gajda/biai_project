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

with open('./data/genres_working.json', 'r') as file:
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

for genre in all_genres:
    dict_entry = []
    if ("choir" or "chorus" or "orkester") in genre:
        dict_entry.append("choir")
    if "mexicana" in genre:
        dict_entry.append("mexican")
    if "electro" in genre:
        dict_entry.append("electronic")
    if "hip " in genre:
        dict_entry.append("hip-hop")
    for label in labels:
        if label in genre:
            if label == "rap" and ("trap" in genre):
                continue
            dict_entry.append(label)
    if len(dict_entry) > 0:
        genres_dictionary[genre] = dict_entry

with open('./data/genres_dictionary.json', 'w') as file:
    file.write(json.dumps(genres_dictionary, sort_keys=False, indent=4))
    file.close()

with open('./data/total.json', 'r') as file:
    all_songs = ujson.load(file)
    file.close()

count = 0

for song_id in all_songs:
    song_has_translation = False
    if all_songs[song_id]['genres'] is None:
        continue
    for genre in all_songs[song_id]['genres']:
        if genre not in genres_without_translation:
            song_has_translation = True
            break

    if not song_has_translation:
        for genre in all_songs[song_id]['genres']:
            # if genre not in genres_in_database_without_translation:
            #     genres_in_database_without_translation.append(genre)
            #     break
            if genre not in genres_in_database_without_translation:
                genres_in_database_without_translation[genre] = 1
            else:
                genres_in_database_without_translation[genre] += 1
        count += 1

with open('./data/database_without_translation.json', 'w') as file:
    file.write(json.dumps(genres_in_database_without_translation, sort_keys=False, indent=4))
    # for genre in genres_in_database_without_translation:
    #     file.write("%s\n" % genre)
    # file.close()
print(count)
