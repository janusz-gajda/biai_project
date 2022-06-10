import spotipy
import json
import string
import random
import time

import ujson as ujson
from spotipy.oauth2 import SpotifyClientCredentials

from genres import Genres

# genres = Genres()
# genres.add("Rock")
# genres.add("Pop")
# genres.add("Rock")
# print(genres.toJSON())
from spotify import Spotify

spotify = Spotify()
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="0e68f0587b144b6bab4a8086ee5fea8e",
                                                           client_secret="e378892e60de4f43b9afde4d15a0184b"))



# results = spotify.random()
# artists_ids = []
#
# with open('data/genres.json', 'w') as genres_file:
#     genres_file.write(genres.toJSON())

track = {}
genres = Genres()
features = {}
analysis = {}

#print(json.dumps(song))




# for idx, track in enumerate(results['tracks']['items']):
#     for artist in enumerate(track['artists']):
#         artists_ids.append(artist[1]['id'])
# for artist_id in artists_ids:
#     artistsGenres = spotify.genres_from_artist(artist_id)
#     for genre in artistsGenres:
#         genres.add(genre)
songs_hashmap = {}
with open('data/total.json', 'r') as song_file:
    songs_hashmap = ujson.load(song_file)
    song_file.close()

print(str(len(songs_hashmap)) + " of 12k")
while len(songs_hashmap) < 12000:
    if len(songs_hashmap) % 10 == 0:
        print(str(len(songs_hashmap)) + " of 12k")
        with open('data/total.json', 'w') as song_file:
            song_file.write(ujson.dumps(songs_hashmap))
            song_file.close()
    song = spotify.get_random_features_only()
    songs_hashmap[song['track']['id']] = song
    time.sleep(1)

# songs_hashless = []
# for song_id in songs_hashmap.keys():
#     songs_hashless.append(songs_hashmap[song_id])


with open('data/total.json', 'w') as song_file:
    song_file.write(ujson.dumps(songs_hashmap))
