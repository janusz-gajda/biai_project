import spotipy
import json
import string
import random
from spotipy.oauth2 import SpotifyClientCredentials

from genres import Genres

# genres = Genres()
# genres.add("Rock")
# genres.add("Pop")
# genres.add("Rock")
# print(genres.toJSON())
from spotify import Spotify

spotify = Spotify()

genres = Genres()

for i in range(1, 8000):
    results = spotify.random()
    artists_ids = []
    number = round(i / 800, 2)
    print(i)
    for idx, track in enumerate(results['tracks']['items']):

        for artist in enumerate(track['artists']):
            artists_ids.append(artist[1]['id'])
    for artist_id in artists_ids:
        artistsGenres = spotify.genres_from_artist(artist_id)
        for genre in artistsGenres:
            genres.add(genre)

with open('data/genres.json', 'w') as genres_file:
    genres_file.write(genres.toJSON())



# results = sp.search(q='offspring', limit=20)
# results = sp.artist('5LfGQac0EIXyAN8aUwmNAQ')
# with open('data/artist.json', 'w') as artist_file:
#     artist_file.write(json.dumps(results, indent=4, sort_keys=False))
#
# results = sp.track('6TfBA04WJ3X1d1wXhaCFVT')
#
# with open('data/track.json', 'w') as track_file:
#     track_file.write(json.dumps(results, indent=4, sort_keys=False))
#
# results = sp.audio_analysis('6TfBA04WJ3X1d1wXhaCFVT')
#
# with open('data/analysis.json', 'w') as analysis_ile:
#     analysis_ile.write(json.dumps(results, indent=4, sort_keys=False))
#
# results = sp.audio_features('6TfBA04WJ3X1d1wXhaCFVT')
#
# with open('data/features.json', 'w') as features_file:
#     features_file.write(json.dumps(results, indent=4, sort_keys=False))
