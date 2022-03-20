import spotipy
import json
import string
import random
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="0e68f0587b144b6bab4a8086ee5fea8e",
                                                           client_secret="e378892e60de4f43b9afde4d15a0184b"))

letter = random.choice(string.ascii_lowercase)
offset = random.randint(0, 1000)
rand = random.randint(0, 2)
if rand == 0:
    pattern = letter + "%"
elif rand == 1:
    pattern = "%" + letter
else:
    pattern = "%" + letter + "%"

results = sp.search(q="track:" + pattern, limit=1, offset=offset)
# print(json.dumps(results, indent=4, sort_keys=False))
print("Pattern: " + pattern + " Offset: " + str(offset))

artist_ids = []

for idx, track in enumerate(results['tracks']['items']):
    print(track['name'] + " " + track['id'])
    for artist in enumerate(track['artists']):
        artist_ids.append(artist[1]['id'])

for artist_id in artist_ids:
    print(artist_id + " ")

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
