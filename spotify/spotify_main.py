import spotipy
import json
import string
import random
import time
import ujson as ujson
import spotipy
from spotipy import SpotifyClientCredentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="0e68f0587b144b6bab4a8086ee5fea8e",
                                                           client_secret="e378892e60de4f43b9afde4d15a0184b"))

with open("spotify_playlists.json", 'r') as file:
    playlists = json.load(file)
    file.close()

print(playlists)
collisions = []
hashmap = {}
no_url = {}
url = {}

for playlist in playlists:
    if playlist["genre"] not in no_url:
        no_url[playlist["genre"]] = 0
        url[playlist["genre"]] = 0
    for playlist_id in playlist["ids"]:
        playlist_result = sp.playlist(playlist_id)
        for track in playlist_result["tracks"]["items"]:
            artists = []
            for artist in track['track']['artists']:
                artists.append({
                    "id": artist['id'],
                    "name": artist["name"]
                })
            track_info = {
                "artists": artists,
                "album": {
                    "id": track['track']['album']['id'],
                    "name": track['track']['album']['name']
                },
                "url": track['track']["preview_url"],
                "name": track['track']["name"],
                "genre": playlist["genre"]
            }
            if track["track"]["id"] in hashmap:
                if hashmap[track["track"]["id"]]["genre"] != playlist["genre"]:
                    collisions.append({
                        "name": track_info["name"],
                        "album": track_info["album"],
                        "artists": artists,
                        "genres": [hashmap[track["track"]["id"]]["genre"], playlist["genre"]]
                    })
            else:
                hashmap[track["track"]["id"]] = track_info


with open("../data/new_results.json", "w") as file:
    file.write(json.dumps(hashmap))
    file.close()
print(json.dumps(collisions, indent=4, sort_keys=False))

for id in hashmap.keys():
    if hashmap[id]['url'] is None:
        no_url[hashmap[id]['genre']] += 1
    else:
        url[hashmap[id]['genre']] += 1


print("No url",no_url)
print("Url",url)
