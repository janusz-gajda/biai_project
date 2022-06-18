import spotipy
import json
import string
import random
import time
import ujson as ujson
import logging
import spotipy
from spotipy import SpotifyClientCredentials

logging.basicConfig(filename="sources.log", filemode='a', format='%(asctime)s %(levelname)s - %(message)s', level=logging.DEBUG)

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="0e68f0587b144b6bab4a8086ee5fea8e",
                                                           client_secret="e378892e60de4f43b9afde4d15a0184b"))


def serialize_tracks_with_url(genre, tracks):
    results = {}

    for track in tracks:
        if track['preview_url'] is None:
            continue

        artists = []

        for artist in track['track']['artists']:
            artists.append({
                "id": artist['id'],
                "name": artist["name"]
            })

        results[track['id']] = {
            "artists": artists,
            "album": {
                "id": track['album']['id'],
                "name": track['album']['name']
            },
            "url": track["preview_url"],
            "name": track["name"],
            "genre": genre
        }
    return results


def append_with_collisions_detection(tracks, collisions, new_tracks):
    for new_track_id in new_tracks.keys():
        if new_track_id in tracks:
            if tracks[new_track_id]['genre'] != new_tracks[new_track_id]['genre']:
                collisions.append({
                    "name": new_tracks[new_track_id]["name"],
                    "album": new_tracks[new_track_id]["album"],
                    "artists": new_tracks[new_track_id],
                    "genres": [tracks[new_track_id]['genre'], new_tracks[new_track_id]['genre']]
                })
        else:
            tracks[new_track_id] = new_tracks[new_track_id]


def main():
    with open("spotify_playlists.json", 'r') as file:
        playlists = json.load(file)
        file.close()

    print(playlists)
    collisions = []
    hashmap = {}
    no_url = {}
    url = {}

    # for playlist in playlists:
    #     if playlist["genre"] not in no_url:
    #         no_url[playlist["genre"]] = 0
    #         url[playlist["genre"]] = 0
    #     for playlist_id in playlist["ids"]:
    #         playlist_result = sp.playlist(playlist_id)
    #         for track in playlist_result["tracks"]["items"]:
    #             artists = []
    #             for artist in track['track']['artists']:
    #                 artists.append({
    #                     "id": artist['id'],
    #                     "name": artist["name"]
    #                 })
    #             track_info = {
    #                 "artists": artists,
    #                 "album": {
    #                     "id": track['track']['album']['id'],
    #                     "name": track['track']['album']['name']
    #                 },
    #                 "url": track['track']["preview_url"],
    #                 "name": track['track']["name"],
    #                 "genre": playlist["genre"]
    #             }
    #             if track["track"]["id"] in hashmap:
    #                 if hashmap[track["track"]["id"]]["genre"] != playlist["genre"]:
    #                     collisions.append({
    #                         "name": track_info["name"],
    #                         "album": track_info["album"],
    #                         "artists": artists,
    #                         "genres": [hashmap[track["track"]["id"]]["genre"], playlist["genre"]]
    #                     })
    #             else:
    #                 hashmap[track["track"]["id"]] = track_info
    #
    #
    # with open("../data/new_results.json", "w") as file:
    #     file.write(json.dumps(hashmap))
    #     file.close()
    # print(json.dumps(collisions, indent=4, sort_keys=False))
    #
    # for id in hashmap.keys():
    #     if hashmap[id]['url'] is None:
    #         no_url[hashmap[id]['genre']] += 1
    #     else:
    #         url[hashmap[id]['genre']] += 1
    #
    #
    # print("No url",no_url)
    # print("Url",url)

    offspring_id = "5LfGQac0EIXyAN8aUwmNAQ"

    offspring = sp.artist(offspring_id)

    offspring_albums = sp.artist_albums(offspring_id)

    offspring_top10 = sp.artist_top_tracks(offspring_id)

    playlist = sp.playlist_items("37i9dQZF1EQpj7X7UK8OOF?si=045424fb3bcb404c")


    with open("spotify_sources.json", "r") as file:
        sources = json.load(file)
        file.close()

    tracks = {}

    for genre in sources.keys():
        for artist_id in sources[genre]['artists']:
            artist_top10 = sp.artist_top_tracks(artist_id)
            results = serialize_tracks_with_url(genre, artist_top10)
            append_with_collisions_detection(tracks, collisions, results)
        #for playlist_id in sources[genre]['playlists']:

    abs(1)


if __name__ == "__main__":
    main()
