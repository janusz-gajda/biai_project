import random
import string

import spotipy
from spotipy import SpotifyClientCredentials


class Spotify:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="0e68f0587b144b6bab4a8086ee5fea8e",
                                                               client_secret="e378892e60de4f43b9afde4d15a0184b"))

    def random(self):
        letter = random.choice(string.ascii_lowercase)
        offset = random.randint(0, 999)
        rand = random.randint(0, 2)
        if rand == 0:
            pattern = letter + "%"
        elif rand == 1:
            pattern = "%" + letter
        else:
            pattern = "%" + letter + "%"

        return self.sp.search(q="track:" + pattern, limit=1, offset=offset)

    def random_song(self):
        data = self.random()
        results = {}
        for track in data['tracks']['items']:
            results['track_id'] = track['id']
            results['artists_ids'] = []
            for artist in track['artists']:
                results['artists_ids'].append(artist['id'])
            return results

    def genres_from_artists(self, artist_ids):
        genres = []
        for artist_id in artist_ids:
            results = self.sp.artist(artist_id)
            for new_genre in results['genres']:
                if new_genre not in genres:
                    genres.append(new_genre)
        return genres

    def get_random_learning(self):
        song = {}
        track_ids = self.random_song()
        song['track'] = self.sp.track(track_ids['track_id'])
        del song['track']['artists']
        del song['track']['album']
        song['features'] = self.sp.audio_features(track_ids['track_id'])
        song['analysis'] = self.sp.audio_analysis(track_ids['track_id'])
        song['genres'] = self.genres_from_artists(track_ids['artists_ids'])
        return song

    def get_random_learning(self):
        song = {}
        track_ids = self.random_song()
        song['track'] = self.sp.track(track_ids['track_id'])
        del song['track']['artists']
        del song['track']['album']
        song['features'] = self.sp.audio_features(track_ids['track_id'])
        song['analysis'] = self.sp.audio_analysis(track_ids['track_id'])
        return song
