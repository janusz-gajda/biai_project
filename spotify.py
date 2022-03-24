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

    def genres_from_artist(self, artist_id):
        genres = []
        results = self.sp.artist(artist_id)
        for genre in results['genres']:
            genres.append(genre)
        return genres
