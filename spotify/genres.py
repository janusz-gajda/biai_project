import json
import string
from dataclasses import dataclass


class Genre:
    genre: string
    count: int

    def __init__(self, name):
        self.genre = name
        self.count = 1


class Genres:
    genres: list

    def __init__(self):
        self.genres = []

    def add(self, name):
        for genre in self.genres:
            if genre.genre == name:
                genre.count += 1
                return
        self.genres.append(Genre(name))

    def toJSON(self):
        self.genres.sort(key=lambda o: o.count, reverse=True)
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=True)
