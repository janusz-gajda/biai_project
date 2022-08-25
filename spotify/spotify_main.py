import glob
import subprocess

import spotipy
import json
import string
import random
import time
import ujson as ujson
import logging
import spotipy
from spotipy import SpotifyClientCredentials
import librosa as lb
import numpy as np
import sys
import threading
import queue
import glob
from scipy.stats import skew
from scipy.stats import kurtosis
from tqdm import tqdm

logging.basicConfig(filename="sources.log", filemode='w', format='%(asctime)s %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="0e68f0587b144b6bab4a8086ee5fea8e",
                                                           client_secret="e378892e60de4f43b9afde4d15a0184b"))


def artists_to_string(artists):
    artists_str = ""
    for artist in artists:
        if artists_str == "":
            artists_str = artist['name']
        else:
            artists_str += (", " + artist["name"])
    return artists_str


def serialize_tracks_with_url(genre, tracks):
    results = {}
    logger.info("Started serializing tracks")
    for track in tracks:
        logger.info("Track: %s By %s", track["name"], artists_to_string(track['artists']))
        artists = []
        for artist in track['artists']:
            artists.append({
                "id": artist['id'],
                "name": artist["name"]
            })

        if track['preview_url'] is None:
            logger.warning("Song doesn't have preview url; skipping...")
            continue

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
        logger.info("Song successfully added to results")
    logger.info("Returning results")
    return results


def append_with_collisions_detection(tracks, collisions, new_tracks):
    logger.info("Starting collision detection")
    for new_track_id in new_tracks.keys():
        if new_track_id in tracks:
            if tracks[new_track_id]['genre'] != new_tracks[new_track_id]['genre']:
                logger.error('Song %s By %s appears at least twice in database and has different genres ( %s vs %s )' %
                             (new_tracks[new_track_id]['name'], artists_to_string(new_tracks[new_track_id]['artists']),
                              tracks[new_track_id]['genre'], new_tracks[new_track_id]['genre']))
                collisions.append({
                    "name": new_tracks[new_track_id]["name"],
                    "album": new_tracks[new_track_id]["album"],
                    "artists": new_tracks[new_track_id]['artists'],
                    "genres": [tracks[new_track_id]['genre'], new_tracks[new_track_id]['genre']]
                })
            else:
                logger.warning("Song %s By %s appears at least twice in database but has the same genre",
                               new_tracks[new_track_id]['name'], artists_to_string(new_tracks[new_track_id]['artists']))
        else:
            tracks[new_track_id] = new_tracks[new_track_id]
    logger.info("Collision detection completed")


def get_features_mean(song_data, sample_rate, hop_length, n_fft):
    # try:
    y_harmonic, y_percussive = lb.effects.hpss(song_data)  # split song into harmonic and percussive parts
    stft_harmonic = lb.core.stft(y_harmonic, n_fft=n_fft, hop_length=hop_length)  # Compute power spectrogram.
    stft_percussive = lb.core.stft(y_percussive, n_fft=n_fft, hop_length=hop_length)  # Compute power spectrogram.

    rmsH = np.sqrt(np.mean(np.abs(lb.feature.rms(S=stft_harmonic)) ** 2, axis=0, keepdims=True))
    rmsH_a = np.mean(rmsH)
    rmsH_std = np.std(rmsH)
    rmsH_skew = skew(np.mean(rmsH, axis=0))
    rmsH_kurtosis = kurtosis(np.mean(rmsH, axis=0), fisher=True, bias=True)

    rmsP = np.sqrt(np.mean(np.abs(lb.feature.rms(S=stft_percussive)) ** 2, axis=0, keepdims=True))
    rmsP_a = np.mean(rmsP)
    rmsP_std = np.std(rmsP)
    rmsP_skew = skew(np.mean(rmsP, axis=0))
    rmsP_kurtosis = kurtosis(np.mean(rmsP, axis=0), fisher=True, bias=True)

    centroid = lb.feature.spectral_centroid(y=song_data, sr=sample_rate, n_fft=n_fft,
                                            hop_length=hop_length)  # Compute the spectral centroid.
    centroid_a = np.mean(centroid)
    centroid_std = np.std(centroid)

    bw = lb.feature.spectral_bandwidth(y=song_data, sr=sample_rate, n_fft=n_fft,
                                       hop_length=hop_length)  # Compute p’th-order spectral bandwidth:
    bw_a = np.mean(bw)
    bw_std = np.std(bw)

    contrast = lb.feature.spectral_contrast(y=song_data, sr=sample_rate, n_fft=n_fft,
                                            hop_length=hop_length)  # Compute spectral contrast [R16]
    contrast_a = np.mean(contrast)
    contrast_std = np.std(contrast)

    polyfeat = lb.feature.poly_features(y=y_harmonic, sr=sample_rate, n_fft=n_fft,
                                        hop_length=hop_length)  # Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
    polyfeat_a = np.mean(polyfeat[0])
    polyfeat_std = np.std(polyfeat[0])

    tonnetz = lb.feature.tonnetz(y=lb.effects.harmonic(y_harmonic),
                                 sr=sample_rate)  # Computes the tonal centroid features (tonnetz), following the method of [R17].
    tonnetz_a = np.mean(tonnetz)
    tonnetz_std = np.std(tonnetz)

    zcr = lb.feature.zero_crossing_rate(y=song_data, frame_length=sample_rate,
                                        hop_length=hop_length)  # zero crossing rate
    zcr_a = np.mean(zcr)
    zcr_std = np.std(zcr)

    onset_env = lb.onset.onset_strength(y=y_percussive, sr=sample_rate)
    onset_a = np.mean(onset_env)
    onset_std = np.std(onset_env)

    D = lb.stft(song_data)
    times = lb.frames_to_time(
        np.arange(D.shape[1]))  # not returned, but could be if you want to plot things as a time series

    bpm, beats = lb.beat.beat_track(y=y_percussive, sr=sample_rate, onset_envelope=onset_env, units='time')
    beats_a = np.mean(beats)
    beats_std = np.std(beats)

    features_dict = {'rmseP_a': rmsP_a, 'rmseP_std': rmsP_std, 'rmseH_a': rmsH_a, 'rmseH_std': rmsH_std,
                     'centroid_a': centroid_a, 'centroid_std': centroid_std, 'bw_a': bw_a, 'bw_std': bw_std,
                     'contrast_a': contrast_a, 'contrast_std': contrast_std, 'polyfeat_a': polyfeat_a,
                     'polyfeat_std': polyfeat_std, 'tonnetz_a': tonnetz_a, 'tonnetz_std': tonnetz_std,
                     'zcr_a': zcr_a, 'zcr_std': zcr_std, 'onset_a': np.float64(onset_a),
                     'onset_std': np.float64(onset_std),
                     'bpm': bpm, 'rmseP_skew': rmsP_skew, 'rmseP_kurtosis': rmsP_kurtosis,
                     'rmseH_skew': rmsH_skew, 'rmseH_kurtosis': rmsH_kurtosis, 'beats_a': beats_a,
                     'beats_std': beats_std}

    # combine_features = {**features_dict, **bands_dict}
    return features_dict


def analyser_thread(thread_id, input_queue, output_queue):
    logger.info("Thread %d started", thread_id)
    while True:
        if input_queue.empty():
            logger.info("Thread %d: Input queue is empty; exiting...", thread_id)
            exit(0)
        song = input_queue.get()
        song_id = song.split('\\')[-1].split('.')[0]
        logger.info("Thread %d: Analyzing song %s", thread_id, song_id)
        y, sr = lb.load(song, sr=44100)
        # wav_data[song_id] = {'y': y, 'sr': sr}
        res = get_features_mean(y, sr, hop_length=512, n_fft=2048)
        output_queue.put({
            "id": song_id,
            "analysis": res
        })


def main():
    collisions = []

    tracks = {}

    with open("spotify_sources.json", "r") as file:
        sources = json.load(file)
        file.close()

    for genre in sources.keys():
        logger.info("Genre %s", genre)
        for artist_id in sources[genre]['artists']:
            logger.info("Artist %s", artist_id)
            artist_top10 = sp.artist_top_tracks(artist_id)['tracks']
            results = serialize_tracks_with_url(genre, artist_top10)
            append_with_collisions_detection(tracks, collisions, results)
        for playlist_id in sources[genre]['playlists']:
            logger.info("Playlist %s", playlist_id)
            playlist = sp.playlist_items(playlist_id)
            playlist_tracks = []
            playlist_tracks_raw = {}
            try:
                playlist_tracks_raw = playlist['tracks']['items']
            except KeyError:
                playlist_tracks_raw = playlist['items']
            for track in playlist_tracks_raw:
                playlist_tracks.append(track['track'])
            results = serialize_tracks_with_url(genre, playlist_tracks)
            append_with_collisions_detection(tracks, collisions, results)
        for album_id in sources[genre]['albums']:
            logger.info("Album %s", album_id)
            album_tracks = sp.album_tracks(album_id)['items']
            results = serialize_tracks_with_url(genre, album_tracks)
            append_with_collisions_detection(tracks, collisions, results)
        logger.info("%s done", genre)

    with open("../data/data_to_analysis.json", 'w') as file:
        file.write(json.dumps(tracks, separators=(',', ':')))
        file.close()

    logger.info("Starting files download")
    files = [f for f in glob.glob("..\\songs\\*.wav", recursive=True)]
    files_ids = []
    for file in files:
        files_ids.append(file.split('\\')[-1].split('.')[0])

    for track_id in tracks.keys():
        if tracks[track_id]['url'] is not None and track_id not in files_ids:
            sub = subprocess.Popen(["ffmpeg", "-i", tracks[track_id]['url'], "-acodec", "pcm_u8", "-ar", "44100", "../songs/" + track_id + ".wav"])
            sub.wait()
    logger.info("Files download completed")

    logger.info("Starting analyzer")

    files = [f for f in glob.glob("..\\songs\\*.wav", recursive=True)]

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    count = 0

    with open("../data/data.json", 'r') as file:
        tracks_with_analysis = json.load(file)
        file.close()

    for file in files:
        song_id = file.split('\\')[-1].split('.')[0]
        if song_id in tracks:
            if song_id not in tracks_with_analysis or tracks_with_analysis[song_id] is None:
                input_queue.put(file)
                count += 1

    processes = []

    for i in range(0, 9):
        processes.append(threading.Thread(target=analyser_thread, args=(i, input_queue, output_queue)))
        processes[i].start()

    with tqdm(total=count) as pbar:
        while True:
            if input_queue.empty():
                break
            i = 0
            while not output_queue.empty():
                results = output_queue.get()
                track = tracks[results['id']]
                track['audio_analysis'] = results['analysis']
                tracks_with_analysis[results["id"]] = track
                i += 1
            pbar.update(i)
            with open("../data/data.json", 'w') as file:
                file.write(json.dumps(tracks_with_analysis, separators=(',', ':')))
                file.close()
            time.sleep(10)
        pbar.close()

    with open("../data/data.json", 'w') as file:
        file.write(json.dumps(tracks_with_analysis, separators=(',', ':')))
        file.close()
    abs(1)


if __name__ == "__main__":
    main()
