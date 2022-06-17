import json
import subprocess

with open("./data/new_results.json", 'r') as file:
    tracks = json.load(file)
    file.close()


for track_id in tracks.keys():
    if tracks[track_id]['url'] is not None:
        sub = subprocess.Popen(["ffmpeg", "-i" ,tracks[track_id]['url'], "-acodec", "pcm_u8", "-ar", "44100", "./songs/" + track_id + ".wav"])
        sub.wait()