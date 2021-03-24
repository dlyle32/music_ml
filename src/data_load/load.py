import json
import argparse
import spotipy
import os
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

def client_authorization():
    scope = "user-library-read"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
    return sp


def client_credentials():
    auth_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

def get_my_saved_tracks(sp):
    offset = 0
    batch_size = 50
    results = sp.current_user_saved_tracks(limit=batch_size, offset=offset)
    tracks = []
    while results:
        tracks.extend([item["track"] for item in results["items"]])
        results = sp.next(results)
    return tracks

def add_audio_features(sp, tracks):
    batch_size = 50
    total = len(tracks)
    for i in range(0,total,batch_size):
        trackids = [t["id"] for t in tracks[i:min(i+batch_size, total)]]
        results = sp.audio_features(trackids)
        for j,features in enumerate(results):
            tracks[i+j]["audio_features"] = features

def add_audio_analysis(sp, tracks):
    for i,track in enumerate(tracks):
        results = sp.audio_analysis(track["id"])
        tracks[i]["audio_analysis"] = results


def load(args):
    sp = client_authorization()
    print("LOADING TRACKS")
    tracks = get_my_saved_tracks(sp)

    if args.audio_features:
        print("LOADING FEATURES")
        add_audio_features(sp, tracks)

    if args.audio_analysis:
        print("LOADING ANALYSIS")
        add_audio_analysis(sp, tracks)

    return tracks


def main(args):
    tracks = load(args)
    with open("music_data.json", "w", encoding="utf-8") as fp:
        json.dump(tracks, fp, ensure_ascii=False, indent=4)

def load_tracks_features():
    args = argparse.Namespace()
    args.audio_features = True
    args.audio_analysis = False
    tracks = load(args)
    return tracks

def format_segments_vectors(seg):
    seg_vector = [seg["loudness_max"]]
    seg_vector.extend(seg["pitches"])
    seg_vector.extend(seg["timbre"])
    return np.array(seg_vector)

def load_track_artists():
    args = argparse.Namespace()
    args.audio_features = True
    args.audio_analysis = False
    tracks = load(args)
    artists = {track["id"]: track["artists"][0]["name"] for track in tracks }
    return artists

def load_audio_analysis(directory, window_len, step, datacap):
    input_vectors = []
    songs_loaded = 0
    for r, d, f in os.walk(directory):
        for fname in f:
            if "music_data" not in fname:
                continue
            if datacap > 0 and songs_loaded > datacap:
                break
            with open(os.path.join(r, fname), 'r') as fp:
                data = json.load(fp)
                tracks = [(track["id"],track["audio_analysis"]["segments"]) for track in data]
                songs_loaded += len(tracks)
                segment_vectors = format_sliding_window_input(tracks, window_len, step)
                input_vectors.extend(segment_vectors)

    # inputs = np.array(input_vectors)
    return input_vectors



def format_sliding_window_input(tracks, window_len, step):
    seg_vectors = []
    for track_id,seg_list in tracks:
        vects = [format_segments_vectors(seg) for seg in seg_list]
        for i in range(0,len(vects),step):
            if i + window_len > len(vects):
                continue
            seg_vectors.append((track_id,vects[i:i+window_len]))
    return seg_vectors

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_features", action="store_true")
    parser.add_argument("--audio_analysis", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # main(args)
    load_audio_analysis("data")
