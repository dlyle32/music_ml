import json
import argparse
import spotipy
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


def main(args):
    sp = client_authorization()
    tracks = get_my_saved_tracks(sp)

    if args.audio_features:
        add_audio_features(sp, tracks)

    if args.audio_analysis:
        add_audio_analysis(sp, tracks)

    with open("music_data.json", "w", encoding="utf-8") as fp:
        json.dump(tracks, fp, ensure_ascii=False, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_features", action="store_true")
    parser.add_argument("--audio_analysis", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
