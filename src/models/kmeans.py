import tensorflow as tf
import numpy as np
import json
import argparse
from sklearn.cluster import KMeans
from data_load.load import load_tracks_features

def write_clusters(k,preds, tracks, score):
    clusters = {key:[] for key in range(k)}
    clusters["score"] = score
    for cluster,track in zip(preds,tracks):
        song_artist = (track["name"], track["artists"][0]["name"])
        clusters[cluster].append(song_artist)
    with open("data/clusters_%d.json" % k, "w", encoding="utf-8") as fp:
        json.dump(clusters,fp,ensure_ascii=False, indent=4)


def kmeans(args):
    feature_keys = {"danceability",
                    "energy",
                    "key",
                    "loudness",
                    "mode",
                    "speechiness",
                    "acousticness",
                    "instrumentalness",
                    "liveness",
                    "valence",
                    "tempo",
                    "time_signature",
                    }
    tracks = load_tracks_features()
    features = [[v for k,v in track["audio_features"].items() if k in feature_keys ] for track in tracks]
    X = np.array(features)
    for k in args.clusters:
        print(k)
        model = KMeans(n_clusters=k).fit(X)
        preds = model.predict(X)
        score = model.score(X)
        write_clusters(k, preds, tracks, score)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", nargs='*', type=int, default=[6])
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    kmeans(args)
