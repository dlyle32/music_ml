import tensorflow as tf
import numpy as np
import math
from data_load.load import load_user_playlists_with_features


class PlaylistFeatures(tf.keras.utils.Sequence):

    def __init__(self, args):
        self.batch_size = args.minibatchsize
        self.track_features, self.playlist_ids = load_user_playlists_with_features()
        self.reverse_pl = {p[0]:i for i,p in enumerate(self.playlist_ids)}
        self.feature_keys = {"danceability",
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

    def __len__(self):
        return math.ceil(len(self.track_features) / self.batch_size)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start+self.batch_size, len(self.analysis_vectors))
        return self.get_input_vectors(self.batch_size, self.track_features[start:end])

    def on_epoch_end(self):
        np.random.shuffle(self.track_features)

    def input_dim(self):
        return len(self.feature_keys)

    def output_dim(self):
        return len(self.playlist_ids)

    def get_input_vectors(self, batch_size, track_features):
        Y = np.zeros((batch_size, len(self.playlist_ids)))
        X = np.zeros((batch_size, len(self.feature_keys)))
        for i,track in enumerate(track_features):
            features = self.get_track_features(track)
            labels = self.get_playlist_label_vector(track)
            X[i] = np.array(features)
            Y[i] = labels.astype(np.int)
        return X, Y

    def get_playlist_label_vector(self, track):
        labels = np.zeros((len(self.playlist_ids)))
        ixs = np.array(range(len(self.playlist_ids)))
        pl_ids = track["playlists"]
        for id in pl_ids:
            labels = np.logical_or(labels, ixs == self.reverse_pl[id])
        return labels

    def get_track_features(self, track):
        features = [v for k, v in track["audio_features"].items() if k in self.feature_keys]
        return features

    def full_set(self):
        return self.get_input_vectors(len(self.track_features), self.track_features)

    def lookup(self, playlist_ix):
        return self.playlist_ids[playlist_ix]



