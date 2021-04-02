import tensorflow as tf
import numpy as np
import math
from data_load.load import load_audio_analysis, load_track_artists, add_audio_analysis, format_audio_input_for_tracks

class TrackArtistSequence(tf.keras.utils.Sequence):

    def __init__(self, batch_size, datadir, seqlen, step, datacap, target=None):
        self.batch_size = batch_size
        analysis_vectors, track_artist_map = load_audio_analysis(datadir, seqlen, step, datacap)
        artists = list(set(track_artist_map.values()))
        self.reverse_artist = {a: i for i, a in enumerate(artists)}
        self.target = target
        self.track_artist_map = track_artist_map
        self.analysis_vectors = analysis_vectors

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start+self.batch_size, len(self.analysis_vectors))
        if self.target is None:
            X,Y = self.get_input_vectors(self.analysis_vectors[start:end],
                                         self.track_artist_map,
                                         self.reverse_artist)
        else:
            X,Y = self.get_input_vectors_for_artist(self.analysis_vectors[start:end],
                                                    self.track_artist_map,
                                                    self.target)
        return X,Y

    def __len__(self):
        return math.ceil(len(self.analysis_vectors) / self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.analysis_vectors)

    def get_input_vectors_for_artist(self, segment_vectors, track_to_artist, target_artist):
        # samples, seqlen, vect_dim = segment_vectors.shape
        pos_tracks = []
        neg_tracks = []
        for track_id, seg in segment_vectors:
            artist = track_to_artist[track_id]
            if artist == target_artist:
                pos_tracks.append(seg)
            else:
                neg_tracks.append(seg)
        neg_tracks = neg_tracks[:len(pos_tracks)]
        pos_out = [1 for track in pos_tracks]
        neg_out = [0 for track in neg_tracks]
        pos_tracks.extend(neg_tracks)
        pos_out.extend(neg_out)
        X = np.array(pos_tracks)
        Y = np.array(pos_out)
        perm = np.random.permutation(X.shape[0])
        X = X[perm]
        Y = Y[perm]
        return X, Y

    def get_input_vectors(self, segment_vectors, track_to_artist, reverse_artist):
        artists_out = []
        for track_id, seg in segment_vectors:
            artists_out.append(reverse_artist[track_to_artist[track_id]])
        X = np.array([seg[1] for seg in segment_vectors])
        Y = np.array(artists_out)
        return X, Y




