import tensorflow as tf
import os
from tensorflow import keras
from attn_utils import AttentionModelBuilder, positional_encoding
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.layers import einsum_dense
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
import numpy as np
import argparse
from data_load.load import load_audio_analysis, load_track_artists, add_audio_analysis, format_audio_input_for_tracks


class PlaylistAttentionModelBuilder(AttentionModelBuilder):
    def __init__(self, args):
        super().__init__(args)

    def create_model(self, num_artists):
        reg = keras.regularizers.l2(self.reg_factor)

        inpt = keras.layers.Input(shape=(self.seqlen,self.input_dim), name="input")
        # out = keras.layers.Embedding(input_dim, self.n_a, input_length=self.seqlen)(inpt)
        out = keras.layers.Dense(self.n_a)(inpt)
        pos_enc = positional_encoding(self.seqlen, self.n_a)
        pos_emb = keras.layers.Embedding(
            input_dim=self.seqlen,
            output_dim=self.n_a,
            weights=[pos_enc],
            name="position_embedding",
        )(tf.range(start=0, limit=self.seqlen, delta=1))
        # encoder_out = out + pos_emb
        encoder_out = tf.math.add(out, pos_emb)
        mask = self.subsequent_mask(self.seqlen)
        for i in range(self.transformer_layers):
            encoder_out = self.transformer_encoder(encoder_out, i, reg, mask)
        # decoder_out = self.transformer_decoder(encoder_out, target_emb, reg, mask)
        out = keras.layers.GlobalAveragePooling1D()(encoder_out)
        out = keras.layers.Dense(self.ffdim, activation="relu", kernel_regularizer=reg, name="penult_dense")(out)

        out = keras.layers.Dense(num_artists, activation="relu", name="final_dense")(out)

        model = keras.Model(inputs=inpt, outputs=out)
        return model