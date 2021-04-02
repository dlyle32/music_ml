import tensorflow as tf
import os
import datetime
import time
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.layers import einsum_dense
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
import numpy as np
import argparse
from data_load.load import load_audio_analysis, load_track_artists, add_audio_analysis, format_audio_input_for_tracks
from spotipy_utils import search_track, client_authorization, get_user_playlists, client_credentials
import spotipy

import logging

logger = logging.getLogger('music_ml')

class EinsumOp(keras.layers.Layer):
    def __init__(self, op, **kwargs):
        super(EinsumOp, self).__init__(**kwargs)
        self.op = op

    def call(self, inputs):
        a1 = inputs[0]
        a2 = inputs[1]
        attn_factor = special_math_ops.einsum(self.op, a1, a2)
        return attn_factor

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "op" : self.op
        })
        return config

def einsum_attn(i,q,k,v, dropout, dk, mask):
    # (1)
    # Query - key dot product:
    # (bd,qad, h, d), (bd,kad, h, d) -> (bd,h, qad, kad)
    # (2)
    # Combination:
    # (bd, h, qad, kad), (bd, vad, h, d) -> (bd, qad, h,d)
    dk = tf.cast(dk, tf.float32)
    dot = "aecd,abcd->aceb"
    com = "acbe,aecd->abcd"

    attn_factor = EinsumOp(dot, name="einsum_dot_%d" % i)([q,k])
    attn_factor = attn_factor / tf.math.sqrt(dk)
    if mask is not None:
        adder = (1.0 - math_ops.cast(mask, attn_factor.dtype)) * -1e9
        attn_factor += adder
    attn_factor = keras.layers.Softmax(name="attention_values_%d" % i)(attn_factor)
    attn_factor = keras.layers.Dropout(dropout, name="attention_dropout_%d" % i)(attn_factor)
    C = EinsumOp(com, name="einsum_com_%d" % i)([attn_factor,v])
    return C, attn_factor

def einsum_multihead_attention(i, q, k, v, h, n_a, reg, dropout,seqlen, mask=None):
    dim = n_a // h
    Wq = einsum_dense.EinsumDense("abc,cde->abde", kernel_regularizer=reg, output_shape=[None, h, dim],bias_axes="de", name="dense_q_%d" % i)
    Wk = einsum_dense.EinsumDense("abc,cde->abde", kernel_regularizer=reg, output_shape=[None, h, dim],bias_axes="de", name="dense_k_%d" % i)
    Wv = einsum_dense.EinsumDense("abc,cde->abde", kernel_regularizer=reg, output_shape=[None, h, dim],bias_axes="de", name="dense_v_%d" % i)
    Wo = einsum_dense.EinsumDense("abcd,cde->abe", kernel_regularizer=reg, output_shape=[None, n_a], bias_axes="e", name="dense_o_%d" % i)

    Q = Wq(q)
    K = Wk(k)
    V = Wv(v)

    C, attn_factor = einsum_attn(i, Q, K, V, dropout, dim, mask)

    return Wo(C)

def positional_encoding(seqlen, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(seqlen)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return keras.layers.Dropout(0.10)(pos_enc)

class AttentionModelBuilder:
    def __init__(self, args):
        self.learning_rate = args.learningrate
        self.dropout_rate = args.dropoutrate
        self.reg_factor = args.regfactor
        self.n_a = args.hiddensize
        self.step = args.step
        self.seqlen = args.seqlength
        self.ffdim = args.ffdim
        self.transformer_layers = args.transformer_layers
        self.attention_heads = args.attention_heads
        self.input_dim = 25

    def get_masked_datasets(self, dataset, mask_token_ix, vocab_size, pad_mask=None):
        #     dataset = dataset.to_numpy().reshape((dataset.shape[0],1))

        # 15% BERT masking
        if pad_mask is None:
            pad_mask = np.zeros(dataset.shape)
        inp_mask = np.logical_and(np.random.rand(*dataset.shape) < 0.15, pad_mask == False)

        labels = -1 * np.ones(dataset.shape)
        labels[inp_mask] = 0

        masked_dataset = np.copy(dataset)

        # 90% of mask indices get set to mask token
        inp_mask = np.logical_and(inp_mask, np.random.rand(*dataset.shape) < 0.9)
        masked_dataset[inp_mask] = mask_token_ix

        # 10% of mask indices get set to a random token (and 10% remain unchanged)
        inp_mask = np.logical_and(inp_mask, np.random.rand(*dataset.shape) < 1 / 9)
        masked_dataset[inp_mask] = np.random.randint(0, mask_token_ix, inp_mask.sum())

        # To be used to scale loss function to only count masked tokens
        loss_mask = np.ones(dataset.shape, dtype=int)
        loss_mask[labels == -1] = 0

        # The Y labels are just the original dataset
        y_labels = np.copy(dataset)

        return masked_dataset, y_labels, loss_mask

    def get_input_vectors_for_artist(self, segment_vectors, track_to_artist, reverse_artist, target_artist):
        # samples, seqlen, vect_dim = segment_vectors.shape
        pos_tracks = []
        neg_tracks = []
        for track_id,seg in segment_vectors:
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
        # samples, seqlen, vect_dim = segment_vectors.shape
        # return self.get_input_vectors_for_artist(segment_vectors, track_to_artist, reverse_artist, "The Beatles")
        artists_out = []
        X = np.zeros((len(segment_vectors),self.seqlen,self.input_dim))
        Y = np.zeros((len(segment_vectors),self.seqlen,self.input_dim))
        i = 0
        for track_id,seg in segment_vectors:
            artists_out.append(reverse_artist[track_to_artist[track_id]])
            Yseg = seg
            Xseg = np.append(np.zeros((1,self.input_dim)), Yseg[:-1], axis=0)
            X[i] = Xseg
            Y[i] = Yseg
            i += 1
        # X = np.array([seg[1] for seg in segment_vectors])
        # Y = np.array(artists_out)
        return X, Y

    def transformer_encoder(self, x, i, reg, mask=None):
        # Embedding, self-attention, dropout, residual layerNorm, ffn, residual layerNorm

        # attn_layer = keras.layers.MultiHeadAttention(self.attention_heads, self.n_a//self.attention_heads)
        # attn_out = attn_layer(x,x,x, attention_mask=mask)
        attn_out = einsum_multihead_attention(i, x, x, x, self.attention_heads, self.n_a, reg, self.dropout_rate,self.seqlen, mask=mask)

        x = keras.layers.add([attn_out, x])
        x = keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_{}/attn_norm".format(i))(x)

        # Feed-forward layer
        ffn = keras.Sequential(
            [
                keras.layers.Dense(self.ffdim, kernel_regularizer=reg, activation="relu"),
                keras.layers.Dense(self.n_a, kernel_regularizer=reg),
            ],
            name="encoder_{}/ffn".format(i),
        )

        ffn_out = ffn(x)
        ffn_out = keras.layers.Dropout(self.dropout_rate, name="encoder_{}/ffn_dropout".format(i))(ffn_out)

        x = keras.layers.add([ffn_out, x])
        x = keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_{}/ffn_norm".format(i))(x)
        return x

    def subsequent_mask(self, shape):
        "Mask out subsequent positions."
        subsequent_mask = np.triu(np.ones(shape), k=1).astype('uint8')
        return subsequent_mask == 0

    def transformer_decoder(self, encoder_out, targets, reg, mask=None):
        # Embedding, self attention, encoder attention, dropout, residual layerNorm, ffn, dropout, res norm, dense softmax
        # m = tf.shape(x)[0]

        attn_layer_1 = keras.layers.MultiHeadAttention(4, self.n_a // 4)
        attn_out_1 = attn_layer_1(targets, targets, targets, attention_mask=mask)
        # attn_out = multihead_attention(x, x, x, 4, self.n_a, m, reg, self.dropout_rate, mask=mask)
        #     attn_out = tf.reshape(out, (m,seqlen*n_a))

        attn_out_1 = keras.layers.LayerNormalization(epsilon=1e-6, name="decoder/attn_norm_1")(targets + attn_out_1)

        attn_layer_2 = keras.layers.MultiHeadAttention(4, self.n_a // 4)
        attn_out_2 = attn_layer_2(encoder_out, attn_out_1, attn_out_1, attention_mask=mask)
        # attn_out = multihead_attention(x, x, x, 4, self.n_a, m, reg, self.dropout_rate, mask=mask)
        #     attn_out = tf.reshape(out, (m,seqlen*n_a))

        attn_out_2 = keras.layers.LayerNormalization(epsilon=1e-6, name="decoder/attn_norm_2")(attn_out_1 + attn_out_2)

        # Feed-forward layer
        ffn = keras.Sequential(
            [
                keras.layers.Dense(self.n_a, kernel_regularizer=reg, activation="relu"),
                keras.layers.Dense(self.n_a, kernel_regularizer=reg),
            ],
            name="decoder/ffn",
        )

        ffn_out = ffn(attn_out_2)
        ffn_out = keras.layers.Dropout(self.dropout_rate, name="decoder/ffn_dropout")(ffn_out)

        x = keras.layers.LayerNormalization(epsilon=1e-6, name="decoder/ffn_norm")(attn_out_2 + ffn_out)
        return x

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

        out = keras.layers.Dense(self.input_dim, activation="relu", name="final_dense")(out)

        model = keras.Model(inputs=inpt, outputs=out)
        return model

def configure_logger(args):
    timestamp = int(time.time())
    logdir = os.path.join(args.volumedir, datetime.datetime.today().strftime('%Y%m%d'), args.logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    hdlr = logging.FileHandler(os.path.join(logdir, "training_output_%d.log" % timestamp))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

def configure_checkpointing(args):
    timestamp = int(time.time())
    checkpoint_dir = os.path.join(args.volumedir, datetime.datetime.today().strftime('%Y%m%d'), "checkpoints/")
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_name = "music_model.%d.{epoch:03d}.h5" % timestamp
    return os.path.join(checkpoint_dir, checkpoint_name)

def loadmodel(modelpath):
    model = load_model(modelpath, custom_objects={"EinsumOp": EinsumOp})
    return model

def test_inputs(model, args):
    sp = client_authorization()
    testing = True
    while testing:
        track = search_track(sp)
        add_audio_analysis(sp, track)
        input_vectors, track_to_artist = format_audio_input_for_tracks(track, args.seqlength, args.step)
        X = [seg for id, seg in input_vectors]
        X = np.array(X)
        preds = model.predict(X)
        avgs = np.mean(preds, axis=0)
        print(avgs)
        mins = np.min(preds, axis=0)
        print(mins)
        maxs = np.max(preds, axis=0)
        print(maxs)
        test_prompt = input("Search again (y/n)?")
        if test_prompt == "n":
            testing = False

def playlists():
    sp = client_authorization()
    user_info = sp.me()
    print(user_info)
    cats = sp.categories()
    get_user_playlists(sp,user_info["id"])
    return

def main(args):
    playlists()
    return
    configure_logger(args)

    if args.loadmodel:
        model = loadmodel(args.loadmodel)
        test_inputs(model, args)
        return

    datadir = os.path.join(args.volumedir, args.datadir)
    input_vectors, track_artist_map = load_audio_analysis(datadir, args.seqlength, args.step, args.datacap)
    # track_artist_map = load_track_artists()
    artists = list(set(track_artist_map.values()))
    reverse_artist = {a:i for i,a in enumerate(artists)}
    modelBuilder = AttentionModelBuilder(args)
    model = modelBuilder.create_model(len(artists))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(name="loss"),
                  optimizer=Adam(learning_rate=args.learningrate),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    logger.info(model.summary())
    X,Y = modelBuilder.get_input_vectors(input_vectors, track_artist_map, reverse_artist)
    logger_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: logger.info("Epoch %d: %s" % (epoch, str(logs))))
    checkpointpath = configure_checkpointing(args)
    checkpoint_callback = ModelCheckpoint(filepath=checkpointpath,
                                          save_weights_only=False,
                                          monitor='val_loss')
    history = model.fit(X, Y,
                        epochs=args.numepochs,
                        batch_size=args.minibatchsize,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=[logger_callback, checkpoint_callback])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loadmodel", type=str)
    parser.add_argument("--datadir", default="data/")
    parser.add_argument("--datacap", type=int, default=-1)
    parser.add_argument("--logdir", default="logs/")
    parser.add_argument("--volumedir", default="/training/")
    parser.add_argument("--checkpointdir", default="checkpoints/")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--hiddensize", type=int, default=128)
    parser.add_argument("--ffdim", type=int, default=256)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--minibatchsize", type=int, default=128)
    parser.add_argument("--numepochs", type=int, default=25)
    parser.add_argument("--seqlength", type=int, default=40)
    parser.add_argument("--learningrate", type=float, default=0.01)
    parser.add_argument("--dropoutrate", type=float, default=0.1)
    parser.add_argument("--regfactor", type=float, default=0.01)
    parser.add_argument("--valsplit", type=float, default=0.2)
    parser.add_argument("--embeddingsize", type=int, default=100)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--decaysteps", type=int, default=10000)
    parser.add_argument("--decayrate", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

