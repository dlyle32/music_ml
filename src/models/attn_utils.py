import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.layers import einsum_dense
import numpy as np

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