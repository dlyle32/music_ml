import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model

class PlaylistFeatureNet():

    def __init__(self, args):
        self.learning_rate = args.learningrate
        self.dropout_rate = args.dropoutrate
        self.reg_factor = args.regfactor
        self.hiddenlayers = args.hiddenlayers
        self.modelpath = args.loadmodel

    def loadmodel(self):
        model = load_model(self.modelpath)
        return model

    def loss(self):
        return tf.keras.losses.BinaryCrossentropy(name="loss")

    def metrics(self):
        return tf.keras.metrics.BinaryAccuracy()

    def predict(self, model, dataset, sample):
        X = np.zeros((1,dataset.input_dim()))
        X[0] = sample
        preds = model.predict(X)[0]
        for i, p in enumerate(preds):
            id, name = dataset.lookup(i)
            print("%s, %s : %d" % (id, name, p))




    def create_model(self, feature_len, num_playlists):
        if self.modelpath:
            return self.loadmodel()

        inpt = keras.layers.Input(shape=(feature_len), name="input")
        out = inpt
        for layer_size in self.hiddenlayers:
            out = keras.layers.Dense(layer_size, activation="relu")(out)

        out = keras.layers.Dense(num_playlists, activation="sigmoid")(out)

        model = keras.Model(inputs=[inpt], outputs=[out])

        return model



