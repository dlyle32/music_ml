import argparse
import time
import datetime
import logging
import os
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint

from data_load.load import add_audio_features
from spotipy_utils import search_track, client_authorization, get_user_playlists, client_credentials


logger = logging.getLogger('music_ml')

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

def test_inputs(model, modelbuilder, dataset, args):
    sp = client_authorization()
    testing = True
    while testing:
        track = search_track(sp)
        add_audio_features(sp, track)
        X = dataset.get_track_features(track[0])
        X = np.array(X)
        modelbuilder.predict(model, dataset, X)

        test_prompt = input("Search again (y/n)?")
        if test_prompt == "n":
            testing = False

def main(args):
    configure_logger(args)

    datadir = os.path.join(args.volumedir, args.datadir)

    # Dynamically load modelBuilder class
    moduleName, klassName = args.modelclass.split(".")
    mod = __import__('models.%s' % moduleName, fromlist=[klassName])
    modelklass = getattr(mod,klassName)
    modelBuilder = modelklass(args)

    # Dynamically load dataset class
    moduleName, klassName = args.dataclass.split(".")
    mod = __import__('datasets.%s' % moduleName, fromlist=[klassName])
    dataklass = getattr(mod,klassName)
    dataset = dataklass(args)

    model = modelBuilder.create_model(dataset.input_dim(), dataset.output_dim())

    model.compile(loss=modelBuilder.loss(),
                  optimizer=Adam(learning_rate=args.learningrate),
                  metrics=modelBuilder.metrics())
    logger.info(model.summary())
    logger_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: logger.info("Epoch %d: %s" % (epoch, str(logs))))
    checkpointpath = configure_checkpointing(args)
    checkpoint_callback = ModelCheckpoint(filepath=checkpointpath,
                                          save_weights_only=False)

    if args.trainfull:
        X,Y = dataset.full_set()
        history = model.fit(X,Y,
                            epochs=args.numepochs,
                            batch_size=args.minibatchsize,
                            validation_split=0.2,
                            shuffle=True,
                            callbacks=[logger_callback, checkpoint_callback])
    else:
        history = model.fit(dataset,
                        epochs=args.numepochs,
                        callbacks=[logger_callback, checkpoint_callback])

    test_inputs(model, modelBuilder, dataset, args)


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
    parser.add_argument("--modelclass", type=str)
    parser.add_argument("--dataclass", type=str)
    parser.add_argument("--trainfull", action="store_true")
    parser.add_argument("--hiddenlayers", type=int, nargs='+')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)