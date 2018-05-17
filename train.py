import tensorflow as tf
import numpy as np
from data_loader.data_generator import DataGenerator
from models.CNNClassifier import CNNClassifier
from trainers.Trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

def main():
    # try:
    #     args = get_args()
    #     config = process_config(args.config)
    #
    # except:
    #     print("missing or invalid arguments")
    #     exit(0)
    config_file = "configs/config.json"
    config = process_config(config_file)
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # embed_path = "data/glove.trimmed.{}.npz".format(config.embedding_size)
    # config.embed_path = embed_path
    # vocab_path = "data/vocab.dat"
    # config.vocab_path = vocab_path

    # create tensorflow session
    sess = tf.Session()

    # create data generator
    data = DataGenerator(config)
    sequence_length = data.sequence_length()
    vocab_size = data.get_vocab_size()
    # create an instance of the model
    model = CNNClassifier(config, sequence_length, vocab_size)
    #load model if exists
    model.load(sess)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, data, config, logger)

    trainer.train()

if __name__ == '__main__':
    main()
