import argparse
import tensorflow as tf

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab] # id, token
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)]) # token, id
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

if __name__ == '__main__':
    vocab_path = '../data/vocab.dat'
    vocab, rev_vocab = initialize_vocab(vocab_path)
    print(vocab['and'])
