from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow
import tensorlayer
import numpy
import pickle
import time
import sys
import os
import re


idx_qs = []
idx_as = []
idx2word = []
word2idx = {}
unk_id = 0
pad_id = 0
start_id = 0
end_id = 0
vocab_size = 0
embedding_size = 1024
batch_size = 32
learning_rate = 0.0001
n_epoch = 50


def load_data():
    global idx_qs
    idx_qs = numpy.load('data/idx_qs.npy')
    global idx_as
    idx_as = numpy.load('data/idx_as.npy')

    with open('data/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    global idx2word
    idx2word = metadata['idx2word']
    global word2idx
    word2idx = metadata['word2idx']

    global unk_id
    unk_id = word2idx['unk']
    global pad_id
    pad_id = word2idx['_']

    global vocab_size
    vocab_size = len(idx2word)

    global start_id
    start_id = vocab_size
    global end_id
    end_id = vocab_size + 1

    word2idx.update({'start_id': start_id})
    word2idx.update({'end_id': end_id})
    idx2word = idx2word + ['start_id', 'end_id']
    vocab_size = len(idx2word)


def get_model(encode_seq, decode_seq, is_train, reuse):
    with tensorflow.variable_scope('model', reuse=reuse):
        with tensorflow.variable_scope('embedding') as vs:
            net_encode = tensorlayer.layers.EmbeddingInputlayer(
                inputs=encode_seq,
                vocabulary_size=vocab_size,
                embedding_size=embedding_size,
                name='embedding'
            )
            vs.reuse_variables()
            tensorlayer.layers.set_name_reuse(True)
            net_decode = tensorlayer.layers.EmbeddingInputlayer(
                inputs=decode_seq,
                vocabulary_size=vocab_size,
                embedding_size=embedding_size,
                name='embedding'
            )

        net_rnn = tensorlayer.layers.Seq2Seq(
            net_encode,
            net_decode,
            cell_fn=tensorflow.contrib.rnn.BasicLSTMCell,
            n_hidden=embedding_size,
            initializer=tensorflow.random_uniform_initializer(-0.1, 0.1),
            encode_sequence_length=tensorlayer.layers.retrieve_seq_length_op2(encode_seq),
            decode_sequence_length=tensorlayer.layers.retrieve_seq_length_op2(decode_seq),
            initial_state_encode=None,
            dropout=(0.5 if is_train else None),
            n_layer=3,
            return_seq_2d=True,
            name='seq2seq'
        )

        net_out = tensorlayer.layers.DenseLayer(net_rnn, n_units=vocab_size, act=tensorflow.identity, name='output')

    return net_out, net_rnn


def get_checkpoint():
    regex = re.compile(r'checkpoint_[0-9]+\.npz')
    files = os.listdir('checkpoints')
    result = None
    max_checkpoint = 0
    for file in files:
        if regex.match(file):
            n_checkpoint = int(file.split('_')[1].split('.')[0])
            if n_checkpoint > max_checkpoint:
                max_checkpoint = n_checkpoint
                result = file
    return result, max_checkpoint


if __name__ == '__main__':
    load_data()

    train_x, test_x, train_y, test_y = train_test_split(idx_qs, idx_as, test_size=0.1)
    train_x = train_x.tolist()
    test_x = test_x.tolist()
    train_y = train_y.tolist()
    test_y = test_y.tolist()
    train_x = tensorlayer.prepro.remove_pad_sequences(train_x)
    test_x = tensorlayer.prepro.remove_pad_sequences(test_x)
    train_y = tensorlayer.prepro.remove_pad_sequences(train_y)
    test_y = tensorlayer.prepro.remove_pad_sequences(test_y)

    encode_seqs = tensorflow.placeholder(tensorflow.int64, [batch_size, None], 'encode_seqs')
    decode_seqs = tensorflow.placeholder(tensorflow.int64, [batch_size, None], 'decode_seqs')
    target_seqs = tensorflow.placeholder(tensorflow.int64, [batch_size, None], 'target_seqs')
    target_mask = tensorflow.placeholder(tensorflow.int64, [batch_size, None], 'target_mask')
    net_out, _ = get_model(encode_seqs, decode_seqs, True, False)

    loss = tensorlayer.cost.cross_entropy_seq_with_mask(net_out.outputs, target_seqs, target_mask, False, 'cost')
    net_out.print_params(False)

    train_op = tensorflow.train.AdamOptimizer(learning_rate).minimize(loss)

    session = tensorflow.Session(config=tensorflow.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tensorlayer.layers.initialize_global_variables(session)

    checkpoint, n_checkpoint = get_checkpoint()
    if checkpoint is not None:
        tensorlayer.files.load_and_assign_npz(session, 'checkpoints/' + checkpoint, net_out)

    n_step = int(len(train_x) / batch_size)
    for epoch in range(n_checkpoint, n_epoch):
        e_time = time.time()
        train_x, train_y = shuffle(train_x, train_y)
        total_error = 0
        n_iter = 0

        for X, Y in tensorlayer.iterate.minibatches(train_x, train_y, batch_size):
            step_time = time.time()
            X = tensorlayer.prepro.pad_sequences(X)
            _target_seqs = tensorlayer.prepro.sequences_add_end_id(Y, end_id)
            _target_seqs = tensorlayer.prepro.pad_sequences(_target_seqs)

            _decode_seqs = tensorlayer.prepro.sequences_add_start_id(Y, start_id)
            _decode_seqs = tensorlayer.prepro.pad_sequences(_decode_seqs)
            _target_mask = tensorlayer.prepro.sequences_get_mask(_target_seqs)

            _, err = session.run([train_op, loss], {
                encode_seqs: X,
                decode_seqs: _decode_seqs,
                target_seqs: _target_seqs,
                target_mask: _target_mask
            })

            sys.stdout.write('\rEpoch {}/{}, step {}/{}, loss {}, time {:.5f}'.format(epoch + 1, n_epoch, n_iter, n_step, err,
                                                                                  time.time() - step_time))
            sys.stdout.flush()
            total_error += err
            n_iter += 1
        print()
        print('Epoch {}/{}, average loss {}, time {:.5f}'.format(epoch + 1, n_epoch, total_error/n_iter,
                                                                    time.time() - e_time))
        tensorlayer.files.save_npz(net_out.all_params, 'checkpoints/checkpoint_{}.npz'.format(epoch + 1), session)
