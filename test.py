import tensorflow
import tensorlayer
import pickle


with open('data/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

idx2word = metadata['idx2word']
word2idx = metadata['word2idx']

vocab_size = len(idx2word)
start_id = vocab_size
end_id = vocab_size + 1
word2idx.update({'start_id': start_id})
word2idx.update({'end_id': end_id})
idx2word = idx2word + ['start_id', 'end_id']
vocab_size = len(idx2word)
embedding_size = 1024


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
            cell_fn=tensorflow.nn.rnn_cell.BasicLSTMCell,
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


encode_seqs = tensorflow.placeholder(tensorflow.int64, [1, None], 'encode_seqs')
decode_seqs = tensorflow.placeholder(tensorflow.int64, [1, None], 'decode_seqs')
net, net_rnn = get_model(encode_seqs, decode_seqs, False, False)
y = tensorflow.nn.softmax(net.outputs)

session = tensorflow.Session(config=tensorflow.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tensorlayer.layers.initialize_global_variables(session)
tensorlayer.files.load_and_assign_npz(session, 'checkpoints/model.npz', net)


while True:
    msg = input('> ')
    idxs = [word2idx.get(word, word2idx['unk']) for word in msg.split(' ')]
    state = session.run(net_rnn.final_state_encode, {
        encode_seqs: [idxs]
    })
    o, state = session.run([y, net_rnn.final_state_decode], {
        net_rnn.initial_state_decode: state,
        decode_seqs: [[start_id]]
    })

    word_idx = tensorlayer.nlp.sample_top(o[0], top_k=3)
    word = idx2word[word_idx]
    sentence = [word]

    for _ in range(30):
        o, state = session.run([y , net_rnn.final_state_decode], {
            net_rnn.initial_state_decode: state,
            decode_seqs: [[word_idx]]
        })
        word_idx = tensorlayer.nlp.sample_top(o[0], top_k=2)
        word = idx2word[word_idx]
        if word_idx == end_id:
            break
        sentence = sentence + [word]
    print("A > ", ' '.join(sentence))
