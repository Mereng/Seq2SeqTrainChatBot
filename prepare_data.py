import re
import nltk
import itertools
import numpy
import pickle


LINE_SEP = ' +++$+++ '
MAX_LEN_QUESTION = 25
MIN_LEN_QUESTION = 2
MAX_LEN_ANSWER = 25
MIN_LEN_ANSWER = 2
VOCABULARY_SIZE = 8000


def get_lines(lines):
    result = [[], []]

    last_char_id = None
    last_movie_id = None
    last_line = None
    last_line_num = None

    for i in range(len(lines) - 1, -1, -1):
        line_id, char_id, movie_id, _, line_text = lines[i].split(LINE_SEP)
        line_num = int(line_id[-1:])
        if movie_id != last_movie_id:
            last_char_id = char_id
            last_movie_id = movie_id
            last_line = line_text
            last_line_num = line_num
            continue

        if abs(line_num - last_line_num) > 1:
            last_char_id = char_id
            last_movie_id = movie_id
            last_line = line_text
            last_line_num = line_num
            continue

        if last_char_id == char_id:
            last_char_id = None
            last_movie_id = None
            last_line = None
            last_line_num = None
            continue

        result[0].append(last_line.lower())
        result[1].append(line_text.lower())

    return result[0], result[1]


def filter_line(line):
    regex = re.compile(r'[^a-z0-9\s]')
    return regex.sub('', line).strip()


def filter_data(questions, answers):
    filtered_qs, filtered_as = [], []
    data_len = len(questions)
    assert data_len == len(answers)

    for i in range(data_len):
        qs_len, as_len = len(questions[i].split(' ')), len(answers[i].split(' '))
        if qs_len >= MIN_LEN_QUESTION and qs_len <= MAX_LEN_QUESTION:
            if as_len >= MIN_LEN_ANSWER and as_len <= MAX_LEN_ANSWER:
                filtered_qs.append(questions[i])
                filtered_as.append(answers[i])
    return filtered_qs, filtered_as


def index(tokens):
    freq_dist = nltk.FreqDist(itertools.chain(*tokens))
    vocabulary = freq_dist.most_common(VOCABULARY_SIZE)
    index2word = ['_'] + ['unk'] + [i[0] for i in vocabulary]
    word2index = dict([(word, index) for index, word in enumerate(index2word)])
    return index2word, word2index, freq_dist


def filter_unk(qs_tokenized, as_tokenized, word2index):
    assert len(qs_tokenized) == len(as_tokenized)
    filtered_qs, filtered_as = [], []

    for qs_line, as_line in zip(qs_tokenized, as_tokenized):
        unk_count_qs = len([word for word in qs_line if word not in word2index])
        if unk_count_qs <= 2:
            filtered_qs.append(qs_line)
            filtered_as.append(as_line)
    return filtered_qs, filtered_as


def fill_sequence(sequence, word2index, length):
    idxs = []

    for word in sequence:
        if word in word2index:
            idxs.append(word2index[word])
        else:
            idxs.append(word2index['unk'])

    return idxs + [0] * (length - len(sequence))


def fill_zero(qs_tokenized, as_tokenized, word2index):
    data_length = len(qs_tokenized)
    idx_qs = numpy.zeros([data_length, MAX_LEN_QUESTION], dtype=numpy.int32)
    idx_as = numpy.zeros([data_length, MAX_LEN_ANSWER], dtype=numpy.int32)

    for i in range(data_length):
        qs_idxs = fill_sequence(qs_tokenized[i], word2index, MAX_LEN_QUESTION)
        as_idxs = fill_sequence(as_tokenized[i], word2index, MAX_LEN_ANSWER)

        idx_qs[i] = numpy.array(qs_idxs)
        idx_as[i] = numpy.array(as_idxs)

    return idx_qs, idx_as


if __name__ == '__main__':
    lines = None
    with open('data/movie_lines.txt', errors='ignore') as f:
        lines = f.readlines()
    print('Обработка файла...')
    answers, questions = get_lines(lines)
    print('Получено {} диалогов'.format(len(questions)))

    print('Очистка предложений от мусора...')
    questions = [filter_line(line) for line in questions]
    answers = [filter_line(line) for line in answers]
    print('Произведена очистка предложений от мусора')

    print('Очистка данных от длинных или коротких предложений...')
    questions, answers = filter_data(questions, answers)
    print('После очистки данных от длинных и коротких предложений осталось {} диалогов'
          .format(len(questions)))

    print('Разделение предложений на слова...')
    qs_tokenized = [[word.strip() for word in sentence.split(' ') if word] for sentence in questions]
    as_tokenized = [[word.strip() for word in sentence.split(' ') if word] for sentence in answers]
    print('Произведенно разделение предложений на слова...')

    print('Создание словаря из {} слов'.format(VOCABULARY_SIZE))
    idx2word, word2idx, freq_dist = index(qs_tokenized + as_tokenized)
    print('Создан словарь из {} слов'.format(VOCABULARY_SIZE))

    print('Очистка данных от предложений с большим количеством неизвестных слов')
    qs_tokenized, as_tokenized = filter_unk(qs_tokenized, as_tokenized, word2idx)
    print('После очистки данных от предложений с большим количеством неизвестных слов осталось {} диалогов'
          .format(len(qs_tokenized)))

    print('Преобразование данных в векторы')
    idx_qs, idx_as = fill_zero(qs_tokenized, as_tokenized, word2idx)
    print('Данные преобразованны векторы')

    print('Сохранение данных')
    metadata = {
        'word2idx': word2idx,
        'idx2word': idx2word
    }

    numpy.save('data/idx_qs.npy', idx_qs)
    numpy.save('data/idx_as.npy', idx_as)

    with open('data/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print('Выполненно сохрание данных')

