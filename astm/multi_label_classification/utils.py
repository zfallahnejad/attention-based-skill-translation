import os
import pickle
import statistics
import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def print_time(t):
    """Function that converts time period in seconds into %h:%m:%s expression.
    Args:
        t (int): time period in seconds
    Returns:
        s (string): time period formatted
    """
    h = t // 3600
    m = (t % 3600) // 60
    s = (t % 3600) % 60
    return '%dh:%dm:%ds' % (h, m, s)


def prepare_keras_input(texts, num_words, max_sentence_len):
    tokenizer = Tokenizer(num_words=num_words, lower=True, filters='!"$%&()*,-./:;<=>?@[\\]^_`{|}~\t\n', )
    tokenizer.fit_on_texts(texts)
    print("Unique Tokens in Training Data: ", len(tokenizer.word_index))  # 945386

    sequences = tokenizer.texts_to_sequences(texts)

    sequences_length = [len(s) for s in sequences]
    length_count = {l: sequences_length.count(l) for l in set(sequences_length)}

    print("Average length=", statistics.mean(sequences_length))
    print("Median length=", statistics.median(sequences_length))
    print("Minimum length={}, sequence count={}".format(min(sequences_length), length_count[min(sequences_length)]))
    print("Maximum length={}, sequence count={}".format(max(sequences_length), length_count[max(sequences_length)]))
    max_count = max(length_count.values())
    length_max_count = list(length_count.keys())[list(length_count.values()).index(max_count)]
    print("Maximum count={}, on which length={}".format(max_count, length_max_count))
    '''
    num_words -> 100000 
    Unique Tokens in Training Data:  945386
    Average length= 239.48585558259254
    Median length= 159
    Minimum length=5, sequence count=1
    Maximum length=13685, sequence count=1
    Maximum count=1221, on which length=97
    '''
    data = pad_sequences(sequences, maxlen=max_sentence_len, padding='post')
    return data, tokenizer.word_index


def texts_to_sequences_generator(texts, word_index):
    for text in texts:
        seq = [i for i in text.split(' ') if i]
        text_vector = []
        for w in seq:
            i = word_index.get(w)
            if i is not None:
                text_vector.append(i)
        yield text_vector


def prepare_word2vec_input(tag, texts, word_index, num_words, max_sentence_len, word_vector_dim, train_part, post_type,
                           result_path):
    word_vectors = KeyedVectors.load(
        "../word_vectors/word2vec_model/{}-wv-d{}-win5-neg5-min10.kv".format(tag, word_vector_dim), mmap='r')
    print('We trained %s word vectors using gensim word2vec module.' % len(word_vectors.vectors))

    j = 1
    absent_words = 0
    new_word_index = {}
    filters = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@',
               '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\t', '\n']
    for word, i in word_index.items():
        if word in filters:
            continue
        if word in word_vectors:
            new_word_index[word] = j
            j += 1
            if j == num_words:
                break
        else:
            absent_words += 1
    # index_word = dict((c, w) for w, c in new_word_index.items())
    print("Unique Tokens in Training Data: ", len(new_word_index))  #

    word_index_text_file = os.path.join(result_path, "new_word_index_{}_tr{}.txt".format(post_type, train_part))
    word_index_path = os.path.join(result_path, "word_index_tr" + str(train_part) + ".pkl")
    with open(word_index_text_file, "w", encoding='utf8') as out_file:
        for word, i in new_word_index.items():
            out_file.write(str(i) + "\t" + word + "\n")
    with open(word_index_path, 'wb') as output:
        pickle.dump(new_word_index, output)

    vocab_size = len(new_word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, word_vector_dim))
    for word, i in new_word_index.items():
        embedding_matrix[i] = word_vectors[word]

    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    print("number of non zero row of embedding matrix: {}".format(nonzero_elements / vocab_size))

    sequences = list(texts_to_sequences_generator(texts, new_word_index))

    sequences_length = [len(s) for s in sequences]
    length_count = {l: sequences_length.count(l) for l in set(sequences_length)}

    print("Average length=", statistics.mean(sequences_length))
    print("Median length=", statistics.median(sequences_length))
    print("Minimum length={}, sequence count={}".format(min(sequences_length), length_count[min(sequences_length)]))
    print("Maximum length={}, sequence count={}".format(max(sequences_length), length_count[max(sequences_length)]))
    max_count = max(length_count.values())
    length_max_count = list(length_count.keys())[list(length_count.values()).index(max_count)]
    print("Maximum count={}, on which length={}".format(max_count, length_max_count))
    '''
    num_words -> 100000
    Unique Tokens in Training Data:  945386
    Average length= 239.48585558259254
    Median length= 159
    Minimum length=5, sequence count=1
    Maximum length=13685, sequence count=1
    Maximum count=1221, on which length=97
    '''
    data = pad_sequences(sequences, maxlen=max_sentence_len, padding='post')

    return data, new_word_index, embedding_matrix


def prepare_topic_vec_input(tag, texts, word_index, num_words, max_sentence_len, word_vector_dim, train_part, post_type,
                            result_path):
    word_vectors = {}
    with open("../word_vectors/word_topic_vectors/word_topic_vectors_{}_{}d.txt".format(tag, word_vector_dim),
              encoding="utf8") as infile:
        for line in infile:
            line = line.strip().split('\t')
            word = line[0]
            vector = [float(i) for i in line[1].split(' ')]
            word_vectors[word] = vector
    print('We trained %s word vectors using lda module.' % len(word_vectors))

    j = 1
    absent_words = 0
    new_word_index = {}
    filters = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@',
               '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\t', '\n']
    for word, i in word_index.items():
        if word in filters:
            continue
        if word in word_vectors:
            new_word_index[word] = j
            j += 1
            if j == num_words:
                break
        else:
            absent_words += 1
    # index_word = dict((c, w) for w, c in new_word_index.items())
    print("Unique Tokens in Training Data: ", len(new_word_index))  #

    word_index_text_file = os.path.join(result_path, "new_word_index_{}_tr{}.txt".format(post_type, train_part))
    word_index_path = os.path.join(result_path, "word_index_tr" + str(train_part) + ".pkl")
    with open(word_index_text_file, "w", encoding='utf8') as out_file:
        for word, i in new_word_index.items():
            out_file.write(str(i) + "\t" + word + "\n")
    with open(word_index_path, 'wb') as output:
        pickle.dump(new_word_index, output)

    vocab_size = len(new_word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, word_vector_dim))
    for word, i in new_word_index.items():
        embedding_matrix[i] = np.array(word_vectors[word])

    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    print("number of non zero row of embedding matrix: {}".format(nonzero_elements / vocab_size))

    sequences = list(texts_to_sequences_generator(texts, new_word_index))

    sequences_length = [len(s) for s in sequences]
    length_count = {l: sequences_length.count(l) for l in set(sequences_length)}

    print("Average length=", statistics.mean(sequences_length))
    print("Median length=", statistics.median(sequences_length))
    print("Minimum length={}, sequence count={}".format(min(sequences_length), length_count[min(sequences_length)]))
    print("Maximum length={}, sequence count={}".format(max(sequences_length), length_count[max(sequences_length)]))
    max_count = max(length_count.values())
    length_max_count = list(length_count.keys())[list(length_count.values()).index(max_count)]
    print("Maximum count={}, on which length={}".format(max_count, length_max_count))
    '''
    num_words -> 100000
    Unique Tokens in Training Data:  945386
    Average length= 239.48585558259254
    Median length= 159
    Minimum length=5, sequence count=1
    Maximum length=13685, sequence count=1
    Maximum count=1221, on which length=97
    '''
    data = pad_sequences(sequences, maxlen=max_sentence_len, padding='post')

    return data, new_word_index, embedding_matrix
