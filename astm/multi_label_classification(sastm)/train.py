import os
import time
import random
import pickle
import argparse
import statistics
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from model import SentenceClassifier, fbeta, macro_f1, evaluation, evaluation_top_label, perf_grid, \
    performance_table, performance_curves, top_words, top_words_2, top_words_3, top_words_4, top_words_5, \
    integrate_scores, integrate_scores_2, AttentionWithContext, AttentionWithContextV2, AttentionWithContextV3

random.seed(7)
np.random.seed(7)
tensorflow.compat.v1.set_random_seed(7)

CWD = os.path.dirname(__file__)
TOP_SO_TAGS = {
    "java": [
        "android", "swing", "eclipse", "spring", "hibernate", "arrays", "multithreading", "xml", "jsp", "string",
        "servlets", "maven", "java-ee", "mysql", "spring-mvc", "json", "regex", "tomcat", "jpa", "jdbc", "javascript",
        "arraylist", "web-services", "sql", "generics", "netbeans", "sockets", "user-interface", "jar", "html", "jsf",
        "database", "file", "google-app-engine", "gwt", "junit", "exception", "algorithm", "rest", "class",
        "performance",
        "applet", "image", "jtable", "c#", "jframe", "collections", "c++", "methods", "oop", "linux",
        "nullpointerexception", "jaxb", "parsing", "oracle", "concurrency", "php", "jpanel", "jboss", "object", "ant",
        "date", "selenium", "javafx", "jvm", "list", "struts2", "hashmap", "sorting", "awt", "http", "inheritance",
        "reflection", "hadoop", "windows", "loops", "unit-testing", "sqlite", "design-patterns", "serialization",
        "security", "intellij-idea", "file-io", "logging", "swt", "apache", "annotations", "jquery", "jersey", "scala",
        "libgdx", "osx", "encryption", "spring-security", "log4j", "python", "jni", "soap", "interface", "io"
    ],
    "php": [
        "mysql", "javascript", "html", "jquery", "arrays", "ajax", "wordpress", "sql", "codeigniter",
        "regex", "forms", "json", "apache", "database", ".htaccess", "symfony2", "laravel", "xml", "zend-framework",
        "curl", "session", "pdo", "css", "mysqli", "facebook", "cakephp", "email", "magento", "yii", "laravel-4",
        "oop", "string", "post", "image", "function", "variables", "api", "date", "mod-rewrite", "android",
        "security", "foreach", "multidimensional-array", "redirect", "url", "class", "validation", "java",
        "doctrine2", "linux", "file-upload", "joomla", "cookies", "loops", "facebook-graph-api", "file", "drupal",
        "soap", "datetime", "login", "preg-replace", "parsing", "csv", "if-statement", "zend-framework2", "html5",
        "upload", "paypal", "preg-match", "sorting", "phpmyadmin", "search", "get", "sql-server", "doctrine",
        "performance", "web-services", "table", "pdf", "utf-8", "simplexml", "object", "phpunit", "mongodb", "dom",
        "select", "http", "include", "authentication", "caching", "cron", "pagination", "twitter", "xampp",
        "python", "rest", "encryption", "wordpress-plugin", "gd", "smarty"
    ]
}


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


def get_post_tags(dataset, post_type, post_tags_path, post_score_path, log_file, remove_multi_label_posts=False):
    with open(post_tags_path, "rb") as input_file:
        post_tags = pickle.load(input_file)

    print("Number of input {}s: {}".format(post_type, len(post_tags)))
    log_file.write("Number of input {}s: {}\n".format(post_type, len(post_tags)))

    tag_count = [(t, [tag for pid, tags in post_tags.items() for tag in tags].count(t)) for t in TOP_SO_TAGS[dataset]]
    print("tag_count of input: {}\n".format(tag_count))
    log_file.write("tag_count of input: {}\n".format(tag_count))

    with open(post_score_path, "rb") as input_file:
        post_score = pickle.load(input_file)
        post_score = {pid: post_score[pid] for pid in post_score if pid in post_tags}

    # remove posts with negative score
    post_tags = {pid: tags for pid, tags in post_tags.items() if post_score[pid] >= 0}
    print("Number of input {}s with non-negative score: {}".format(post_type, len(post_tags)))
    log_file.write("Number of input {}s with non-negative score: {}\n".format(post_type, len(post_tags)))

    # remove posts with more than one labels
    if remove_multi_label_posts:
        post_tags = {pid: tags for pid, tags in dict(post_tags).items() if len(tags) == 1}
        print("Number of input {}s with just one tags: {}".format(post_type, len(post_tags)))
        log_file.write("Number of input {}s with just one tags: {}\n".format(post_type, len(post_tags)))

    tag_count = [(t, [tag for pid, tags in post_tags.items() for tag in tags].count(t)) for t in TOP_SO_TAGS[dataset]]
    print("tag_count of input: {}\n".format(tag_count))
    log_file.write("tag_count of input: {}\n".format(tag_count))

    return post_tags


def prepare_word_index(dataset, data_path, post_tags, word_index_dir, word_vector_dim, post_type, dataset_percent,
                       cls_tokens, num_words, result_path, log_file):
    all_docs = []
    with open(data_path, encoding="utf8") as infile:
        for line in infile:
            line = line.strip().split()
            post_id = line[0]
            if post_id in post_tags:
                post_text = line[1:]
                all_docs += post_text

    words_freq = Counter(all_docs)
    del all_docs
    print("Number of unique words in this sample dataset: {}".format(len(words_freq)))
    log_file.write("Number of unique words in this sample dataset: {}\n".format(len(words_freq)))

    word_vectors = set([])
    with open(os.path.join(word_index_dir, "word_topic_vectors_{}_{}d.txt".format(dataset, word_vector_dim)),
              encoding="utf8") as infile:
        for line in infile:
            line = line.strip().split('\t')
            word_vectors.add(line[0])
    print("We trained {} word vectors using lda module.".format(len(word_vectors)))
    log_file.write("We trained {} word vectors using lda module.".format(len(word_vectors)))

    j = 1
    absent_words = 0
    word_index = {}
    filters = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@',
               '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\t', '\n']
    if cls_tokens:
        for cls_token in cls_tokens:
            word_index[cls_token] = j
            j += 1
    for (word, freq) in words_freq.most_common():
        if word in filters:
            continue
        if word in word_vectors:
            word_index[word] = j
            j += 1
            if j == num_words:
                break
        else:
            absent_words += 1
    # index_word = dict((c, w) for w, c in new_word_index.items())
    print("Unique Tokens in Training Data: {}\n#absent_words: {}\n".format(len(word_index), absent_words))
    log_file.write("Unique Tokens in Training Data: {}\n#absent_words: {}\n".format(len(word_index), absent_words))

    word_index_text_file = os.path.join(result_path, "word_index_{}_{}s_tr{}_{}d_{}wr.txt".format(
        dataset, post_type, dataset_percent, word_vector_dim, num_words))
    with open(word_index_text_file, "w", encoding='utf8') as out_file:
        for word, i in word_index.items():
            out_file.write(str(i) + "\t" + word + "\n")

    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, word_vector_dim))
    with open(os.path.join(word_index_dir, "word_topic_vectors_{}_{}d.txt".format(dataset, word_vector_dim)),
              encoding="utf8") as infile:
        for line in infile:
            line = line.strip().split('\t')
            word = line[0]
            if word not in word_index:
                continue
            vector = [float(i) for i in line[1].split(' ')]
            embedding_matrix[word_index[word]] = np.array(vector)
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    print("number of non zero row of embedding matrix: {}\n".format(nonzero_elements / vocab_size))
    log_file.write("number of non zero row of embedding matrix: {}\n".format(nonzero_elements / vocab_size))
    print("Using trained word embedding with shape: {}\n".format(embedding_matrix.shape))
    log_file.write("Using trained word embedding with shape: {}\n".format(embedding_matrix.shape))
    return word_index, embedding_matrix


def get_train_test_posts(dataset, data_path, post_tags, word_index, dataset_percent, test_size, log_file):
    all_tags = []
    last_pid_index = 0
    max_post_tags = 0
    post_id_text = {}
    post_index_id = {}
    sequences_length = []
    with open(data_path, encoding="utf8") as infile:
        for line in infile:
            line = line.strip().split()
            post_id = line[0]
            if post_id not in post_tags:
                continue

            post_text = line[1:]
            post_word_indices = [word_index.get(i) for i in post_text]
            # post_word_indices = [_ for _ in post_word_indices if _]  # remove None values?
            post_word_indices = [_ for _ in post_word_indices if _ is not None]
            if len(post_word_indices) == 0:
                continue

            sequences_length.append(len(post_word_indices))
            max_post_tags = max(max_post_tags, len(post_tags[post_id]))
            all_tags += [tag for tag in post_tags[post_id]]
            post_index_id[last_pid_index] = post_id
            post_id_text[post_id] = post_word_indices
            last_pid_index += 1
    print("maximum number of tags: {}\n".format(max_post_tags))
    log_file.write("maximum number of tags: {}\n".format(max_post_tags))

    length_count = {l: sequences_length.count(l) for l in set(sequences_length)}
    print("Average length={}, Median length={}\nMinimum length={}, sequence count={}\n"
          "Maximum length={}, sequence count={}\n".format(
        statistics.mean(sequences_length),
        statistics.median(sequences_length),
        min(sequences_length), length_count[min(sequences_length)],
        max(sequences_length), length_count[max(sequences_length)]
    ))
    log_file.write("Average length={}, Median length={}\nMinimum length={}, sequence count={}\n"
                   "Maximum length={}, sequence count={}\n".format(
        statistics.mean(sequences_length),
        statistics.median(sequences_length),
        min(sequences_length), length_count[min(sequences_length)],
        max(sequences_length), length_count[max(sequences_length)]
    ))
    max_count = max(length_count.values())
    length_max_count = list(length_count.keys())[list(length_count.values()).index(max_count)]
    print("Maximum count={}, on which length={}\n".format(max_count, length_max_count))
    log_file.write("Maximum count={}, on which length={}\n".format(max_count, length_max_count))
    del sequences_length, length_count, max_count, length_max_count

    print("class_count_weight of samples: {}".format(
        [(t, all_tags.count(t), len(all_tags) / all_tags.count(t)) for t in TOP_SO_TAGS[dataset]]))
    log_file.write("class_count_weight of samples: {}\n".format(
        [(t, all_tags.count(t), len(all_tags) / all_tags.count(t)) for t in TOP_SO_TAGS[dataset]]))
    tag_weight = {t: (len(all_tags) / all_tags.count(t)) for t in TOP_SO_TAGS[dataset]}
    class_weight = {index: tag_weight[label] for index, label in enumerate(TOP_SO_TAGS[dataset])}
    print("class_weight: {}\n".format(class_weight))
    log_file.write("class_weight: {}\n".format(class_weight))

    tag_count = [(t, [tag for pid, tags in post_tags.items() for tag in tags].count(t)) for t in TOP_SO_TAGS[dataset]]
    print("tag_count of input: {}\n".format(tag_count))
    log_file.write("tag_count of input: {}\n".format(tag_count))
    sorted_tag_count = sorted(tag_count, key=lambda tup: tup[1], reverse=True)  # [('android', 98926), ('swing', 45154),
    tag_count_rank = {item[0]: idx for idx, item in enumerate(sorted_tag_count)}

    print("number of sample posts: {}\n".format(len(post_id_text)))
    log_file.write("number of sample posts: {}\n".format(len(post_id_text)))

    print("apply dataset_percent {} and the split data to train and test parts\n".format(dataset_percent))
    log_file.write("apply dataset_percent {} and the split data to train and test parts\n".format(dataset_percent))

    train_part, test_part = [], []
    for num_tags in range(1, max_post_tags + 1):
        print("number of tags:{}".format(num_tags))
        log_file.write("number of tags:{}\n".format(num_tags))

        sample_tag_posts = {t: [] for t in TOP_SO_TAGS[dataset]}
        for pid in post_id_text:
            if len(post_tags[pid]) != num_tags:
                continue
            if num_tags == 1:
                sample_tag_posts[post_tags[pid][0]].append(pid)
            else:
                pid_tag_count_ranks = [tag_count_rank[_] for _ in post_tags[pid]]
                representative_tag = sorted_tag_count[max(pid_tag_count_ranks)][0]
                sample_tag_posts[representative_tag].append(pid)
        print("num of posts for each tag:{}".format({t: len(sample_tag_posts[t]) for t in TOP_SO_TAGS[dataset]}))
        log_file.write("num of posts for each tag:{}\n".format(
            {t: len(sample_tag_posts[t]) for t in TOP_SO_TAGS[dataset]}
        ))

        for t in sample_tag_posts:
            posts = sample_tag_posts[t]
            random.shuffle(posts)
            posts = posts[:int(dataset_percent * len(posts))]
            num_train_samples = int((1 - test_size) * len(posts))
            train_part += posts[:num_train_samples]
            test_part += posts[num_train_samples:]

    print("#train posts: {}, #test posts: {}\n".format(len(train_part), len(test_part)))
    log_file.write("#train posts: {}, #test posts: {}\n".format(len(train_part), len(test_part)))

    train_tags_counter, test_tags_counter = Counter(), Counter()
    for pid in train_part:
        train_tags_counter.update(post_tags[pid])
    for pid in test_part:
        test_tags_counter.update(post_tags[pid])
    print("train tag counter: {}\ntest tag counter: {}\n".format(train_tags_counter, test_tags_counter))
    log_file.write("train tag counter: {}\ntest tag counter: {}\n".format(train_tags_counter, test_tags_counter))
    del train_tags_counter, test_tags_counter

    return train_part, test_part, post_id_text, post_index_id, class_weight


def get_inputs_for_binary_label(dataset, post_id_text, word_index, post_tags, train_posts, test_posts, max_sentence_len,
                                log_file):
    num_posts = len(train_posts) + len(test_posts)

    train_sequence_labels = []
    for pid in train_posts:
        # positive samples
        for tag in post_tags[pid]:
            train_sequence_labels.append(([word_index["CLS_" + tag]] + post_id_text[pid], 1))
        # negative samples
        negative_labels = [_ for _ in TOP_SO_TAGS[dataset] if _ not in post_tags[pid]]
        for tag in random.sample(negative_labels, len(post_tags[pid])):
            train_sequence_labels.append(([word_index["CLS_" + tag]] + post_id_text[pid], 0))

    random.shuffle(train_sequence_labels)
    train_sequences, train_labels = [], []
    while train_sequence_labels:
        seq, label = train_sequence_labels.pop(0)
        train_sequences.append(seq)
        train_labels.append(label)
    del train_sequence_labels

    label_count = train_labels.count(1)

    train_sequences = pad_sequences(train_sequences, maxlen=max_sentence_len, padding='post', truncating='post')
    train_labels = np.array(train_labels).reshape(len(train_labels), 1)
    print("x_train shape: {}, y_train shape: {}\n".format(train_sequences.shape, train_labels.shape))
    log_file.write("x_train shape: {}, y_train shape: {}\n".format(train_sequences.shape, train_labels.shape))
    print('first train sample: {}, label: {}\n'.format(train_sequences[0], train_labels[0]))
    log_file.write('first train sample: {}, label: {}\n'.format(train_sequences[0], train_labels[0]))

    test_sequences, test_labels = [], []
    for pid in test_posts:
        test_sequences.append(post_id_text[pid])
        test_labels.append(post_tags[pid])
        label_count += len(post_tags[pid])
    del post_id_text, test_posts

    print("#test set samples: {}\n".format(len(test_sequences)))
    log_file.write("#test set samples: {}\n".format(len(test_sequences)))
    print('first test sample: {}, label{}\n'.format(test_sequences[0], test_labels[0]))
    log_file.write('first test sample: {}, label{}\n'.format(test_sequences[0], test_labels[0]))

    label_count = {0: (num_posts * len(TOP_SO_TAGS[dataset])) - label_count, 1: label_count}
    print("label_count: {}".format(label_count))
    log_file.write("label_count: {}".format(label_count))
    class_weight = {
        0: (num_posts * len(TOP_SO_TAGS[dataset])) / label_count[0],
        1: (num_posts * len(TOP_SO_TAGS[dataset])) / label_count[1]
    }
    print("class_weight: {}\n".format(class_weight))
    log_file.write("class_weight: {}\n".format(class_weight))

    return train_sequences, test_sequences, train_labels, test_labels, class_weight


def get_inputs_for_multi_label(dataset, post_id_text, word_index, post_tags, train_posts, test_posts,
                               add_single_cls_token, max_sentence_len, log_file):
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit([TOP_SO_TAGS[dataset]])
    labels_names = multilabel_binarizer.classes_
    print("{} classes: {}".format(len(labels_names), labels_names))
    log_file.write("{} classes: {}\n".format(len(labels_names), labels_names))

    train_sequences, train_labels = [], []
    for pid in train_posts:
        if add_single_cls_token:
            train_sequences.append([word_index["CLS"]] + post_id_text[pid])
        else:
            train_sequences.append(post_id_text[pid])
        train_labels.append(post_tags[pid])
    del train_posts

    test_sequences, test_labels = [], []
    for pid in test_posts:
        if add_single_cls_token:
            test_sequences.append([word_index["CLS"]] + post_id_text[pid])
        else:
            test_sequences.append(post_id_text[pid])
        test_labels.append(post_tags[pid])
    del post_id_text, test_posts

    train_sequences = pad_sequences(train_sequences, maxlen=max_sentence_len, padding='post', truncating='post')
    test_sequences = pad_sequences(test_sequences, maxlen=max_sentence_len, padding='post', truncating='post')
    train_labels = multilabel_binarizer.transform(train_labels)
    test_labels = multilabel_binarizer.transform(test_labels)
    print("x_train shape: {}, y_train shape: {}\nx_test shape: {}, y_test shape: {}\n".format(
        train_sequences.shape, train_labels.shape, test_sequences.shape, test_labels.shape))
    log_file.write("x_train shape: {}, y_train shape: {}\nx_test shape: {}, y_test shape: {}\n".format(
        train_sequences.shape, train_labels.shape, test_sequences.shape, test_labels.shape))

    print('first train sample: {}, label: {}\n'.format(train_sequences[0], train_labels[0]))
    log_file.write('first train sample: {}, label: {}\n'.format(train_sequences[0], train_labels[0]))
    print('first test sample: {}, label: {}\n'.format(test_sequences[0], test_labels[0]))
    log_file.write('first test sample: {}, label: {}\n'.format(test_sequences[0], test_labels[0]))

    return train_sequences, test_sequences, train_labels, test_labels, labels_names


def get_inputs_for_multi_label_2(dataset, post_id_text, word_index, post_tags, train_posts, test_posts,
                                 add_single_cls_token, max_sentence_len, log_file):
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit([TOP_SO_TAGS[dataset]])
    labels_names = multilabel_binarizer.classes_
    print("{} classes: {}".format(len(labels_names), labels_names))
    log_file.write("{} classes: {}\n".format(len(labels_names), labels_names))

    train_tag_pid = {}
    for pid in train_posts:
        representative_tag = min([TOP_SO_TAGS[dataset].index(tag) for tag in post_tags[pid]])
        if representative_tag not in train_tag_pid:
            train_tag_pid[representative_tag] = []
        train_tag_pid[representative_tag].append(pid)
    validation_posts = []
    for tag in train_tag_pid:
        validation_posts += random.sample(train_tag_pid[tag], int(0.1 * len(train_tag_pid[tag])))
    print("number of posts for validation: {}\n".format(len(validation_posts)))
    log_file.write("number of posts for validation: {}\n".format(len(validation_posts)))
    random.shuffle(validation_posts)
    del train_tag_pid

    train_posts = list(set(train_posts) - set(validation_posts))
    print("number of posts remain for training: {}\n".format(len(train_posts)))
    log_file.write("number of posts remain for training: {}\n".format(len(train_posts)))

    train_sequences, train_labels, val_sequences, val_labels = [], [], [], []
    for pid in validation_posts:
        if add_single_cls_token:
            val_sequences.append([word_index["CLS"]] + post_id_text[pid])
        else:
            val_sequences.append(post_id_text[pid])
        val_labels.append(post_tags[pid])
    for pid in train_posts:
        if add_single_cls_token:
            train_sequences.append([word_index["CLS"]] + post_id_text[pid])
        else:
            train_sequences.append(post_id_text[pid])
        train_labels.append(post_tags[pid])
    del train_posts, validation_posts

    test_sequences, test_labels = [], []
    for pid in test_posts:
        if add_single_cls_token:
            test_sequences.append([word_index["CLS"]] + post_id_text[pid])
        else:
            test_sequences.append(post_id_text[pid])
        test_labels.append(post_tags[pid])
    del post_id_text, test_posts

    train_sequences = pad_sequences(train_sequences, maxlen=max_sentence_len, padding='post', truncating='post')
    val_sequences = pad_sequences(val_sequences, maxlen=max_sentence_len, padding='post', truncating='post')
    test_sequences = pad_sequences(test_sequences, maxlen=max_sentence_len, padding='post', truncating='post')
    train_labels = multilabel_binarizer.transform(train_labels)
    val_labels = multilabel_binarizer.transform(val_labels)
    test_labels = multilabel_binarizer.transform(test_labels)
    print("x_train shape: {}, y_train shape: {}\nx_val shape: {}, y_val shape: {}\nx_test shape: {}, y_test shape: {}"
          "\n".format(train_sequences.shape, train_labels.shape, val_sequences.shape, val_labels.shape,
                      test_sequences.shape, test_labels.shape)
          )
    log_file.write("x_train shape: {}, y_train shape: {}\nx_val shape: {}, y_val shape: {}\nx_test shape: {}, "
                   "y_test shape: {}\n".format(train_sequences.shape, train_labels.shape, val_sequences.shape,
                                               val_labels.shape, test_sequences.shape, test_labels.shape)
                   )

    print('first train sample: {}, label: {}\n'.format(train_sequences[0], train_labels[0]))
    log_file.write('first train sample: {}, label: {}\n'.format(train_sequences[0], train_labels[0]))
    print('first val sample: {}, label: {}\n'.format(val_sequences[0], val_labels[0]))
    log_file.write('first val sample: {}, label: {}\n'.format(val_sequences[0], val_labels[0]))
    print('first test sample: {}, label: {}\n'.format(test_sequences[0], test_labels[0]))
    log_file.write('first test sample: {}, label: {}\n'.format(test_sequences[0], test_labels[0]))

    return train_sequences, val_sequences, test_sequences, train_labels, val_labels, test_labels, labels_names


def get_inputs_for_multi_class(dataset, post_id_text, word_index, post_tags, train_posts, test_posts,
                               add_tag_specific_cls_token, max_sentence_len, log_file):
    # label_binarizer = LabelBinarizer()
    # label_binarizer.fit(TOP_SO_TAGS[dataset])
    # labels_names = label_binarizer.classes_
    labels_names = TOP_SO_TAGS[dataset]
    print("{} classes: {}".format(len(labels_names), labels_names))
    log_file.write("{} classes: {}\n".format(len(labels_names), labels_names))

    train_tag_pid = {}
    for pid in train_posts:
        single_tag = post_tags[pid][0]
        if single_tag not in train_tag_pid:
            train_tag_pid[single_tag] = []
        train_tag_pid[single_tag].append(pid)
    validation_posts = []
    for tag in train_tag_pid:
        validation_posts += random.sample(train_tag_pid[tag], int(0.1 * len(train_tag_pid[tag])))
    print("number of posts for validation: {}\n".format(len(validation_posts)))
    log_file.write("number of posts for validation: {}\n".format(len(validation_posts)))
    random.shuffle(validation_posts)
    del train_tag_pid

    train_posts = list(set(train_posts) - set(validation_posts))
    print("number of posts remain for training: {}\n".format(len(train_posts)))
    log_file.write("number of posts remain for training: {}\n".format(len(train_posts)))

    train_sequences, train_labels, val_sequences, val_labels = [], [], [], []
    for pid in validation_posts:
        if add_tag_specific_cls_token:
            val_sequences.append([word_index["CLS_" + post_tags[pid][0]]] + post_id_text[pid])
        else:
            val_sequences.append(post_id_text[pid])
        # val_labels.append([post_tags[pid][0]])
        val_labels.append(labels_names.index(post_tags[pid][0]))
    for pid in train_posts:
        if add_tag_specific_cls_token:
            train_sequences.append([word_index["CLS_" + post_tags[pid][0]]] + post_id_text[pid])
        else:
            train_sequences.append(post_id_text[pid])
        # train_labels.append([post_tags[pid][0]])
        train_labels.append(labels_names.index(post_tags[pid][0]))
    del train_posts, validation_posts

    test_sequences, test_labels = [], []
    for pid in test_posts:
        if add_tag_specific_cls_token:
            test_sequences.append([word_index["CLS_" + post_tags[pid][0]]] + post_id_text[pid])
        else:
            test_sequences.append(post_id_text[pid])
        # test_labels.append([post_tags[pid][0]])
        test_labels.append(labels_names.index(post_tags[pid][0]))
    del post_id_text, test_posts

    train_sequences = pad_sequences(train_sequences, maxlen=max_sentence_len, padding='post', truncating='post')
    val_sequences = pad_sequences(val_sequences, maxlen=max_sentence_len, padding='post', truncating='post')
    test_sequences = pad_sequences(test_sequences, maxlen=max_sentence_len, padding='post', truncating='post')
    # train_labels = label_binarizer.transform(train_labels)
    # val_labels = label_binarizer.transform(val_labels)
    # test_labels = label_binarizer.transform(test_labels)
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)
    print("x_train shape: {}, y_train shape: {}\nx_val shape: {}, y_val shape: {}\nx_test shape: {}, y_test shape: {}"
          "\n".format(train_sequences.shape, train_labels.shape, val_sequences.shape, val_labels.shape,
                      test_sequences.shape, test_labels.shape)
          )
    log_file.write("x_train shape: {}, y_train shape: {}\nx_val shape: {}, y_val shape: {}\nx_test shape: {}, "
                   "y_test shape: {}\n".format(train_sequences.shape, train_labels.shape, val_sequences.shape,
                                               val_labels.shape, test_sequences.shape, test_labels.shape)
                   )

    print('first train sample: {}, label: {}\n'.format(train_sequences[0], train_labels[0]))
    log_file.write('first train sample: {}, label: {}\n'.format(train_sequences[0], train_labels[0]))
    print('first val sample: {}, label: {}\n'.format(val_sequences[0], val_labels[0]))
    log_file.write('first val sample: {}, label: {}\n'.format(val_sequences[0], val_labels[0]))
    print('first test sample: {}, label: {}\n'.format(test_sequences[0], test_labels[0]))
    log_file.write('first test sample: {}, label: {}\n'.format(test_sequences[0], test_labels[0]))

    return train_sequences, val_sequences, test_sequences, train_labels, val_labels, test_labels, labels_names


def plot_1(history, result_path):
    style.use("bmh")
    plt.figure(figsize=(15, 15))

    # summarize history for macro f1
    # plt.subplot(411)
    plt.subplot(3, 2, 1)
    plt.plot(history.history['macro_f1'])
    plt.plot(history.history['val_macro_f1'])
    plt.title('model macro f1')
    plt.ylabel('macro f1')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for categorical accuracy
    # plt.subplot(412)
    plt.subplot(3, 2, 2)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model categorical accuracy')
    plt.ylabel('categorical accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    # plt.subplot(413)
    plt.subplot(3, 2, 3)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    # plt.subplot(414)
    plt.subplot(3, 2, 4)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for top_k_categorical_accuracy
    plt.subplot(3, 2, 5)
    plt.plot(history.history['top_k_categorical_accuracy'])
    plt.plot(history.history['val_top_k_categorical_accuracy'])
    plt.title('model top k categorical accuracy')
    plt.ylabel('top_k_categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for fbeta
    plt.subplot(3, 2, 6)
    plt.plot(history.history['fbeta'])
    plt.plot(history.history['val_fbeta'])
    plt.title('model fbeta')
    plt.ylabel('fbeta')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(result_path, "plot_1.png"))
    plt.close()


def learning_curves(history, result_path):
    """Plot the learning curves of loss and macro f1 score
    for the training and validation datasets.

    Args:
        history: history callback of fitting a model
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    macro_f1 = history.history['macro_f1']
    val_macro_f1 = history.history['val_macro_f1']
    epochs = len(loss)

    style.use("bmh")
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), loss, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), macro_f1, label='Training Macro F1-score')
    plt.plot(range(1, epochs + 1), val_macro_f1, label='Validation Macro F1-score')
    plt.legend(loc='lower right')
    plt.ylabel('Macro F1-score')
    plt.title('Training and Validation Macro F1-score')
    plt.xlabel('epoch')

    plt.savefig(os.path.join(result_path, "plot_2.png"))

    return loss, val_loss, macro_f1, val_macro_f1


def train(dataset, data_path, post_tags_path, post_score_path, result_path, dataset_percent, test_size, post_type,
          num_words, word_vector_dim, train_embedding, max_sentence_len, epochs, batch_size, model_name,
          continue_training, initial_epoch, num_top_trans, lstm_dim, dropout_value, word_index_dir,
          use_class_weight):
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    log_file = open(os.path.join(result_path, "log.txt"), "w")

    if model_name in ["ASTM1", "ASTM2"]:
        cls_tokens = []
        remove_multi_label_posts = False
    elif model_name in ["SASTM"]:
        cls_tokens = ["CLS"]
        remove_multi_label_posts = False
    else:
        cls_tokens = []
        remove_multi_label_posts = False

    post_tags = get_post_tags(dataset, post_type, post_tags_path, post_score_path, log_file, remove_multi_label_posts)
    word_index, embedding_matrix = prepare_word_index(
        dataset, data_path, post_tags, word_index_dir, word_vector_dim, post_type, dataset_percent, cls_tokens,
        num_words, result_path, log_file
    )
    index_word = {index: word for word, index in word_index.items()}
    train_posts, test_posts, post_id_text, post_index_id, class_weight = get_train_test_posts(
        dataset, data_path, post_tags, word_index, dataset_percent, test_size, log_file
    )

    if model_name in ["ASTM1", "ASTM2"]:
        x_train, x_test, y_train, y_test, labels_names = get_inputs_for_multi_label(
            dataset, post_id_text, word_index, post_tags, train_posts, test_posts, False, max_sentence_len, log_file
        )
        loss_function = 'binary_crossentropy'
    elif model_name in ["SASTM"]:
        x_train, x_test, y_train, y_test, labels_names = get_inputs_for_multi_label(
            dataset, post_id_text, word_index, post_tags, train_posts, test_posts, True, max_sentence_len, log_file
        )
        loss_function = 'binary_crossentropy'
    else:
        x_train, x_test, y_train, y_test, labels_names = get_inputs_for_multi_label(
            dataset, post_id_text, word_index, post_tags, train_posts, test_posts, False, max_sentence_len, log_file
        )
        loss_function = 'binary_crossentropy'

    classifier = SentenceClassifier(
        max_sentence_len=max_sentence_len, num_words=num_words, word_vector_dim=word_vector_dim,
        label_count=len(labels_names)
    )
    print("Train word embedding: {}".format(train_embedding))
    log_file.write("Train word embedding: {}\n".format(train_embedding))

    if continue_training:
        model = tensorflow.keras.models.load_model(
            os.path.join(result_path, 'model.h5'),
            custom_objects={
                "macro_f1": macro_f1, "fbeta": fbeta, "AttentionWithContext": AttentionWithContext,
                "AttentionWithContextV2": AttentionWithContextV2, "AttentionWithContextV3": AttentionWithContextV3,
                'GlorotUniform': glorot_uniform()
            })
    else:
        print("loss function: {}".format(loss_function))
        log_file.write("loss function: {}\n".format(loss_function))

        if model_name == "ASTM1":
            model = classifier.astm_1(train_embedding, embedding_matrix, lstm_dim, dropout_value)
            model.compile(loss=loss_function, optimizer='adam',
                          metrics=[macro_f1, 'acc', 'categorical_accuracy', 'top_k_categorical_accuracy', fbeta])
        elif model_name == "ASTM2":
            model = classifier.astm_2(train_embedding, embedding_matrix, lstm_dim, dropout_value)
            model.compile(loss=loss_function, optimizer='adam',
                          metrics=[macro_f1, 'acc', 'categorical_accuracy', 'top_k_categorical_accuracy', fbeta])
        elif model_name == "SASTM":
            model = classifier.sastm(train_embedding, embedding_matrix, lstm_dim, dropout_value)
            model.compile(loss=loss_function, optimizer='adam',
                          metrics=[macro_f1, 'acc', 'categorical_accuracy', 'top_k_categorical_accuracy', fbeta])
        else:
            exit(1)

    print("Model Summary:\n{}".format(model.summary()))
    model.summary(print_fn=lambda x: log_file.write(x + '\n'))
    # log_file.write("Model Summary:\n{}".format(model.summary()))
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    print("\nTraining model...")
    log_file.write("\nTraining model...\n")

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint(filepath=os.path.join(result_path, 'model.h5'), save_best_only=True, verbose=1),
        TensorBoard(log_dir=os.path.join(result_path, "logs"))
    ]
    start = time.time()
    if use_class_weight:
        print("Using class_weight!\n")
        log_file.write("Using class_weight!\n")
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                            callbacks=callbacks, class_weight=class_weight, initial_epoch=initial_epoch)
    else:
        print("Without class_weight!\n")
        log_file.write("Without class_weight!\n")
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                            callbacks=callbacks, initial_epoch=initial_epoch)
    print('\nTraining took {}'.format(print_time(int(time.time() - start))))
    log_file.write('\nTraining took {}\n'.format(print_time(int(time.time() - start))))

    try:
        if history.history:
            plot_1(history, result_path)
            losses, val_losses, macro_f1s, val_macro_f1s = learning_curves(history, result_path)
            print("Validation Macro Loss: %.2f" % val_losses[-1])
            print("Validation Macro F1-score: %.2f" % val_macro_f1s[-1])
            log_file.write("Validation Macro Loss: %.2f\n" % val_losses[-1])
            log_file.write("Validation Macro F1-score: %.2f\n" % val_macro_f1s[-1])
    except:
        pass

    if model_name == "ASTM1":
        print("Extract top words of test data:")
        log_file.write("Extract top words of test data:\n")
        test_word_scores = top_words(
            model, x_test, y_test, "Test", labels_names, index_word, 64, log_file, num_top_trans
        )

        print("Extract top words of train data:")
        log_file.write("Extract top words of train data:\n")
        train_word_scores = top_words(
            model, x_train, y_train, "Train", labels_names, index_word, 64, log_file, num_top_trans
        )
        integrate_scores(train_word_scores, test_word_scores, labels_names, log_file)
    elif model_name == "ASTM2":
        print("Extract top words of test data:")
        log_file.write("Extract top words of test data:\n")
        top_words_2(model, x_test, y_test, "Test", labels_names, index_word, 64, log_file, result_path, num_top_trans)

        print("Extract top words of train data:")
        log_file.write("Extract top words of train data:\n")
        top_words_2(
            model, x_train, y_train, "Train", labels_names, index_word, 64, log_file, result_path, num_top_trans
        )
        integrate_scores_2(labels_names, log_file, result_path)
    elif model_name == "SASTM":
        print("Extract top words of test data:")
        log_file.write("Extract top words of test data:\n")
        top_words_3(
            model, x_test, y_test, "Test", labels_names, index_word, 32, log_file, result_path,
            'attention_with_context_v3', num_top_trans
        )

        print("Extract top words of train data:")
        log_file.write("Extract top words of train data:\n")
        top_words_3(
            model, x_train, y_train, "Train", labels_names, index_word, 32, log_file, result_path,
            'attention_with_context_v3', num_top_trans
        )
        integrate_scores_2(labels_names, log_file, result_path)

    try:
        if model_name in ["ASTM1", "ASTM2", "SASTM"]:
            metrics = model.evaluate(x_test, y_test)
            for i in range(len(metrics)):
                print("{}: {}".format(model.metrics_names[i], metrics[i]))
                log_file.write("{}: {}\n".format(model.metrics_names[i], metrics[i]))

            print("evaluate without threshold for prediction:\n")
            log_file.write("evaluate without threshold for prediction:\n")
            y_test_preds = model.predict(x_test, verbose=0)
            evaluation(y_test, y_test_preds, log_file)

            print("evaluate with 0.5 as threshold for prediction:\n")
            log_file.write("evaluate with 0.5 threshold for prediction:\n")
            y_test_preds_t05 = (model.predict(x_test, verbose=0) >= 0.5).astype(int)
            evaluation(y_test, y_test_preds_t05, log_file)

            # evaluation_top_label(model, x_test, y_test, log_file)
            max_perf, grid = performance_table(y_test_preds, y_test, labels_names, result_path, log_file)
            performance_curves(max_perf, grid, y_test_preds, labels_names, result_path)
    except:
        pass

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="", default='java')
    parser.add_argument('--data_path', help='', required=True)
    parser.add_argument('--post_tags_path', help='', required=True)
    parser.add_argument('--post_score_path', help='', required=True)
    parser.add_argument('--result_path', help='', required=True)
    parser.add_argument('--dataset_percent', type=float, help="", required=True)
    parser.add_argument('--test_size', type=float, help='', default=0.2)
    parser.add_argument('--post_type', help="post, question or answer", required=True)
    parser.add_argument('--num_words', type=int, help="", required=True)
    parser.add_argument('--max_sentence_len', type=int, help="", required=True)
    parser.add_argument('--epochs', type=int, help="", required=True)
    parser.add_argument('--batch_size', type=int, help="", required=True)
    parser.add_argument('--word_vector_dim', type=int, help="", required=True)
    parser.add_argument('--word_index_dir', help='', required=True)
    parser.add_argument('--model_name', help="", required=True)
    parser.add_argument('--train_embedding', type=bool, help="", default="True")
    parser.add_argument('--continue_training', help="", action='store_true')
    parser.add_argument('--initial_epoch', type=int, help='', default=0)
    parser.add_argument('--num_top_trans', type=int, help='', default=200)
    parser.add_argument('--lstm_dim', type=int, help='', default=128)
    parser.add_argument('--dropout_value', type=float, help='', default=0.15)
    parser.add_argument('--class_weight', help="", action='store_true')
    args = parser.parse_args()

    assert type(args.train_embedding) == bool
    train(args.dataset, args.data_path, args.post_tags_path, args.post_score_path, args.result_path,
          args.dataset_percent, args.test_size,
          args.post_type, args.num_words, args.word_vector_dim, args.train_embedding, args.max_sentence_len,
          args.epochs, args.batch_size, args.model_name, args.continue_training, args.initial_epoch, args.num_top_trans,
          args.lstm_dim, args.dropout_value, args.word_index_dir, args.class_weight)
