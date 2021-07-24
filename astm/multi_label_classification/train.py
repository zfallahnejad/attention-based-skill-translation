import os
import time
import keras
import random
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.style as style
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from utils import print_time, prepare_keras_input, prepare_word2vec_input, prepare_topic_vec_input
from model import SentenceClassifier, fbeta, macro_f1, evaluation, evaluation_top_label, perf_grid, \
    performance_table, performance_curves, top_words, top_words_2, integrate_scores, integrate_scores_2, \
    AttentionWithContext, AttentionWithContextV2

CWD = os.path.dirname(__file__)
JAVA_TOP_SO_TAGS = [
    "android", "swing", "eclipse", "spring", "hibernate", "arrays", "multithreading", "xml", "jsp", "string",
    "servlets", "maven", "java-ee", "mysql", "spring-mvc", "json", "regex", "tomcat", "jpa", "jdbc", "javascript",
    "arraylist", "web-services", "sql", "generics", "netbeans", "sockets", "user-interface", "jar", "html", "jsf",
    "database", "file", "google-app-engine", "gwt", "junit", "exception", "algorithm", "rest", "class", "performance",
    "applet", "image", "jtable", "c#", "jframe", "collections", "c++", "methods", "oop", "linux",
    "nullpointerexception", "jaxb", "parsing", "oracle", "concurrency", "php", "jpanel", "jboss", "object", "ant",
    "date", "selenium", "javafx", "jvm", "list", "struts2", "hashmap", "sorting", "awt", "http", "inheritance",
    "reflection", "hadoop", "windows", "loops", "unit-testing", "sqlite", "design-patterns", "serialization",
    "security", "intellij-idea", "file-io", "logging", "swt", "apache", "annotations", "jquery", "jersey", "scala",
    "libgdx", "osx", "encryption", "spring-security", "log4j", "python", "jni", "soap", "interface", "io"
]  # "java"

PHP_TOP_SO_TAGS = [
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
]  # "php"


def so_post_sampling(tag, post_tags_path, post_score_path, result_path, train_part, post_type, log_file):
    if tag == "java":
        TOP_SO_TAGS = JAVA_TOP_SO_TAGS
    elif tag == "php":
        TOP_SO_TAGS = PHP_TOP_SO_TAGS

    if os.path.exists(os.path.join(result_path, "sample_{}s_tr{}.pkl".format(post_type, train_part))):
        with open(os.path.join(result_path, "sample_{}s_tr{}.pkl".format(post_type, train_part)), "rb") as input_file:
            sample_post_tag = pickle.load(input_file)
    else:
        with open(post_tags_path, "rb") as input_file:
            post_tags = pickle.load(input_file)

        print("Number of input {}s: {}".format(post_type, len(post_tags)))
        log_file.write("Number of input {}s: {}\n".format(post_type, len(post_tags)))
        print("tag_count of input: ",
              [(t, [tag for pid, tags in post_tags.items() for tag in tags].count(t)) for t in TOP_SO_TAGS])
        log_file.write("tag_count of input: {}\n".format(
            [(t, [tag for pid, tags in post_tags.items() for tag in tags].count(t)) for t in TOP_SO_TAGS]))

        with open(post_score_path, "rb") as input_file:
            post_score = pickle.load(input_file)
            post_score = {pid: post_score[pid] for pid in post_score if pid in post_tags}
        # remove posts with negative score
        post_tags = {pid: tags for pid, tags in post_tags.items() if post_score[pid] >= 0}
        print("Number of input {}s with non-negative score: {}".format(post_type, len(post_tags)))
        log_file.write(
            "Number of input {}s with non-negative score: {}\n".format(post_type, len(post_tags)))
        print("tag_count of input: {}".format(
            [(t, [tag for pid, tags in post_tags.items() for tag in tags].count(t)) for t in TOP_SO_TAGS]))
        log_file.write("tag_count of input: {}\n".format(
            [(t, [tag for pid, tags in post_tags.items() for tag in tags].count(t)) for t in TOP_SO_TAGS]))

        posts = list(post_tags.keys())
        random.seed(7)
        random.shuffle(posts)
        sample_post_tag = {}
        num_train_samples = int(train_part * len(posts))
        for pid in posts[: num_train_samples]:
            sample_post_tag[pid] = post_tags[pid]

        with open(os.path.join(result_path, "sample_{}s_tr{}.pkl".format(post_type, train_part)), 'wb') as output:
            pickle.dump(sample_post_tag, output)

    print("#total sample {}s: {}".format(post_type, len(sample_post_tag)))
    log_file.write("#total sample {}s: {}\n".format(post_type, len(sample_post_tag)))
    all_tags = [tag for pid, tags in sample_post_tag.items() for tag in tags]
    print("class_count_weight of samples: {}".format(
        [(t, all_tags.count(t), len(all_tags) / all_tags.count(t)) for t in TOP_SO_TAGS]))
    log_file.write("class_count_weight of samples: {}\n".format(
        [(t, all_tags.count(t), len(all_tags) / all_tags.count(t)) for t in TOP_SO_TAGS]))
    return sample_post_tag


def preprocess_stack_overflow(data_path, post_tag, result_path, train_part, post_type, log_file):
    texts_path = os.path.join(result_path, "texts_{}_tr{}.pkl".format(post_type, train_part))
    labels_path = os.path.join(result_path, "labels_{}_tr{}.pkl".format(post_type, train_part))
    primitive_word_index_path = os.path.join(result_path,
                                             "primitive_word_index_{}_tr{}.pkl".format(post_type, train_part))
    multilabel_binarizer_path = os.path.join(result_path,
                                             "multilabel_binarizer_{}_tr{}.pkl".format(post_type, train_part))
    post_index_id_path = os.path.join(result_path, "post_index_id_{}_tr{}.pkl".format(post_type, train_part))
    post_id_length_path = os.path.join(result_path, "post_id_length_{}_tr{}.pkl".format(post_type, train_part))
    if os.path.exists(texts_path) and os.path.exists(labels_path) and os.path.exists(primitive_word_index_path):
        with open(texts_path, "rb") as input_file:
            texts = pickle.load(input_file)
        with open(labels_path, "rb") as input_file:
            labels = pickle.load(input_file)
        with open(primitive_word_index_path, "rb") as input_file:
            word_index = pickle.load(input_file)
        with open(multilabel_binarizer_path, "rb") as input_file:
            multilabel_binarizer = pickle.load(input_file)
        with open(post_index_id_path, "rb") as input_file:
            post_index_id = pickle.load(input_file)
        with open(post_id_length_path, "rb") as input_file:
            post_id_length = pickle.load(input_file)
        print("Number of unique words in this sample dataset: {}".format(len(word_index)))
        log_file.write("Number of unique words in this sample dataset: {}\n".format(len(word_index)))
        return texts, labels, word_index, multilabel_binarizer, post_index_id, post_id_length

    texts, labels = [], []
    all_docs = []
    post_index_id = {}
    post_id_length = {}
    with open(data_path, encoding="utf8") as infile:
        for line in infile:
            line = line.strip().split()
            post_id = line[0]
            if post_id in post_tag:
                post_text = line[1:]
                post_id_length[post_id] = len(post_text)
                doc = ' '.join(post_text)
                post_index_id[len(texts)] = post_id
                texts.append(doc)
                labels.append(post_tag[post_id])
                all_docs += post_text
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(labels)
    print("{} classes: {}".format(len(multilabel_binarizer.classes_), multilabel_binarizer.classes_))
    log_file.write("{} classes: {}\n".format(len(multilabel_binarizer.classes_), multilabel_binarizer.classes_))
    labels = multilabel_binarizer.transform(labels)

    words_freq = Counter(all_docs)
    del all_docs
    print("Number of unique words in this sample dataset: {}".format(len(words_freq)))
    log_file.write("Number of unique words in this sample dataset: {}\n".format(len(words_freq)))

    word_index = {}
    out_file = open(os.path.join(result_path, "primitive_word_index_{}_tr{}.txt".format(post_type, train_part)), "w",
                    encoding='utf8')
    for i, (word, freq) in enumerate(words_freq.most_common()):
        word_index[word] = i + 1
        out_file.write(word + "\t" + str(freq) + "\n")
    out_file.close()

    with open(texts_path, 'wb') as output:
        pickle.dump(texts, output)
    with open(labels_path, 'wb') as output:
        pickle.dump(labels, output)
    with open(primitive_word_index_path, 'wb') as output:
        pickle.dump(word_index, output)
    with open(multilabel_binarizer_path, 'wb') as output:
        pickle.dump(multilabel_binarizer, output)
    with open(post_index_id_path, 'wb') as output:
        pickle.dump(post_index_id, output)
    with open(post_id_length_path, 'wb') as output:
        pickle.dump(post_id_length, output)
    return texts, labels, word_index, multilabel_binarizer, post_index_id, post_id_length


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
        history: history callback of fitting a tensorflow keras model
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


def train(tag, data_path, post_tags_path, post_score_path, result_path, train_part, post_type, embedding_type,
          num_words, word_vector_dim, train_embedding, max_sentence_len, epochs, batch_size, validation_split,
          model_name, continue_training, initial_epoch, num_top_trans):
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    log_file = open(os.path.join(result_path, "log.txt"), "w")

    post_tag = so_post_sampling(tag, post_tags_path, post_score_path, result_path, train_part, post_type, log_file)
    texts, labels, word_index, multilabel_binarizer, post_index_id, post_id_length = preprocess_stack_overflow(
        data_path, post_tag, result_path, train_part, post_type, log_file)
    data = np.random.random((len(texts), max_sentence_len))
    embedding_matrix = None
    if embedding_type == "keras":
        data, word_index = prepare_keras_input(texts, num_words, max_sentence_len)
    elif embedding_type == "glove":
        # TODO
        exit(1)
    elif embedding_type == "word2vec":
        data, word_index, embedding_matrix = prepare_word2vec_input(tag, texts, word_index, num_words, max_sentence_len,
                                                                    word_vector_dim, train_part, post_type, result_path)
        print("Using trained word2vec embedding with shape: {}".format(embedding_matrix.shape))
        log_file.write("Using trained word2vec embedding with shape: {}\n".format(embedding_matrix.shape))
    elif embedding_type == "lda":
        data, word_index, embedding_matrix = prepare_topic_vec_input(tag, texts, word_index, num_words,
                                                                     max_sentence_len, word_vector_dim, train_part,
                                                                     post_type, result_path)
        print("Using trained word2vec embedding with shape: {}".format(embedding_matrix.shape))
        log_file.write("Using trained word2vec embedding with shape: {}\n".format(embedding_matrix.shape))

        length_diff_path = os.path.join(result_path, "post_length_diff_{}_tr{}.txt".format(post_type, train_part))
        with open(length_diff_path, "w", encoding='utf8') as out_file:
            for i, text in enumerate(texts):
                dsl = [e for e in data[i] if e != 0]
                out_file.write("{}\t{}\t{}\t{}\n".format(i, post_id_length[post_index_id[i]], len(dsl),
                                                         post_id_length[post_index_id[i]] - len(dsl)))
    else:
        print("{} is not valid!".format(embedding_type))
        exit(1)
    index_word = {index: word for word, index in word_index.items()}

    print('Shape of data tensor: {}\nShape of labels tensor: {}\n'.format(data.shape, labels.shape))

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=9000)
    print("x_train shape: {}, y_train shape: {}".format(x_train.shape, y_train.shape))
    print("x_test shape: {}, y_test shape: {}".format(x_test.shape, y_test.shape))
    log_file.write("x_train shape: {}, y_train shape: {}\n".format(x_train.shape, y_train.shape))
    log_file.write("x_test shape: {}, y_test shape: {}\n".format(x_test.shape, y_test.shape))
    with open(os.path.join(result_path, "data_{}_tr{}.pkl".format(post_type, train_part)), 'wb') as output:
        pickle.dump(data, output)
    with open(os.path.join(result_path, "train_test_{}_tr{}.pkl".format(post_type, train_part)),
              'wb') as output:
        pickle.dump([x_train, x_test, y_train, y_test], output)

    np.random.seed(7)
    tf.compat.v1.set_random_seed(20)
    num_classes = labels.shape[1]
    classifier = SentenceClassifier(max_sentence_len=max_sentence_len, num_words=num_words,
                                    word_vector_dim=word_vector_dim, label_count=num_classes,
                                    word_index=word_index, label_encoder=multilabel_binarizer)
    print("Train word embedding: {}".format(train_embedding))
    log_file.write("Train word embedding: {}\n".format(train_embedding))

    loss_function = 'binary_crossentropy'
    print("loss function: {}".format(loss_function))
    log_file.write("loss function: {}\n".format(loss_function))

    if continue_training:
        model = keras.models.load_model(os.path.join(result_path, 'model.h5'),
                                        custom_objects={"macro_f1": macro_f1, "fbeta": fbeta,
                                                        "AttentionWithContext": AttentionWithContext,
                                                        "AttentionWithContextV2": AttentionWithContextV2,
                                                        "tf": tf})
    else:
        if model_name == "ASTM1":
            if tag == "java":
                model = classifier.astm_1(train_embedding, embedding_matrix, 128, 0.15)
            elif tag == "php":
                model = classifier.astm_1(train_embedding, embedding_matrix, 64, 0.1)
            else:
                exit(1)
        elif model_name == "ASTM2":
            if tag == "java":
                model = classifier.astm_2(train_embedding, embedding_matrix, 64)
            elif tag == "php":
                model = classifier.astm_2(train_embedding, embedding_matrix, 32)
            else:
                exit(1)
        else:
            exit(1)

        model.compile(loss=loss_function, optimizer='adam',
                      metrics=[macro_f1, 'acc', 'categorical_accuracy', 'top_k_categorical_accuracy', fbeta])

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
    print("Without class_weight!\n")
    log_file.write("Without class_weight!\n")
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                        callbacks=callbacks, initial_epoch=initial_epoch)
    print('\nTraining took {}'.format(print_time(time.time() - start)))
    log_file.write('\nTraining took {}\n'.format(print_time(time.time() - start)))

    if model_name == "ASTM1":
        print("Extract top words of test data:")
        log_file.write("Extract top words of test data:\n")
        test_word_scores = top_words(model, x_test, y_test, "Test", multilabel_binarizer.classes_, index_word, 64,
                                     log_file, num_top_trans)

        print("Extract top words of train data:")
        log_file.write("Extract top words of train data:\n")
        train_word_scores = top_words(model, x_train, y_train, "Train", multilabel_binarizer.classes_, index_word, 64,
                                      log_file, num_top_trans)
        integrate_scores(train_word_scores, test_word_scores, multilabel_binarizer.classes_, log_file)
    elif model_name == "ASTM2":
        print("Extract top words of test data:")
        log_file.write("Extract top words of test data:\n")
        top_words_2(model, x_test, y_test, "Test", multilabel_binarizer.classes_, index_word, 64, log_file, result_path,
                    num_top_trans)

        print("Extract top words of train data:")
        log_file.write("Extract top words of train data:\n")
        top_words_2(model, x_train, y_train, "Train", multilabel_binarizer.classes_, index_word, 64, log_file,
                    result_path, num_top_trans)
        integrate_scores_2(multilabel_binarizer.classes_, log_file, result_path)

    if history.history:
        plot_1(history, result_path)
        losses, val_losses, macro_f1s, val_macro_f1s = learning_curves(history, result_path)
        print("Validation Macro Loss: %.2f" % val_losses[-1])
        print("Validation Macro F1-score: %.2f" % val_macro_f1s[-1])
        log_file.write("Validation Macro Loss: %.2f\n" % val_losses[-1])
        log_file.write("Validation Macro F1-score: %.2f\n" % val_macro_f1s[-1])

    evaluation(x_test, y_test, num_classes, result_path, log_file)
    # evaluation_top_label(x_test, y_test, num_classes, result_path, log_file)
    max_perf, grid = performance_table(x_test, y_test, multilabel_binarizer.classes_, model, result_path, log_file)
    performance_curves(max_perf, grid, model, multilabel_binarizer.classes_, x_test, result_path)

    if model_name == "ASTM1":
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('attention_with_context').output)

        print("Extract words scores of data:")
        weighted_seq, attention = intermediate_layer_model.predict(data, batch_size=64)
        print("attention shape: {}".format(attention.shape))
        score_file = os.path.join(result_path, "data_scores_{}_tr{}.txt".format(post_type, train_part))
        with open(score_file, "w", encoding='utf8') as out_file:
            for i, sample in enumerate(data):
                sample_scores = []
                sample_labels = [multilabel_binarizer.classes_[j] for j, l in enumerate(labels[i]) if l == 1]
                for k, word_index in enumerate(sample):
                    if word_index == 0:
                        continue
                    word = index_word[word_index]
                    score = attention[i][k]
                    sample_scores.append((word, score))
                out_file.write("{}\t{}\t{}\n".format(post_index_id[i], sample_labels, sample_scores))
    elif model_name == "ASTM2":
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer('attention_with_context_v2').output)

        last_index = 0
        num_parts = 100
        print("Extract words scores of data:")
        score_file = os.path.join(result_path, "data_scores_{}_tr{}.txt".format(post_type, train_part))
        with open(score_file, "w", encoding='utf8') as out_file:
            for p, xp in enumerate(np.array_split(data, num_parts)):
                print("x part {} shape: {}".format(p, xp.shape))
                weighted_seq, attention = intermediate_layer_model.predict(xp, batch_size=64)
                del weighted_seq
                print("attention shape: {}".format(attention.shape))
                log_file.write("attention shape: {}\n".format(attention.shape))
                for i, sample in enumerate(xp):
                    sample_labels = [j for j, l in enumerate(labels[last_index]) if l == 1]
                    sample_label_scores = {l: [] for l in sample_labels}
                    for k, word_index in enumerate(sample):
                        if word_index == 0:
                            continue
                        word = index_word[word_index]
                        for label in sample_labels:
                            score = attention[i][label][k]
                            sample_label_scores[label].append((word, score))
                    for l in sample_labels:
                        out_file.write("{}\t{}\t{}\t{}\n".format(
                            post_index_id[last_index], [multilabel_binarizer.classes_[l] for l in sample_labels],
                            multilabel_binarizer.classes_[l], sample_label_scores[l]))
                    last_index += 1

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', help="", default='java')
    parser.add_argument('--data_path', help='', required=True)
    parser.add_argument('--post_tags_path', help='', required=True)
    parser.add_argument('--post_score_path', help='', required=True)
    parser.add_argument('--result_path', help='', required=True)
    parser.add_argument('--train_part', type=float, help="", required=True)
    parser.add_argument('--post_type', help="post, question or answer", required=True)
    parser.add_argument('--embedding_type', help="keras, word2vec, glove or lda", required=True)
    parser.add_argument('--num_words', type=int, help="", required=True)
    parser.add_argument('--max_sentence_len', type=int, help="", required=True)
    parser.add_argument('--epochs', type=int, help="", required=True)
    parser.add_argument('--batch_size', type=int, help="", required=True)
    parser.add_argument('--validation_split', type=float, help="", required=True)
    parser.add_argument('--word_vector_dim', type=int, help="", required=True)
    parser.add_argument('--model_name', help="", required=True)
    parser.add_argument('--train_embedding', type=bool, help="", default="True")
    parser.add_argument('--continue_training', help="", action='store_true')
    parser.add_argument('--initial_epoch', type=int, help='', default=0)
    parser.add_argument('--num_top_trans', type=int, help='', default=10)
    args = parser.parse_args()

    train(args.tag, args.data_path, args.post_tags_path, args.post_score_path, args.result_path,
          args.train_part, args.post_type,
          args.embedding_type, args.num_words, args.word_vector_dim, args.train_embedding, args.max_sentence_len,
          args.epochs, args.batch_size, args.validation_split, args.model_name, args.continue_training,
          args.initial_epoch, args.num_top_trans)
