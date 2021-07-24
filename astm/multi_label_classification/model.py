import os
import keras
import imgkit
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.style as style
from keras.layers import Dense, Input, Lambda
from keras.layers import Embedding, Dropout
from keras.layers import Bidirectional, CuDNNLSTM
from keras.models import Model
from keras import backend as K
from keras.engine import Layer
from keras import initializers, regularizers, constraints
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score, fbeta_score, multilabel_confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, \
    accuracy_score


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        scores = K.expand_dims(a)
        weighted_input = x * scores
        return [K.sum(weighted_input, axis=1), a]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]


class AttentionWithContextV2(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, num_classes,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.num_classes = num_classes
        self.bias = bias
        super(AttentionWithContextV2, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((self.num_classes, input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContextV2, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait_list = [dot_product(uit, up) for up in tf.unstack(self.u, axis=0)]  # 100 ta (?, 240)
        a_list = [K.exp(ait) for ait in ait_list]  # 100 ta (?, 240)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a_list = [a * K.cast(mask, K.floatx()) for a in a_list]

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a_list = [a / K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx()) for a in a_list]

        score_list = [K.expand_dims(a) for a in a_list]  # 100 ta (?, 240, 1)
        weighted_input_list = [x * scores for scores in score_list]  # 100 ta (?, 240, 256)
        output_list = [K.sum(weighted_input, axis=1) for weighted_input in weighted_input_list]

        output = K.stack(output_list, axis=1)
        attention_scores = K.stack(a_list, axis=1)
        # print("output.shape:", output.shape)  # (?, 100, 256)
        # print("attention_scores.shape:", attention_scores.shape)  # (?, 100, 240)
        return [output, attention_scores]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.num_classes, input_shape[-1]), (input_shape[0], self.num_classes, input_shape[1])]

    def get_config(self):
        base_config = super(AttentionWithContextV2, self).get_config()
        base_config["num_classes"] = self.num_classes
        return base_config


def create_custom_objects():
    instance_holder = {"instance": None}

    class ClassWrapper(AttentionWithContext):
        def __init__(self, *args, **kwargs):
            instance_holder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instance_holder["instance"], "loss_function")
        return method(*args)

    def accuracy(*args):
        method = getattr(instance_holder["instance"], "accuracy")
        return method(*args)

    return {"ClassWrapper": ClassWrapper, "AttentionWithContext": ClassWrapper, "loss": loss,
            "accuracy": accuracy}


def fbeta(y_true, y_pred, beta=2):
    '''
    calculate fbeta score for multi-class/label classification
    '''
    # clip predictions
    y_pred = K.clip(y_pred, 0, 1)
    # calculate elements
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + K.epsilon())
    # calculate recall
    r = tp / (tp + fn + K.epsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = K.mean((1 + bb) * (p * r) / (bb * p + r + K.epsilon()))
    return fbeta_score


def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = K.cast(K.greater(K.clip(y_hat, 0, 1), thresh), tf.float32)
    tp = K.cast(tf.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = K.cast(tf.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = K.cast(tf.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


def evaluation(x_test, y_test, num_classes, result_path, log_file):
    model = keras.models.load_model(os.path.join(result_path, 'model.h5'),
                                    custom_objects={"macro_f1": macro_f1, "fbeta": fbeta,
                                                    "AttentionWithContext": AttentionWithContext,
                                                    "AttentionWithContextV2": AttentionWithContextV2,
                                                    "tf": tf})
    metrics = model.evaluate(x_test, y_test)
    for i in range(len(metrics)):
        print("{}: {}".format(model.metrics_names[i], metrics[i]))
        log_file.write("{}: {}\n".format(model.metrics_names[i], metrics[i]))

    preds = model.predict(x_test)
    roc = roc_auc_score(y_test, preds)
    print("\nROC={}\n".format(roc))
    log_file.write("\nROC={}\n".format(roc))

    preds_t05 = (model.predict(x_test) >= 0.5).astype(int)

    accuracy_classification_score = accuracy_score(y_test, preds_t05)
    print("\naccuracy_classification_score={}\n".format(accuracy_classification_score))
    log_file.write("\naccuracy_classification_score={}\n".format(accuracy_classification_score))

    f1_score_macro = f1_score(y_test, preds_t05, average='macro')
    f1_score_micro = f1_score(y_test, preds_t05, average='micro')
    f1_score_weighted = f1_score(y_test, preds_t05, average='weighted')
    f1_score_samples = f1_score(y_test, preds_t05, average='samples')
    roc = roc_auc_score(y_test, preds_t05)
    print("macro f1 score (th 0.5)={}\nmicro f1 score (th 0.5)={}\n".format(f1_score_macro, f1_score_micro))
    print("weighted f1 score (th 0.5)={}\nROC (th 0.5)={}\n".format(f1_score_weighted, roc))
    print("samples f1 score={}\n".format(f1_score_samples))
    log_file.write("macro f1 score (th 0.5)={}\nmicro f1 score (th 0.5)={}\n".format(f1_score_macro, f1_score_micro))
    log_file.write("weighted f1 score (th 0.5)={}\nROC (th 0.5)={}\n".format(f1_score_weighted, roc))
    log_file.write("samples f1 score={}\n".format(f1_score_samples))

    report = classification_report(y_test, preds_t05)
    print("classification report (th 0.5):\n{}\n".format(report))
    log_file.write("classification report (th 0.5):\n{}\n".format(report))

    multilabel_cmatrix = multilabel_confusion_matrix(y_test, preds_t05)
    print("multilabel_confusion_matrix:\n{}\n".format(multilabel_cmatrix))
    log_file.write("multilabel_confusion_matrix:\n{}\n".format(multilabel_cmatrix))

    prf_macro = precision_recall_fscore_support(y_test, preds_t05, average='macro')
    prf_micro = precision_recall_fscore_support(y_test, preds_t05, average='micro')
    prf_weighted = precision_recall_fscore_support(y_test, preds_t05, average='weighted')
    prf_samples = precision_recall_fscore_support(y_test, preds_t05, average='samples')
    print("macro precision_recall_fscore_support={}\n".format(prf_macro))
    print("micro precision_recall_fscore_support={}\n".format(prf_micro))
    print("weighted precision_recall_fscore_support={}\n".format(prf_weighted))
    print("samples precision_recall_fscore_support={}\n".format(prf_samples))
    log_file.write("macro precision_recall_fscore_support={}\n".format(prf_macro))
    log_file.write("micro precision_recall_fscore_support={}\n".format(prf_micro))
    log_file.write("weighted precision_recall_fscore_support={}\n".format(prf_weighted))
    log_file.write("samples precision_recall_fscore_support={}\n".format(prf_samples))

    fbeta_score_macro = fbeta_score(y_test, preds_t05, average='macro', beta=0.5)
    fbeta_score_micro = fbeta_score(y_test, preds_t05, average='micro', beta=0.5)
    fbeta_score_weighted = fbeta_score(y_test, preds_t05, average='weighted', beta=0.5)
    fbata_score_samples = fbeta_score(y_test, preds_t05, average='samples', beta=0.5)
    print(
        "macro fbeta score(beta=0.5)={}\nmicro fbeta score(beta=0.5)={}\n".format(fbeta_score_macro, fbeta_score_micro))
    print("weighted fbeta score(beta=0.5)={}\nsamples fbeta score(beta=0.5)={}\n".format(fbeta_score_weighted,
                                                                                         fbata_score_samples))
    log_file.write(
        "macro fbeta score(beta=0.5)={}\nmicro fbeta score(beta=0.5)={}\n".format(fbeta_score_macro, fbeta_score_micro))
    log_file.write("macro fbeta score(beta=0.5)={}\nmicro fbeta score(beta=0.5)={}\n".format(fbeta_score_weighted,
                                                                                             fbata_score_samples))


def evaluation_top_label(x_test, y_test, num_classes, result_path, log_file):
    model = keras.models.load_model(os.path.join(result_path, 'model.h5'),
                                    custom_objects={"macro_f1": macro_f1, "fbeta": fbeta,
                                                    "AttentionWithContext": AttentionWithContext,
                                                    "AttentionWithContextV2": AttentionWithContextV2,
                                                    "tf": tf})
    y_test_bool = np.argmax(y_test, axis=1)
    y_pred = model.predict(x_test, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    precision = precision_score(y_test_bool, y_pred_bool, average="macro")
    recall = recall_score(y_test_bool, y_pred_bool, average="macro")
    f1 = f1_score(y_test_bool, y_pred_bool, average="macro")
    report = classification_report(y_test_bool, y_pred_bool)

    print("precision_score: {}\n".format(precision))
    print("recall_score: {}\n".format(recall))
    print("f1_score: {}\n".format(f1))
    print("classification_report: {}\n".format(report))

    log_file.write("precision_score: {}\n".format(precision))
    log_file.write("recall_score: {}\n".format(recall))
    log_file.write("f1_score: {}\n".format(f1))
    log_file.write("classification_report: {}\n".format(report))


def perf_grid(ds, target, label_names, model, n_thresh=100):
    """Computes the performance table containing target, label names,
    label frequencies, thresholds between 0 and 1, number of tp, fp, fn,
    precision, recall and f-score metrics for each label.

    Args:
        ds (tf.data.Datatset): contains the features array
        target (numpy array): target matrix of shape (BATCH_SIZE, N_LABELS)
        label_names (list of strings): column names in target matrix
        model (tensorflow keras model): model to use for prediction
        n_thresh (int) : number of thresholds to try

    Returns:
        grid (Pandas dataframe): performance table
    """
    # Get predictions
    y_hat_val = model.predict(ds)
    # Define target matrix
    y_val = target
    # Find label frequencies in the validation set
    label_freq = target.sum(axis=0)
    # Get label indexes
    label_index = [i for i in range(len(label_names))]
    # Define thresholds
    thresholds = np.linspace(0, 1, n_thresh + 1).astype(np.float32)

    # Compute all metrics for all labels
    ids, labels, freqs, tps, fps, fns, precisions, recalls, f1s = [], [], [], [], [], [], [], [], []
    for l in label_index:
        for thresh in thresholds:
            ids.append(l)
            labels.append(label_names[l])
            freqs.append(round(label_freq[l] / len(y_val), 2))
            y_hat = y_hat_val[:, l]
            y = y_val[:, l]
            y_pred = y_hat > thresh
            tp = np.count_nonzero(y_pred * y)
            fp = np.count_nonzero(y_pred * (1 - y))
            fn = np.count_nonzero((1 - y_pred) * y)
            precision = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)
            f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    # Create the performance dataframe
    grid = pd.DataFrame({
        'id': ids,
        'label': labels,
        'freq': freqs,
        'threshold': list(thresholds) * len(label_index),
        'tp': tps,
        'fp': fps,
        'fn': fns,
        'precision': precisions,
        'recall': recalls,
        'f1': f1s})

    grid = grid[['id', 'label', 'freq', 'threshold', 'tp', 'fn', 'fp', 'precision', 'recall', 'f1']]
    return grid


def performance_table(x_test, y_test, labels_names, model, result_path, log_file):
    # Performance table with the first model (macro soft-f1 loss)
    grid = perf_grid(x_test, y_test, labels_names, model)
    print(grid.head())
    log_file.write("{}\n".format(grid.head()))

    # Get the maximum F1-score for each label when using the model and varying the threshold
    max_perf = grid.groupby(['id', 'label', 'freq'])[['f1']].max().sort_values('f1', ascending=False).reset_index()
    max_perf.rename(columns={'f1': 'f1max'}, inplace=True)
    styled_table = max_perf.style.background_gradient(subset=['freq', 'f1max'],
                                                      cmap=sns.light_palette("lightgreen", as_cmap=True))
    print("Correlation between label frequency and optimal F1: %.2f" % max_perf['freq'].corr(max_perf['f1max']))
    log_file.write(
        "Correlation between label frequency and optimal F1: %.2f \n" % max_perf['freq'].corr(max_perf['f1max']))
    html = styled_table.render()
    imgkit.from_string(html, os.path.join(result_path, 'performance_styled_table.png'))
    return max_perf, grid


def performance_curves(max_perf, grid, model, label_names, x_test, result_path):
    top5 = max_perf.head(5)['id']

    style.use("default")
    for i, l in enumerate(top5):
        label_grid = grid.loc[grid['id'] == l, ['precision', 'recall', 'f1']]
        label_grid = label_grid.reset_index().drop('index', axis=1)

        plt.figure(figsize=(9, 3))
        ax = plt.subplot(1, 2, 1)
        plt.xticks(ticks=np.arange(0, 110, 10), labels=np.arange(0, 110, 10) / 100, fontsize=10)
        plt.yticks(fontsize=8)
        plt.title('Performance curves - Label ' + str(l) + ' (' + label_names[l] + ')\nMacro Soft-F1', fontsize=10)
        label_grid.plot(ax=ax)
        plt.tight_layout()
    plt.savefig(os.path.join(result_path, "performance_curve_1.png"))

    style.use("default")
    y_hat_test = model.predict(x_test)
    for l in top5:
        plt.figure(figsize=(9, 3))

        ax = plt.subplot(1, 2, 1)
        plt.xticks(ticks=np.arange(0, 1.1, 0.1), fontsize=8)
        plt.yticks(fontsize=8)
        plt.title('Probability distribution - Label ' + str(l) + ' (' + label_names[l] + ')\nBCE', fontsize=10)
        plt.xlim(0, 1)
        ax = sns.distplot(y_hat_test[:, l], bins=30, kde=True, color="b")
        plt.tight_layout()
    plt.savefig(os.path.join(result_path, "performance_curve_2.png"))


def top_words(model, x, y, data_type, labels_names, index_word, batch_size, log_file,
              num_top_trans=100):
    metrics = model.evaluate(x, y)
    for i in range(len(metrics)):
        # print("{}: {}".format(model.metrics_names[i], metrics[i]))
        log_file.write("{}: {}\n".format(model.metrics_names[i], metrics[i]))

    layer_name = 'attention_with_context'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    weighted_seq, attention = intermediate_layer_model.predict(x, batch_size=batch_size)
    # print("attention shape: {}".format(attention.shape))
    log_file.write("attention shape: {}\n".format(attention.shape))

    word_scores = {}
    for label in labels_names:
        word_scores[label] = {}
    for i, sample in enumerate(x):
        sample_labels = [labels_names[j] for j, l in enumerate(y[i]) if l == 1]
        for k, word_index in enumerate(sample):
            if word_index == 0:
                continue
            word = index_word[word_index]
            score = attention[i][k]
            for label in sample_labels:
                if word in word_scores[label]:
                    word_scores[label][word] += score
                else:
                    word_scores[label][word] = score
    # print("Word Scores:")
    log_file.write("Word Scores:\n")
    for label in word_scores:
        sorted_word_scores = sorted(word_scores[label].items(), key=lambda x: x[1], reverse=True)
        # print("Label={}, Translations={}".format(label, str(sorted_word_scores[:100])))
        log_file.write("Type=Word, Dataset={}, Label={}, Translations={}\n".format(
            data_type, label, str(sorted_word_scores[:num_top_trans])))
    return word_scores


def top_words_2(model, x, y, data_type, labels_names, index_word, batch_size, log_file, result_path, num_top_trans=100):
    word_scores = {}
    for label in labels_names:
        word_scores[label] = {}

    layer_name = 'attention_with_context_v2'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    last_index = 0
    num_parts = 50
    for p, xp in enumerate(np.array_split(x, num_parts)):
        print("x part {} shape: {}".format(p, xp.shape))
        weighted_seq, attention = intermediate_layer_model.predict(xp, batch_size=batch_size)
        del weighted_seq
        print("attention shape: {}".format(attention.shape))
        # log_file.write("attention shape: {}\n".format(attention.shape))

        for i, sample in enumerate(xp):
            sample_labels = [j for j, l in enumerate(y[last_index]) if l == 1]
            for k, word_index in enumerate(sample):
                if word_index == 0:
                    continue
                word = index_word[word_index]
                for label in sample_labels:
                    score = attention[i][label][k]
                    if word in word_scores[labels_names[label]]:
                        word_scores[labels_names[label]][word] += score
                    else:
                        word_scores[labels_names[label]][word] = score
            last_index += 1

    # print("Word Scores:")
    log_file.write("Word Scores:\n")
    for label in word_scores:
        sorted_word_scores = sorted(word_scores[label].items(), key=lambda x: x[1], reverse=True)
        # print("Label={}, Translations={}".format(label, str(sorted_word_scores[:100])))
        log_file.write("Type=Word, Dataset={}, Label={}, Translations={}\n".format(
            data_type, label, str(sorted_word_scores[:num_top_trans])))
    with open(os.path.join(result_path, "word_scores_{}.txt".format(data_type.lower())), 'w',
              encoding="utf8") as outfile:
        for label in word_scores:
            for word in word_scores[label]:
                outfile.write(label + "\t" + word + "\t" + str(word_scores[label][word]) + "\n")


def integrate_scores(train_word_scores, test_word_scores, labels_names, log_file):
    word_scores = {}
    for label in labels_names:
        word_scores[label] = {}
        for word in train_word_scores[label]:
            word_scores[label][word] = train_word_scores[label][word]
        for word in test_word_scores[label]:
            if word in word_scores[label]:
                word_scores[label][word] += test_word_scores[label][word]
            else:
                word_scores[label][word] = test_word_scores[label][word]
    log_file.write("Word Scores:\n")
    for label in word_scores:
        sorted_word_scores = sorted(word_scores[label].items(), key=lambda x: x[1], reverse=True)
        # print("Label={}, Translations={}".format(label, str(sorted_word_scores[:100])))
        log_file.write(
            "Type=Word, Dataset=All, Label={}, Translations={}\n".format(label, str(sorted_word_scores[:100])))


def integrate_scores_2(labels_names, log_file, result_path):
    word_scores = {}
    for label in labels_names:
        word_scores[label] = {}
    with open(os.path.join(result_path, "word_scores_test.txt"), encoding="utf8") as infile:
        for line in infile:
            label, word, score = line.strip().split('\t')
            word_scores[label][word] = float(score)
    with open(os.path.join(result_path, "word_scores_train.txt"), encoding="utf8") as infile:
        for line in infile:
            label, word, score = line.strip().split('\t')
            if word in word_scores[label]:
                word_scores[label][word] += float(score)
            else:
                word_scores[label][word] = float(score)

    log_file.write("Word Scores:\n")
    for label in word_scores:
        sorted_word_scores = sorted(word_scores[label].items(), key=lambda x: x[1], reverse=True)
        # print("Label={}, Translations={}".format(label, str(sorted_word_scores[:100])))
        log_file.write(
            "Type=Word, Dataset=All, Label={}, Translations={}\n".format(label, str(sorted_word_scores[:100])))


class SentenceClassifier:
    def __init__(self, max_sentence_len, num_words, word_vector_dim, label_count, word_index, label_encoder):
        self.MAX_SEQUENCE_LENGTH = max_sentence_len
        self.EMBEDDING_DIM = word_vector_dim
        self.LABEL_COUNT = label_count
        self.WORD_INDEX = word_index
        self.LABEL_ENCODER = label_encoder
        self.NUM_WORDS = num_words

    def astm_1(self, train_embedding, embedding_matrix, lstm_dim, dropout_value):
        inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        if embedding_matrix is None:
            embedding = Embedding(input_dim=(self.NUM_WORDS), output_dim=self.EMBEDDING_DIM,
                                  input_length=self.MAX_SEQUENCE_LENGTH, trainable=train_embedding)(inputs)
        else:
            embedding = Embedding(input_dim=(self.NUM_WORDS), output_dim=self.EMBEDDING_DIM, weights=[embedding_matrix],
                                  input_length=self.MAX_SEQUENCE_LENGTH, trainable=train_embedding)(inputs)
        word_seq = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True, kernel_regularizer=regularizers.l2(1e-13)))(
            embedding)
        word_seq = Dropout(dropout_value)(word_seq)
        X, attention_scores = AttentionWithContext(name='attention_with_context')(word_seq)
        X = Dropout(dropout_value)(X)
        outputs = Dense(units=self.LABEL_COUNT, activation="sigmoid")(X)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def astm_2(self, train_embedding, embedding_matrix, lstm_dim):
        """
        Makes uses of Keras functional API for constructing the model.
        If load_saved=1, THEN load old model, ELSE train new model
        """
        inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        if embedding_matrix is None:
            embedding = Embedding(input_dim=(self.NUM_WORDS), output_dim=self.EMBEDDING_DIM,
                                  input_length=self.MAX_SEQUENCE_LENGTH, trainable=train_embedding)(inputs)
        else:
            embedding = Embedding(input_dim=(self.NUM_WORDS), output_dim=self.EMBEDDING_DIM, weights=[embedding_matrix],
                                  input_length=self.MAX_SEQUENCE_LENGTH, trainable=train_embedding)(inputs)
        word_seq = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True, kernel_regularizer=regularizers.l2(1e-13)))(
            embedding)
        word_seq = Dropout(0.15)(word_seq)

        X, attention_scores = AttentionWithContextV2(name='attention_with_context_v2', num_classes=self.LABEL_COUNT)(
            word_seq)
        X = Dropout(0.15)(X)
        # print("X.shape: ", X.shape)  # (?, 100, 256)
        # print("attention_scores.shape: ", attention_scores.shape)  # (?, 100, 240)

        unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(X)
        final_prediction = [Dense(units=1, activation="sigmoid")(xr) for xr in unstacked]
        outputs = Lambda(lambda x: K.squeeze(K.stack(x, axis=1), axis=-1))(final_prediction)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def tag_question(self, model, question_id, question, true_labels):
        question_words = [w for w in question.split(' ') if w in self.WORD_INDEX]
        question_encoded = [[self.WORD_INDEX[w] for w in question_words]]
        question_encoded_padded = pad_sequences(question_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')
        predictions = model.predict(question_encoded_padded)

        possible_tags = []
        for i, probability in enumerate(predictions[0]):
            if probability >= 0.05:
                possible_tags.append([self.LABEL_ENCODER.classes_[i], probability])

        possible_tags.sort(reverse=True, key=lambda x: x[
            1])  # sort in place on the basis of the probability in each sub-list in descending order

        layer_name = 'attention_with_context'
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        weighted_seq, attention = intermediate_layer_model.predict(question_encoded_padded)
        word_score = {w: attention[0][i] for i, w in enumerate(question_words)}
        sorted_word_scores = sorted(word_score.items(), key=lambda item: item[1], reverse=True)
        print("question={}, true labels={}, predicted labels={}, word_scores={}".format(question_id, true_labels,
                                                                                        possible_tags,
                                                                                        sorted_word_scores))
