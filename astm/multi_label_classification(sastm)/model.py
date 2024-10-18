import os
import imgkit
import tensorflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from tensorflow.keras.layers import Layer, Dense, Input, Lambda
from tensorflow.keras.layers import Embedding, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
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

    def get_config(self):
        base_config = super(AttentionWithContext, self).get_config()
        return base_config


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

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(self.num_classes, input_shape[-1],),
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
        ait_list = [dot_product(uit, up) for up in tensorflow.unstack(self.u, axis=0)]  # 100 ta (?, 240)
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


class AttentionWithContextV3(Layer):
    def __init__(self, num_words, embedding_dim, max_sentence_len,
                 # bias=True,
                 **kwargs):
        self.num_words = num_words
        self.embedding_dim = embedding_dim
        self.max_sentence_len = max_sentence_len

        super(AttentionWithContextV3, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttentionWithContextV3, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, layer_inputs, mask=None):
        queries, keys, values = layer_inputs[0], layer_inputs[1], layer_inputs[2]
        # print("queries.shape:", queries.shape)  # queries.shape: (None, 256, 50)
        # print("keys.shape:", keys.shape)  # keys.shape: (None, 256, 50)
        # print("values.shape:", values.shape)  # values.shape: (None, 256, 50)

        # qk = dot_product(queries, keys)
        # print("qk.shape:", qk.shape)  # qk.shape: (None, 256, None, 256)
        keys_transpose = tensorflow.transpose(keys, perm=[0, 2, 1])
        # print("keys_transpose.shape:", keys_transpose.shape)  # (None, 50, 256)
        qk = tensorflow.matmul(queries, keys_transpose)
        # print("qk.shape:", qk.shape)  # (None, 256, 256)
        scores = qk / np.sqrt(self.embedding_dim)
        # print("scores.shape:", scores.shape)  # (None, 256, 256)
        scores_softmax = K.softmax(scores, axis=-1)
        # print("scores_softmax.shape:", scores_softmax.shape)  # (None, 256, 256)
        scores_mul_values = tensorflow.matmul(scores_softmax, values)
        # print("scores_mul_values.shape:", scores_mul_values.shape)  # (None, 256, 50)
        return [scores_mul_values, scores_softmax]

    def compute_output_shape(self, input_shape):
        return [input_shape[2], (input_shape[2][0], input_shape[2][1], input_shape[2][1])]

    def get_config(self):
        base_config = super(AttentionWithContextV3, self).get_config()
        base_config["num_words"] = self.num_words
        base_config["embedding_dim"] = self.embedding_dim
        base_config["max_sentence_len"] = self.max_sentence_len
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
    y_pred = K.cast(K.greater(K.clip(y_hat, 0, 1), thresh), tensorflow.float32)
    tp = K.cast(tensorflow.math.count_nonzero(y_pred * y, axis=0), tensorflow.float32)
    fp = K.cast(tensorflow.math.count_nonzero(y_pred * (1 - y), axis=0), tensorflow.float32)
    fn = K.cast(tensorflow.math.count_nonzero((1 - y_pred) * y, axis=0), tensorflow.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    return tensorflow.reduce_mean(input_tensor=f1)


def get_evaluation_metric(metric_name, y, pred):
    try:
        if metric_name == "classification_report":
            return classification_report(y, pred)
        elif metric_name == "roc_auc_score":
            return roc_auc_score(y, pred)
        elif metric_name == "accuracy_score":
            return accuracy_score(y, pred)
        elif metric_name == "macro_f1_score":
            return f1_score(y, pred, average='macro')
        elif metric_name == "micro_f1_score":
            return f1_score(y, pred, average='micro')
        elif metric_name == "weighted_f1_score":
            return f1_score(y, pred, average='weighted')
        elif metric_name == "samples_f1_score":
            return f1_score(y, pred, average='samples')
        elif metric_name == "macro_precision_recall_fscore_support":
            return precision_recall_fscore_support(y, pred, average='macro')
        elif metric_name == "micro_precision_recall_fscore_support":
            return precision_recall_fscore_support(y, pred, average='micro')
        elif metric_name == "weighted_precision_recall_fscore_support":
            return precision_recall_fscore_support(y, pred, average='weighted')
        elif metric_name == "samples_precision_recall_fscore_support":
            return precision_recall_fscore_support(y, pred, average='samples')
        elif metric_name == "macro_fbeta_score":
            return fbeta_score(y, pred, average='macro', beta=0.5)
        elif metric_name == "micro_fbeta_score":
            return fbeta_score(y, pred, average='micro', beta=0.5)
        elif metric_name == "weighted_fbeta_score":
            return fbeta_score(y, pred, average='weighted', beta=0.5)
        elif metric_name == "samples_fbeta_score":
            return fbeta_score(y, pred, average='samples', beta=0.5)
        elif metric_name == "multilabel_confusion_matrix":
            return multilabel_confusion_matrix(y, pred)
    except:
        return None


def evaluation(y_test, preds, log_file):
    roc = get_evaluation_metric("roc_auc_score", y_test, preds)
    accuracy_classification_score = get_evaluation_metric("accuracy_score", y_test, preds)
    f1_score_macro = get_evaluation_metric("macro_f1_score", y_test, preds)
    f1_score_micro = get_evaluation_metric("micro_f1_score", y_test, preds)
    f1_score_weighted = get_evaluation_metric("weighted_f1_score", y_test, preds)
    f1_score_samples = get_evaluation_metric("samples_f1_score", y_test, preds)
    report = get_evaluation_metric("classification_report", y_test, preds)
    prf_macro = get_evaluation_metric("macro_precision_recall_fscore_support", y_test, preds)
    prf_micro = get_evaluation_metric("micro_precision_recall_fscore_support", y_test, preds)
    prf_weighted = get_evaluation_metric("weighted_precision_recall_fscore_support", y_test, preds)
    prf_samples = get_evaluation_metric("samples_precision_recall_fscore_support", y_test, preds)
    fbeta_score_macro = get_evaluation_metric("macro_fbeta_score", y_test, preds)
    fbeta_score_micro = get_evaluation_metric("micro_fbeta_score", y_test, preds)
    fbeta_score_weighted = get_evaluation_metric("weighted_fbeta_score", y_test, preds)
    fbeta_score_samples = get_evaluation_metric("samples_fbeta_score", y_test, preds)
    multilabel_cmatrix = get_evaluation_metric("multilabel_confusion_matrix", y_test, preds)
    print(
        "\naccuracy_classification_score={}\n\nROC={}\n"
        "macro f1 score={}\nmicro f1 score={}\nweighted f1 score={}\nsamples f1 score={}\n"
        "macro precision_recall_fscore_support={}\nmicro precision_recall_fscore_support={}\n"
        "weighted precision_recall_fscore_support={}\nsamples precision_recall_fscore_support={}\n"
        "macro fbeta score={}\nmicro fbeta score={}\nweighted fbeta score={}\nsamples fbeta score={}\n"
        "classification report:\n{}\nmultilabel_confusion_matrix:\n{}\n".format(
            accuracy_classification_score, roc, f1_score_macro, f1_score_micro, f1_score_weighted, f1_score_samples,
            prf_macro, prf_micro, prf_weighted, prf_samples,
            fbeta_score_macro, fbeta_score_micro, fbeta_score_weighted, fbeta_score_samples,
            report, multilabel_cmatrix
        )
    )
    log_file.write(
        "\naccuracy_classification_score={}\n\nROC={}\n"
        "macro f1 score={}\nmicro f1 score={}\nweighted f1 score={}\nsamples f1 score={}\n"
        "macro precision_recall_fscore_support={}\nmicro precision_recall_fscore_support={}\n"
        "weighted precision_recall_fscore_support={}\nsamples precision_recall_fscore_support={}\n"
        "macro fbeta score={}\nmicro fbeta score={}\nweighted fbeta score={}\nsamples fbeta score={}\n"
        "classification report:\n{}\nmultilabel_confusion_matrix:\n{}\n".format(
            accuracy_classification_score, roc, f1_score_macro, f1_score_micro, f1_score_weighted, f1_score_samples,
            prf_macro, prf_micro, prf_weighted, prf_samples,
            fbeta_score_macro, fbeta_score_micro, fbeta_score_weighted, fbeta_score_samples,
            report, multilabel_cmatrix
        )
    )


def evaluation_top_label(model, x_test, y_test, log_file):
    y_test_bool = np.argmax(y_test, axis=1)
    y_pred = model.predict(x_test, batch_size=64, verbose=0)
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


def perf_grid(predictions, target, label_names, n_thresh=100):
    """Computes the performance table containing target, label names,
    label frequencies, thresholds between 0 and 1, number of tp, fp, fn,
    precision, recall and f-score metrics for each label.

    Args:
        predictions (numpy array): predictions
        target (numpy array): target matrix of shape (BATCH_SIZE, N_LABELS)
        label_names (list of strings): column names in target matrix
        model: model to use for prediction
        n_thresh (int) : number of thresholds to try

    Returns:
        grid (Pandas dataframe): performance table
    """
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
            y_hat = predictions[:, l]
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


def performance_table(y_test_predicted, y_test_target, labels_names, result_path, log_file):
    # Performance table with the first model (macro soft-f1 loss)
    grid = perf_grid(y_test_predicted, y_test_target, labels_names)
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
    # html = styled_table.render()
    # imgkit.from_string(html, os.path.join(result_path, 'performance_styled_table.png'))
    return max_perf, grid


def performance_curves(max_perf, grid, y_test_predicted, label_names, result_path):
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

    for l in top5:
        plt.figure(figsize=(9, 3))
        ax = plt.subplot(1, 2, 1)
        plt.xticks(ticks=np.arange(0, 1.1, 0.1), fontsize=8)
        plt.yticks(fontsize=8)
        plt.title('Probability distribution - Label ' + str(l) + ' (' + label_names[l] + ')\nBCE', fontsize=10)
        plt.xlim(0, 1)
        ax = sns.distplot(y_test_predicted[:, l], bins=30, kde=True, color="b")
        plt.tight_layout()
    plt.savefig(os.path.join(result_path, "performance_curve_2.png"))


def top_words(model, x, y, data_type, labels_names, index_word, batch_size, log_file,
              num_top_trans=100):
    metrics = model.evaluate(x, y, verbose=0)
    for i in range(len(metrics)):
        # print("{}: {}".format(model.metrics_names[i], metrics[i]))
        log_file.write("{}: {}\n".format(model.metrics_names[i], metrics[i]))

    layer_name = 'attention_with_context'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    weighted_seq, attention = intermediate_layer_model.predict(x, batch_size=batch_size, verbose=0)
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
        # print("x part {} shape: {}".format(p, xp.shape))
        weighted_seq, attention = intermediate_layer_model.predict(xp, batch_size=batch_size, verbose=0)
        del weighted_seq
        # print("attention shape: {}".format(attention.shape))
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


def top_words_3(model, x, y, data_type, labels_names, index_word, batch_size, log_file, result_path, layer_name,
                num_top_trans=100, step_size=512):
    word_scores = {}
    for label in labels_names:
        word_scores[label] = {}

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # num_parts = len(y) // 512
    # for p, xp in enumerate(np.array_split(x, num_parts)):
    #     word_seq, attention_scores = intermediate_layer_model.predict(xp, batch_size=batch_size)
    #     for i, sample in enumerate(xp):

    last_index = 0
    for idx in range(0, len(y), step_size):
        # print("x part {} shape: {}".format(p, xp.shape))
        word_seq, attention_scores = intermediate_layer_model.predict(
            x[idx:idx + step_size], batch_size=batch_size, verbose=0
        )
        del word_seq
        # print("attention_scores shape: {}".format(attention_scores.shape))
        # log_file.write("attention_scores shape: {}\n".format(attention_scores.shape))

        for i, sample in enumerate(x[idx:idx + step_size]):
            sample_labels = [labels_names[j] for j, l in enumerate(y[last_index]) if l == 1]
            for k, word_index in enumerate(sample):
                if word_index == 0:
                    continue
                if k == 0:
                    assert index_word[word_index] == "CLS"
                    continue
                word = index_word[word_index]
                for label in sample_labels:
                    score = attention_scores[i][0][k]  # attention score of CLS (first token of each sequence) to word
                    if word in word_scores[label]:
                        word_scores[label][word] += score
                    else:
                        word_scores[label][word] = score
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


def top_words_4(model, x, y, data_type, labels_names, word_index, index_word, batch_size, log_file, result_path,
                layer_name, max_sentence_len, num_top_trans=200, step_size=256):
    def update_attention_scores(model, index_word, word_scores, xp, yp, batch_size):
        word_seq, attention_scores = model.predict(xp, batch_size=batch_size, verbose=0)
        del word_seq
        # print("attention_scores shape: {}".format(attention_scores.shape))
        # log_file.write("attention_scores shape: {}\n".format(attention_scores.shape))

        for i, sample in enumerate(xp):
            if yp[i] == 0:  # skip negative samples
                continue
            cls_token = index_word[sample[0]]
            label = cls_token.split('_')[1]
            for k, windex in enumerate(sample):
                if k == 0 or windex == 0:  # skip cls and padding
                    continue
                word = index_word[windex]
                score = attention_scores[i][0][k]  # attention score of CLS (first token of each sequence) to word
                if word in word_scores[label]:
                    word_scores[label][word] += score
                else:
                    word_scores[label][word] = score

    word_scores = {}
    for label in labels_names:
        word_scores[label] = {}

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    if data_type == "Train":
        for idx in range(0, len(y), step_size):
            update_attention_scores(
                model=intermediate_layer_model,
                index_word=index_word,
                word_scores=word_scores,
                xp=x[idx:idx + step_size],
                yp=y[idx:idx + step_size],
                batch_size=batch_size
            )
    else:
        x_part, y_part = [], []
        for idx, sample in enumerate(x):
            for tag in y[idx]:
                x_part.append([word_index["CLS_" + tag]] + sample)
                y_part.append(1)

            if len(x_part) >= step_size:
                x_part = pad_sequences(x_part, maxlen=max_sentence_len, padding='post', truncating='post')
                y_part = np.array(y_part).reshape(len(y_part), 1)
                update_attention_scores(
                    model=intermediate_layer_model,
                    index_word=index_word,
                    word_scores=word_scores,
                    xp=x_part,
                    yp=y_part,
                    batch_size=batch_size
                )
                x_part, y_part = [], []

        if len(x_part) >= step_size:
            x_part = pad_sequences(x_part, maxlen=max_sentence_len, padding='post', truncating='post')
            y_part = np.array(y_part).reshape(len(y_part), 1)
            update_attention_scores(
                model=intermediate_layer_model,
                index_word=index_word,
                word_scores=word_scores,
                xp=x_part,
                yp=y_part,
                batch_size=batch_size
            )

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


def top_words_5(model, x, y, data_type, labels_names, index_word, batch_size, log_file, result_path,
                layer_name, num_top_trans=200, step_size=256):
    def update_attention_scores(model, index_word, word_scores, xp, batch_size):
        word_seq, attention_scores = model.predict(xp, batch_size=batch_size, verbose=0)
        del word_seq
        # print("attention_scores shape: {}".format(attention_scores.shape))
        # log_file.write("attention_scores shape: {}\n".format(attention_scores.shape))

        for i, sample in enumerate(xp):
            cls_token = index_word[sample[0]]
            label = cls_token.split('_')[1]
            for k, windex in enumerate(sample):
                if k == 0 or windex == 0:  # skip cls and padding
                    continue
                word = index_word[windex]
                score = attention_scores[i][0][k]  # attention score of CLS (first token of each sequence) to word
                if word in word_scores[label]:
                    word_scores[label][word] += score
                else:
                    word_scores[label][word] = score

    word_scores = {}
    for label in labels_names:
        word_scores[label] = {}

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    for idx in range(0, len(y), step_size):
        update_attention_scores(
            model=intermediate_layer_model,
            index_word=index_word,
            word_scores=word_scores,
            xp=x[idx:idx + step_size],
            # yp=y[idx:idx + step_size],
            batch_size=batch_size
        )

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
    def __init__(self, max_sentence_len, num_words, word_vector_dim, label_count):
        self.MAX_SEQUENCE_LENGTH = max_sentence_len
        self.EMBEDDING_DIM = word_vector_dim
        self.LABEL_COUNT = label_count
        self.NUM_WORDS = num_words

    def astm_1(self, train_embedding, embedding_matrix, lstm_dim, dropout_value):
        inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        if embedding_matrix is None:
            embedding = Embedding(input_dim=(self.NUM_WORDS), output_dim=self.EMBEDDING_DIM,
                                  input_length=self.MAX_SEQUENCE_LENGTH, trainable=train_embedding)(inputs)
        else:
            embedding = Embedding(input_dim=(self.NUM_WORDS), output_dim=self.EMBEDDING_DIM, weights=[embedding_matrix],
                                  input_length=self.MAX_SEQUENCE_LENGTH, trainable=train_embedding)(inputs)
        word_seq = Bidirectional(LSTM(lstm_dim, return_sequences=True, kernel_regularizer=regularizers.l2(1e-13)))(
            embedding)
        word_seq = Dropout(dropout_value)(word_seq)
        X, attention_scores = AttentionWithContext(name='attention_with_context')(word_seq)
        X = Dropout(dropout_value)(X)
        outputs = Dense(units=self.LABEL_COUNT, activation="sigmoid")(X)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def astm_2(self, train_embedding, embedding_matrix, lstm_dim, dropout_value):
        inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        if embedding_matrix is None:
            embedding = Embedding(input_dim=(self.NUM_WORDS), output_dim=self.EMBEDDING_DIM,
                                  input_length=self.MAX_SEQUENCE_LENGTH, trainable=train_embedding)(inputs)
        else:
            embedding = Embedding(input_dim=(self.NUM_WORDS), output_dim=self.EMBEDDING_DIM, weights=[embedding_matrix],
                                  input_length=self.MAX_SEQUENCE_LENGTH, trainable=train_embedding)(inputs)
        word_seq = Bidirectional(LSTM(lstm_dim, return_sequences=True, kernel_regularizer=regularizers.l2(1e-13)))(
            embedding)
        word_seq = Dropout(dropout_value)(word_seq)

        X, attention_scores = AttentionWithContextV2(name='attention_with_context_v2', num_classes=self.LABEL_COUNT)(
            word_seq)
        X = Dropout(dropout_value)(X)

        unstacked = Lambda(lambda x: tensorflow.unstack(x, axis=1))(X)
        final_prediction = [Dense(units=1, activation="sigmoid")(xr) for xr in unstacked]
        outputs = Lambda(lambda x: K.squeeze(K.stack(x, axis=1), axis=-1))(final_prediction)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def sastm(self, train_embedding, embedding_matrix, lstm_dim, dropout_value):
        inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedding = Embedding(input_dim=(self.NUM_WORDS), output_dim=self.EMBEDDING_DIM, weights=[embedding_matrix],
                              input_length=self.MAX_SEQUENCE_LENGTH, trainable=train_embedding)(inputs)
        w_queries = Embedding(input_dim=(self.NUM_WORDS), output_dim=self.EMBEDDING_DIM,
                              input_length=self.MAX_SEQUENCE_LENGTH, trainable=train_embedding)(inputs)
        w_keys = Embedding(input_dim=(self.NUM_WORDS), output_dim=self.EMBEDDING_DIM,
                           input_length=self.MAX_SEQUENCE_LENGTH, trainable=train_embedding)(inputs)
        # print("embedding.shape:", embedding.shape)  # (None, 256, 50)
        # print("w_queries.shape:", w_queries.shape)  # (None, 256, 50)
        # print("w_keys.shape:", w_keys.shape)  # (None, 256, 50)

        word_seq = Bidirectional(LSTM(lstm_dim, return_sequences=True, kernel_regularizer=regularizers.l2(1e-13)))(
            embedding)
        # print("word_seq.shape:", word_seq.shape)  # (None, 256, 256)
        word_seq = Dropout(dropout_value)(word_seq)
        # print("word_seq.shape:", word_seq.shape)  # (None, 256, 256)
        # print(type(word_seq)) -> <class 'keras.engine.keras_tensor.KerasTensor'>

        word_seq_2, attention_scores = AttentionWithContextV3(
            name='attention_with_context_v3',
            num_words=self.NUM_WORDS,
            embedding_dim=self.EMBEDDING_DIM,
            max_sentence_len=self.MAX_SEQUENCE_LENGTH
        )([w_queries, w_keys, word_seq])
        # print("word_seq_2.shape:", word_seq_2.shape)  # (None, 256, 256)
        # print("attention_scores.shape:", attention_scores.shape)  # (None, 256, 256)
        word_seq_2 = Dropout(dropout_value)(word_seq_2)
        # print("word_seq_2[:, 0, :].shape:", word_seq_2[:, 0, :].shape)  # (None, 256)

        outputs = Dense(units=self.LABEL_COUNT, activation="sigmoid")(word_seq_2[:, 0, :])
        # print("outputs.shape:", outputs.shape)  # (None, 100)
        model = Model(inputs=inputs, outputs=outputs)
        return model
