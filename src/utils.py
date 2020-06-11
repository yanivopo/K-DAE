from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import logging
import tensorflow as tf


def acc(y_true, y_pred):

    """ Calculate clustering accuracy

    Require scikit-learn installed

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def k_means(x_train, n_class, n_init=100):
    """ compute k_means algorithm

    use scikit-learn to compute k-means

    :param x_train: data points
    :param n_class: number of clusters
    :param n_init: The number of different initialization
    :return: k_means model
    """
    k_mean = KMeans(n_clusters=n_class, n_init=n_init)
    km_model = k_mean.fit(x_train)
    return km_model


def cluster_performance(y_pred, y_train, label='kmean'):
    """ calculate performance of clustering


    :param y_pred: Predication vector
    :param y_train: Ground truth vector
    :param label: Method name
    :return: NMI, ACC, ARI
    """
    k_means_nmi = metrics.normalized_mutual_info_score(y_train, y_pred)
    k_means_ari = metrics.adjusted_rand_score(y_train, y_pred)
    k_means_acc = acc(y_train, y_pred)
    # print('{} NMI is {}'.format(label, k_means_nmi))
    # print('{} ARI is {}'.format(label, k_means_ari))
    # print('{} Acc is {}'.format(label, k_means_acc))
    logging.info("NMI - {:0.2f},ARI - {:0.2f},ACC - {:0.2f}".format(k_means_nmi, k_means_ari, k_means_acc))
    return k_means_nmi, k_means_acc, k_means_ari


def load_data(data_name):
    if data_name == 'mnist' or data_name == 'fashion':
        if data_name == 'mnist':
            data = tf.keras.datasets.mnist
        else:
            data = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = data.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        x_train = np.concatenate((x_train, x_test), axis=0)
        y_train = np.concatenate((y_train, y_test))
        return [x_train, y_train]

    elif data_name == 'usps':
        x_train = np.load('data/usps/x_usps.npy')
        y_train = np.load('data/usps/y_usps.npy')
        x_train = (x_train + 1) / 2     #change the value to be between 0 to 1
        x_train = np.array(x_train)
        return [x_train, y_train]