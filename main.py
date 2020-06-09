from src.k_dae import KDae
from src.utils import load_data, cluster_performance
import numpy as np
import logging


def config_logger():
    logging.basicConfig(filename='outputs\\k_dae.log', level=logging.DEBUG, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
    dataset_name = 'mnist' 
    config_logger()
    logging.debug('Start running dataset name - {}'.format(dataset_name))
    x_train, y_train = load_data(dataset_name)
    n_cluster = len(np.unique(y_train))
    model = KDae(number_cluster=n_cluster, k_dae_epoch=1, epoch_ae=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    cluster_performance(y_pred, y_train)


