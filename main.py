from src.k_dae import KDae
from src.utils import load_data, cluster_performance
import numpy as np
import logging
from pathlib import Path
import os
import argparse


def config_logger(log_path='k_dae.log'):
    logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-sd', '--save_dir',
                        type=str,
                        default='save',
                        help='path to save ')

    parser.add_argument('-dn', '--dataset_name',
                        choices=['mnist', 'fashion', 'usps'],
                        default='mnist',
                        help='dataset name [mnist,fashion,usps] ')

    FLAGS, unparsed = parser.parse_known_args()
    save_dir_name = FLAGS.save_dir
    dataset_name = FLAGS.dataset_name
    path_dir = Path(os.path.join(save_dir_name, dataset_name))
    path_dir.mkdir(exist_ok=True)
    log_path = os.path.join(save_dir_name, dataset_name, 'k_dae.log')
    config_logger(log_path)
    logging.debug('Start running dataset name - {}'.format(dataset_name))

    x_train, y_train = load_data(dataset_name)
    n_cluster = len(np.unique(y_train))
    model = KDae(number_cluster=n_cluster, k_dae_epoch=40, epoch_ae=10, initial_epoch=80, dataset_name=dataset_name)
    model.fit(x_train, y_train, dataset_name=dataset_name)
    y_pred = model.predict(x_train)
    cluster_performance(y_pred, y_train)


