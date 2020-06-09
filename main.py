from src.k_dae import KDae
from src.utils import load_data


if __name__ == '__main__':
    dataset_name = 'mnist'
    x_train, y_train = load_data(dataset_name)
    model = KDae()

