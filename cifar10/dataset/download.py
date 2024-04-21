from dataset import download_dataset
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_path', type=str, default='/data')
    args = parser.parse_args()

    download_dataset(args.dataset, args.data_path)