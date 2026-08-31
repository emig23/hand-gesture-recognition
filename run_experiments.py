from dataset import download_dataset
from train import train
from evaluation import evaluate_model

if __name__ == '__main__':
    download_dataset()

    train(architecture='mobilenet_v2', num_epochs=5)
    evaluate_model(architecture='mobilenet_v2')

    train(architecture='mobilenet_v3_small', num_epochs=5)
    evaluate_model(architecture='mobilenet_v3_small')

    train(architecture='efficientnet_b0', num_epochs=5)
    evaluate_model(architecture='efficientnet_b0')