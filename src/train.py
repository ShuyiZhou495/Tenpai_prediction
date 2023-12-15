import torch
from generate_train_data import generate_train_test_local

if __name__ == "__main__":
    print(torch.cuda.is_available())
    # x_train, x_test, y_train, y_test = generate_train_test()
    x_train, x_test, y_train, y_test = generate_train_test_local()
    print(x_train.shape)
    print(y_train.shape)