from keras.models import load_model
import numpy as np
from generate_train_data import generate_test_data
import torch

def acc(y_true, y_pred):
    y_true = torch.tensor(y_true, device=y_pred.device, dtype=torch.float32)
    y_true = torch.nn.functional.normalize(y_true, dim=-1)
    y_pred = torch.nn.functional.normalize(y_pred, dim=-1)
    return torch.sum(y_true * y_pred, axis=-1)

model = load_model("./model2/tenpai_mjsoul.keras", custom_objects={'acc': acc})
# x_data, y_data = generate_test_data('test')
x_data = np.load("./data/x_test_mjsoul.npy")
y_data = np.load("./data/y_test_mjsoul.npy")

pad_dim = int(model.input[0].shape[1])


def pad_sample(x, pad_dim):
    zeros = np.zeros((pad_dim - x.shape[0], 52))
    return np.concatenate((zeros, x), axis=0).reshape(1, pad_dim, 52)


def predict(x, sute):
    p = model.predict(pad_sample(x, pad_dim))[0]
    for i in range(len(sute)):
        if sute[i]:
            p[i] = 0
    return p / sum(p)


def number_to_tile(num):
    if num <= 26:
        k = num // 9
        numb = num % 9 + 1
        follow = "m"
        if k == 1:
            follow = "p"
        if k == 2:
            follow = "s"
        return str(numb) + follow
    z_list = "東南西北白発中"
    return z_list[num - 27]


def print_tenpai(y):
    print("Richi player tenpai:",
          [number_to_tile(num) for num in range(34) if y[num] == 1])


def predict_with_assistant(x, y, assist):
    player, sute = assist
    result = dict()
    threshold = 0.01

    prob = predict(x, sute)
    print_tenpai(y)
    print("Player " + str(player) + ": ")
    for i in range(34):
        if prob[i] > threshold:
            result[number_to_tile(i)] = prob[i]
    print(result)


def predict_by_order(i, assistant_data):
    predict_with_assistant(x_data[i], y_data[i], assistant_data[i])

def predict_with(x, y):
    result = dict()
    threshold = 0.1

    prob = model.predict(pad_sample(x, pad_dim))[0]
    prob /= prob.sum()
    print_tenpai(y[:34])
    # print("Player " + str(player) + ": ")
    for i in range(34):
        if prob[i] > threshold:
            result[number_to_tile(i)] = prob[i]
    if prob[-1] > threshold:
        result["safe"] = prob[-1]
    print(result)

if __name__ == "__main__":
    cnt = x_data.shape[0]
    for i in range(cnt):
        predict_with(x_data[i], y_data[i])

