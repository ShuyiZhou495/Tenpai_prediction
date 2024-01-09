import numpy as np
import time
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from generate_fake_haifu import generate_fake_haifu
from haifu_parser import load_data
from haifu_parser import richi_filter
from haifu_parser import parse_haifu
from haifu_parser import action_to_vector
import torch


def generate_train_data(file_name, mjs=False):
    # 600 haifus/1s
    print("Haifu name:" + file_name)

    if os.path.exists("./data/" + file_name + ".npy"):
        with open("./data/" + file_name + ".npy", "rb") as f:
            x_data_numpy = np.load(f, allow_pickle=True)
            y_data_numpy = np.load(f, allow_pickle=True)
        return x_data_numpy, y_data_numpy
    time_start = time.time()
    x_data = []
    y_data = []

    file_path = "./data/" + file_name + ".txt"
    data = load_data(file_path)
    print("data size:", len(data))

    print("Embed to vectors...")
    for haifu in data:
        inputs, outputs = parse_haifu(haifu)
        for inp, out in zip(inputs, outputs):
            if len(out) > 0:
                x_data.append(inp)
                y_data.append(out)
                if mjs:
                    break
            else:
                continue
    x_data_numpy = np.array(x_data, dtype=object)
    y_data_numpy = np.array(y_data, dtype=object)

    time_end = time.time()
    print('Generate train data cost %s seconds.' %
          round((time_end - time_start), 2))
    print('Haifu number: %s' % y_data_numpy.shape[0])
    with open("./data/" + file_name + ".npy", "wb") as f:
        np.save(f, x_data_numpy)
        np.save(f, y_data_numpy)
    return x_data_numpy, y_data_numpy


def generate_test_data(file_name):
    x_data = []
    y_data = []
    assistant_data = []

    file_path = "./data/" + file_name + ".txt"
    test_list = load_data(file_path)

    for haifu in test_list:
        inputs, outputs = parse_haifu(haifu)
        for inp, out in zip(inputs, outputs):
            if len(inp) > 53:
                index = np.random.randint(51, len(inp)-2)
                x_data.append(inp[:index])
                y_data.append(out[index-14])
            else:
                continue
    x_data_numpy = np.array(x_data, dtype=object)
    y_data_numpy = np.column_stack([np.array(y_data), np.zeros((len(y_data), 1))])
    y_data_numpy[:, 34] = ~y_data_numpy[:, :34].any(1)
    return x_data_numpy, y_data_numpy


def pad_x(x_data):
    x_len = []
    for i in range(len(x_data)):
        x_len.append(len(x_data[i]))
    max_x_len = max(x_len)
    x_data_ret = np.zeros((len(x_data), max_x_len, x_data[0].shape[-1]), dtype=np.int8)

    for i in range(len(x_data)):
        zeros = np.zeros((max_x_len - x_data[i].shape[0], x_data[0].shape[-1]), dtype=np.int8)
        x_data_ret[i] = np.concatenate((zeros, x_data[i]), axis=0)
    return x_data_ret, x_len

permutes = [
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0]
            ]

def shift_bfz(x, y, code=0):
    res_x = x.copy()
    res_y = y.copy()
    p = permutes[code]
    res_x[:, 42] = x[:, p[0] + 42]
    res_x[:, 43] = x[:, p[1] + 42]
    res_x[:, 44] = x[:, p[2] + 42]
    res_y[31] = y[p[0] + 31]
    res_y[32] = y[p[1] + 31]
    res_y[33] = y[p[2] + 31]
    return res_x, res_y

def shift_mps(x, y, code=0):
    res_x = x.copy()
    res_y = y.copy()
    p = permutes[code]
    res_x[:, 11:20] = x[:, p[0]*9+11:p[0]*9+20]
    res_x[:, 20:29] = x[:, p[1]*9+11:p[1]*9+20]
    res_x[:, 29:38] = x[:, p[2]*9+11:p[2]*9+20]
    res_y[0:9] = y[p[0]*9:p[0]*9+9]
    res_y[9:18] = y[p[1]*9:p[1]*9+9]
    res_y[18:27] = y[p[2]*9:p[2]*9+9]
    return res_x, res_y

def shift_19(x, y, code=0):
    res_x = x.copy()
    res_y = y.copy()
    res_x[:, 11:20] = res_x[:, 19:10:-1]
    res_x[:, 20:29] = res_x[:, 28:19:-1]
    res_x[:, 29:38] = res_x[:, 37:28:-1]
    res_y[0:9] = res_y[0:9][::-1]
    res_y[9:18] = res_y[17:8:-1]
    res_y[18:27] = res_y[26:17:-1]
    return res_x, res_y

def augment(x, y):
    dice = np.random.randint(0, 12)
    if dice < 5:
        return shift_bfz(x, y, dice)
    dice -= 5
    if dice < 5:
        return shift_mps(x, y, dice)
    return shift_19(x, y)

def decide_num(l):
    if l <= 53:
        return []
    randint = np.random.randint(0, 1 + min(l - 53, 8))
    left = max(l - 53 - randint, 0)
    len = (left//8) + 1
    return (np.linspace(51 + randint, l-3, len)+0.5).astype(int)

def choose(x_data, y_data, n=4):
    x_len = []
    x_len_flat = []

    aug_len = []
    aug_len_flat = []

    max_len = 119

    for i in range(len(x_data)):
        indices = decide_num(len(x_data[i]))
        x_len.append(indices)
        x_len_flat += list(indices)
        if len(indices) == 0:
            aug_len.append([])
            continue
        max_len = max(max_len, max(indices))
        aug_n = max(n - len(indices), 0)
        aug_indices = [indices[np.random.randint(0, len(indices))] for _ in range(aug_n)]
        aug_len.append(aug_indices)
        aug_len_flat += aug_indices
            
    x_data_ret_o = np.zeros((len(x_len_flat), max_len, 52), dtype=bool)
    y_data_ret_o = np.zeros((len(x_len_flat), 35), dtype=bool)
    curr = 0
    for xl, x, y in zip(x_len, x_data, y_data):
        for xll in xl:
            x_data_ret_o[curr][-xll:] = x[:xll]
            y_data_ret_o[curr][:34] = y[xll - 14]
            y_data_ret_o[curr][34] = ~y[xll - 14].any()
            curr += 1
    x_data_ret_aug = np.zeros((len(aug_len_flat), max_len, 52), dtype=bool)
    y_data_ret_aug = np.zeros((len(aug_len_flat), 35), dtype=bool)
    curr = 0
    for aug, x, y in zip(aug_len, x_data, y_data):
        for a in aug:
            augx, augy = augment(x[:a], y[a - 14])
            x_data_ret_aug[curr][-a:] = augx
            y_data_ret_aug[curr][:34] = augy
            y_data_ret_aug[curr][34] = ~augy.any()
            curr += 1
    return x_data_ret_o, y_data_ret_o, x_len_flat, x_data_ret_aug, y_data_ret_aug, aug_len_flat

def shuffle_split(x_data, y_data, x_len, y_len):
    # x_data, y_data = shuffle(x_data, y_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, x_len, y_data,
                                                        test_size=0.2)
    
    return x_train, x_test, y_train, y_test


def generate_train_test():
    
    x_data_1, y_data_1 = generate_train_data("kintaro")
    x_data_2, y_data_2 = generate_train_data("mjscore")
    x_data_3, y_data_3 = generate_train_data("momosescore")
    x_data_4, y_data_4 = generate_train_data("score")
    x_data_0, y_data_0 = generate_train_data("totuhaihu")
    
    x_data = np.concatenate((x_data_0, x_data_1, x_data_2, x_data_3, x_data_4), axis=0)
    y_data = np.concatenate((y_data_0, y_data_1, y_data_2, y_data_3, y_data_4), axis=0)
    
    x_data, x_len = pad_x(x_data)
    y_data, y_len = pad_x(y_data)
    with open("./data/sizes.npy", "wb") as f:
        np.save(x_len)
        np.save(y_len)
    print(x_data.shape)
    with open("./data/sizes.npy", "rb") as f:
        x_len = np.load(f, allow_pickle=True)
        y_len = np.load(f, allow_pickle=True)
    x_train, x_test, x_len_train, x_len_test, y_train, y_test = train_test_split(x_data, x_len, y_data,
                                                        test_size=0.2)
    return torch.tensor(x_train, dtype=torch.float16), \
            torch.tensor(x_test, dtype=torch.float16), \
            torch.tensor(x_len_train, dtype=torch.long), \
            torch.tensor(x_len_test, dtype=torch.long), \
            torch.tensor(y_train, dtype=torch.float16), \
            torch.tensor(y_test, dtype=torch.float16)


local_place = "./data/"
def save_train_data():
    x_data_1, y_data_1 = generate_train_data("kintaro")
    x_data_2, y_data_2 = generate_train_data("mjscore")
    x_data_3, y_data_3 = generate_train_data("momosescore")
    x_data_4, y_data_4 = generate_train_data("score")
    x_data_0, y_data_0 = generate_train_data("totuhaihu")
    
    x_data = np.concatenate((x_data_0, x_data_1, x_data_2, x_data_3, x_data_4), axis=0)
    y_data = np.concatenate((y_data_0, y_data_1, y_data_2, y_data_3, y_data_4), axis=0)
    x_data, y_data, x_len, x_data_aug, y_data_aug, x_len_aug = choose(x_data, y_data)
    # x_data, x_len = pad_x(x_data)
    # y_data, y_len = pad_x(y_data)
    with open("./data/sizes.npy", "wb") as f:
        np.save(f, x_len)
        np.save(f, x_len_aug)
    print(x_data.shape)
    # np.save("../model/x_data.npy", x_data)
    # np.save("../model/y_data.npy", y_data)
    np.save(local_place + "x_data.npy", x_data)
    np.save(local_place + "y_data.npy", y_data)
    np.save(local_place + "x_data_aug.npy", x_data_aug)
    np.save(local_place + "y_data_aug.npy", y_data_aug)
    print("Train data generate and save on local place")

def save_train_data_mjsoul():
    x_data_1, y_data_1 = generate_train_data("sue", True)
    x_data_2, y_data_2 = generate_train_data("xsx", True)
    
    x_data = np.concatenate((x_data_1, x_data_2), axis=0)
    y_data = np.concatenate((y_data_1, y_data_2), axis=0)
    x_data, y_data, x_len, x_data_aug, y_data_aug, x_len_aug = choose(x_data, y_data, n=8)
    # x_data, x_len = pad_x(x_data)
    # y_data, y_len = pad_x(y_data)
    with open("./data/sizes_mjsoul.npy", "wb") as f:
        np.save(f, x_len)
        np.save(f, x_len_aug)
    print(x_data.shape)
    # np.save("../model/x_data.npy", x_data)
    # np.save("../model/y_data.npy", y_data)
    np.save(local_place + "x_data_mjsoul.npy", x_data)
    np.save(local_place + "y_data_mjsoul.npy", y_data)
    np.save(local_place + "x_data_aug_mjsoul.npy", x_data_aug)
    np.save(local_place + "y_data_aug_mjsoul.npy", y_data_aug)
    print("Train data generate and save on local place")

def generate_train_test_local():
    x_data = np.load(local_place + "x_data.npy")
    y_data = np.load(local_place + "y_data.npy")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        test_size=0.2)
    x_data_aug = np.load(local_place + "x_data_aug.npy")
    y_data_aug = np.load(local_place + "y_data_aug.npy")
    x_train_aug, _, y_train_aug, _ = train_test_split(x_data_aug, y_data_aug,
                                                        test_size=0.01)
    # return torch.tensor(x_train, dtype=torch.float32), \
    #         torch.tensor(x_test, dtype=torch.float32), \
    #         torch.tensor(x_len_train, dtype=torch.long), \
    #         torch.tensor(x_len_test, dtype=torch.long), \
    #         torch.tensor(y_train, dtype=torch.float32), \
    #         torch.tensor(y_test, dtype=torch.float32)
    return np.concatenate((x_train, x_train_aug)), x_test, np.concatenate((y_train, y_train_aug)), y_test

def generate_train_test_local_mjsoul():
    x_data = np.load(local_place + "x_data_mjsoul.npy")
    y_data = np.load(local_place + "y_data_mjsoul.npy")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        test_size=0.01)
    np.save(local_place + "x_test_mjsoul.npy", x_test)
    np.save(local_place + "y_test_mjsoul.npy", y_test)
    # x_data_aug = np.load(local_place + "x_data_aug_mjsoul.npy")
    # y_data_aug = np.load(local_place + "y_data_aug_mjsoul.npy")
    # x_train_aug = x_data_aug[:-100]
    # y_train_aug = y_data_aug[:-100]

    # x_data = np.concatenate((x_train, x_train_aug))
    # y_data = np.concatenate((y_train, y_train_aug))
    # x_train, _, y_train, _ = train_test_split(x_data, y_data, test_size=0.01)
    # return torch.tensor(x_train, dtype=torch.float32), \
    #         torch.tensor(x_test, dtype=torch.float32), \
    #         torch.tensor(x_len_train, dtype=torch.long), \
    #         torch.tensor(x_len_test, dtype=torch.long), \
    #         torch.tensor(y_train, dtype=torch.float32), \
    #         torch.tensor(y_test, dtype=torch.float32)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    save_train_data_mjsoul()

