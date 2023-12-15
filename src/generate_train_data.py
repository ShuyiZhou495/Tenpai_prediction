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


def generate_train_data(file_name):
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
    # test_list = load_data("./data/sample.txt")
    # test_list = load_data("./data/totuhaihu.txt")
    data = load_data(file_path)
    # richi_data = richi_filter(test_list)
    print("data size:", len(data))
    # print("Generate fake haifu:")
    # fake_haifu_number = 5 # Too big will cause Out of Memory
    # fake_haifu_number = 0
    # new_richi_data = []
    # for haifu in richi_data:
    #     new_richi_data.append(haifu)
        # new_richi_data += generate_fake_haifu(haifu, fake_haifu_number)
        # if len(new_richi_data) % 1000 == 0:
        #     print(len(new_richi_data), round(time.time() - time_start, 2))
    # print(len(data), round(time.time() - time_start, 2))    

    print("Embed to vectors...")
    for haifu in data:
        inputs, outputs = parse_haifu(haifu)
        for inp, out in zip(inputs, outputs):
            if len(out) > 0:
                x_data.append(inp)
                y_data.append(out)
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

    file_path = "../data/" + file_name + ".txt"
    test_list = load_data(file_path)
    richi_data = richi_filter(test_list)

    for haifu in richi_data:
        inp, chanfon, jikaze, dora_list, tenpai_result, sute = parse_haifu(haifu)
        for each_inp in inp:
            x = []
            player = each_inp[0]
            for action in each_inp.split(" "):
                if action != "":
                    if not(player != action[0] and action[1] == "G"):
                        x.append(action_to_vector(action, player, chanfon, jikaze, dora_list))
            x_data.append(np.array(x, dtype=np.int8))
            y_data.append(tenpai_result)
            assistant_data.append([player, sute])
    x_data_numpy = np.array(x_data, dtype=object)
    y_data_numpy = np.array(y_data)

    return x_data_numpy, y_data_numpy, assistant_data


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


def shuffle_split(x_data, y_data):
    # x_data, y_data = shuffle(x_data, y_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
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
    x_train, x_test, y_train, y_test = shuffle_split(x_data, y_data)
    return x_train, x_test, y_train, y_test


local_place = "./data/"
def save_train_data():
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
        np.save(f, x_len)
        np.save(f, y_len)
    print(x_data.shape)
    # np.save("../model/x_data.npy", x_data)
    # np.save("../model/y_data.npy", y_data)
    np.save(local_place + "x_data.npy", x_data)
    np.save(local_place + "y_data.npy", y_data)
    print("Train data generate and save on local place")


def generate_train_test_local():
    x_data = np.load(local_place + "x_data.npy").astype(np.int8)
    y_data = np.load(local_place + "y_data.npy").astype(np.int8)
    x_train, x_test, y_train, y_test = shuffle_split(x_data, y_data)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    save_train_data()

