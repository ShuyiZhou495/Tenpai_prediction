import numpy as np
from tenpai import tenpai


# man pin sou z1234567 東南西北白発中

def change_tile_to_number(tile):
    if len(tile) > 1:
        number = int(tile[0])
        k = 0
        if tile[1] == "p":
            k = 1
        if tile[1] == "s":
            k = 2
        return k * 9 + number - 1
    z_list = "東南西北白発中"
    return z_list.find(tile) + 27


def parse_start_hand(tiles):
    tiles_34 = [0] * 34
    i = 0
    while i < len(tiles):
        if '1' <= tiles[i] <= '9':
            now_tile = tiles[i:i + 2]
            tiles_34[change_tile_to_number(now_tile)] += 1
            i += 2
        else:
            now_tile = tiles[i]
            tiles_34[change_tile_to_number(now_tile)] += 1
            i += 1
    return tiles_34


def parse_haifu(haifu):
    # haifu has 7 lines
    dora_position = haifu[4].find("[裏ドラ]")
    dora = haifu[4][5: dora_position - 1]
    i = 0
    dora_list = []
    while i < len(dora):
        if '1' <= dora[i] <= '9':
            now_tile = dora[i:i + 2]
            dora_list.append(dora[i:i + 2])
            i += 2
        else:
            now_tile = dora[i]
            dora_list.append(dora[i])
            i += 1

    # richi_position = haifu[5].find("R")
    # first_richi_player = int(haifu[5][richi_position - 1])
    # haifu[5] = haifu[5][0:richi_position + 6]

    # # search for kan
    # kan_index = -1
    # kan_list = {1:[], 2:[], 3:[], 4:[]}
    # while "K" in haifu[5][kan_index + 1:]:
    #     kan_index += 1 + haifu[5][kan_index + 1:].index("K")
    #     kan_player = int(haifu[5][kan_index-1]) - 1
    #     kan_hai = haifu[5][kan_index + 1: kan_index + 3].replace(" ", "")
    #     kan_list[int(kan_player)].append(change_tile_to_number(kan_hai))

    # chanfon = '東'
    chanfon = haifu[6].strip()[0]

    jikaze = [haifu[i][2] for i in range(4)]

    ############## Generate input
    # write the start hand as actions
    inputs = [] * 4
    kan = 0
    for player in range(4):
        inp = []
        start_hand = haifu[player][4:]
        i = 0
        prefix = str(player+1) + "G"
        while i < len(start_hand):
            if '1' <= start_hand[i] <= '9':
                now_tile = start_hand[i:i + 2]
                i += 2
            else:
                now_tile = start_hand[i]
                i += 1
            inp.append(
                action_to_vector(prefix + now_tile, player + 1, chanfon, jikaze[player], dora_list[:kan+1])
                )
        inputs.append(inp)

    ############  Generate input of actions and corresponding output
    actions = [a for a in haifu[5].split(" ") if a != ""]
    if actions[-1][-1] == "A":
        actions = actions[:-2]
    start_hand = [parse_start_hand(haifu[i][4:]) for i in range(4)]
    sute = [[False] * 34 for _ in range(4)]
    houju = np.array([[0] * 34 for _ in range(4)], dtype=np.uint8)
    outputs = [[], [], [], []]
    last_sute = None
    K = False
    for action in actions:
        ##### change hand #######
        player = int(action[0]) - 1
        if action[1] == "G":
            now_tile = action[2:]
            start_hand[player][change_tile_to_number(now_tile)] += 1
        elif action[1] == "N" or action[1] == "C":
            start_hand[player][last_sute] += 1
        elif action[1] == "D" or action[1] == "d":
            now_tile = action[2:]
            tile_number = change_tile_to_number(now_tile)
            last_sute = tile_number
            start_hand[player][tile_number] -= 1
            sute[player][tile_number] = True

        if action[1] == "K":
            K = True

        ##### update houjuu hai ######
        if action[1] == "D" or action[1] == "d":
            tenpai(houju[player], start_hand[player], sute[player])

        ###### update input ############
        a2v = action_to_vector(action, player + 1, chanfon, jikaze[player], dora_list[:kan+1])
        curr_houju = houju.sum(0, dtype=np.uint8)
        if action[1] == "G":
            inputs[player].append(a2v)
            outputs[player].append((curr_houju - houju[player]).astype(bool))
        else:
            for i in range(4):
                inputs[i].append(a2v)
                outputs[i].append((curr_houju - houju[i]).astype(bool))
        if K and (action[1] == "D" or action[1] == "d"):
            kan += 1
            K = False

    return [np.array(inp, dtype=bool) for inp in inputs], [np.array(out, dtype=bool) for out in outputs]


def show_sutehai(haifu, player):
    sutehai_list = []
    for action in haifu[5].split(" "):
        if action[0] == str(player) and (action[1] == "D" or action[1] == "d"):
            now_tile = action[2:]
            sutehai_list.append(now_tile)
    return sutehai_list


def show_richi_player_sutehai(haifu):
    richi_position = haifu[5].find("R")
    first_richi_player = int(haifu[5][richi_position - 1])
    return show_sutehai(haifu, first_richi_player)


def show_sutehai_list(haifu, player):
    sutehai_tiles = show_sutehai(haifu, player)
    sutehai_list = [False] * 34
    for sutehai in sutehai_tiles:
        sutehai_list[change_tile_to_number(sutehai)] = True
    return sutehai_list


def show_richi_player_sutehai_list(haifu):
    richi_position = haifu[5].find("R")
    first_richi_player = int(haifu[5][richi_position - 1])
    return show_sutehai_list(haifu, first_richi_player)


def action_to_vector(action, player, chanfon, jikaze, dora_list):
    # action like: 1G1m, 2N, 3R, 4d5p
    vector = [0] * 52
    vector[int(action[0]) - 1] = 1  # player number 0-3
    ch = action[1]
    if ch == 'G':
        act = 0
    elif ch == 'd':
        act = 1
    elif ch == 'D':
        act = 2
    elif ch == 'N':
        act = 3
    elif ch == 'C':
        act = 4
    elif ch == 'K':
        act = 5
    elif ch == 'R':
        act = 6
    vector[act + 4] = 1  # act 4-10

    tile = action[2:]
    if len(tile) != 0 and not (ch == 'G' and str(player) != action[0]):

        tile_num = change_tile_to_number(tile)
        vector[tile_num + 11] = 1  # tile 11-45
        if tile == chanfon:
            vector[46] = 1  # chanfon 46
        if tile == jikaze:
            vector[47] = 1  # jikaze 47

        dora_counter = 0
        for dora in dora_list:
            dora_counter += 1
            if tile[0:2] == dora and dora_counter <= 4:
                vector[47 + dora_counter] = 1  # uradora 48-51

        if len(tile) == 4:  # Deal with Chi
            tile_num_2 = change_tile_to_number(tile[2:])
            vector[tile_num_2 + 11] = 1  # tile 11-45

            dora_counter = 0
            for dora in dora_list:
                dora_counter += 1
                if tile[2:] == dora and dora_counter <= 4:
                    vector[47 + dora_counter] = 1  # uradora 48-51

    return np.array(vector, dtype=bool)


def check_haifu(st):
    haifu_checker = ["[1東", "[1南", "[1西", "[1北"]
    for checker in haifu_checker:
        if st.find(checker) != -1:
            return 1
    return -1


def load_data(file_name):
    # For haifu from 東風荘
    haifu_list = []

    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        use_line = 0
        while use_line != len(lines):
            if check_haifu(lines[use_line]) == -1:
                use_line += 1
            else:
                haifu_now = lines[use_line: use_line + 5]
                if haifu_now[3][:2] == "[4":
                    chanfon_richi_honba = lines[use_line - 2]
                    use_line = use_line + 5
                    sute = ""
                    star = lines[use_line].find("*")
                    while star != -1:
                        sute += lines[use_line][star + 1:]
                        use_line += 1
                        star = lines[use_line].find("*")
                    haifu_now.append(sute.strip())
                    haifu_now.append(chanfon_richi_honba)
                    # Get Chanfon, richi, honba
                    haifu_list.append(haifu_now)
                else:
                    use_line += 4
                    while lines[use_line].find("*") != -1:
                        use_line += 1

    return haifu_list


def richi_filter(haifu_list):
    return [haifu for haifu in haifu_list if haifu[5].find("R") != -1]


if __name__ == "__main__":
    test_list = load_data("../data/sample.txt")
    # test_list = load_data("../data/totuhaihu.txt")
    richi_data = richi_filter(test_list)
    print(richi_data[0])
    # Test parse_haifu
    print(parse_haifu(richi_data[0]))

    # Test action_to_vector
    print(action_to_vector("1d1m", 1, '東', '西', ['1m']))
    print(action_to_vector("1d東", 1, '東', '西', ['1m']))
    print(action_to_vector("1G1m", 1, '東', '西', ['1m']))
    print(action_to_vector("2G1m", 1, '東', '西', ['1m']))
    print(action_to_vector("1C2s4s", 2, '東', '西', ['2s']))
    print(action_to_vector("1C2s3s", 2, '東', '西', ['3s', '2s']))

    # Test show_sutehai and show_richi_player_sutehai
    print(show_sutehai(richi_data[0], 1))
    print(show_richi_player_sutehai(richi_data[0]))
