# extract information from folder `./output/kd_bench/nb201_kd` and `./output/kd_bench/nb101_kd`
import hashlib
import os
import pickle


def extract_nb101_folder():
    # hash -> acc
    nb101_kd_dict = {}
    folder_path = './output/kd_bench/nb101_kd'
    # traverse folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.log'):
                # get file name, also is md5 in nb101
                md5 = file.split('.')[0]
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # get last line and judge if it's started with 'best'
                    last_line = lines[-1]
                    if last_line.startswith('best'):
                        # extract 62.6100 from "best accuracy: tensor(62.6100, device='cuda:0')" using grep
                        acc = last_line.split('tensor(')[1].split(',')[0]

                        # print(md5, acc)
                        nb101_kd_dict[md5] = acc
    return nb101_kd_dict


def extract_nb201_folder():
    nb201_kd_dict = {}
    folder_path = './output/kd_bench/nb201_kd'
    # traverse folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.log'):
                # get file name, also is idx in nb201
                idx = file.split('.')[0].split('_')[-1]
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # get last line and judge if it's started with 'best'
                    last_line = lines[-1]
                    if last_line.startswith('best'):
                        # extract 62.6100 from "best accuracy: tensor(62.6100, device='cuda:0')" using grep
                        acc = last_line.split('tensor(')[1].split(',')[0]

                        # print(idx, acc)
                        nb201_kd_dict[idx] = acc
    return nb201_kd_dict


def build_nb101_kd_file():
    nb101_kd_dict = extract_nb101_folder()

    # save to pickle

    with open('./data/nb101_kd_dict.pkl', 'wb') as f:
        pickle.dump(nb101_kd_dict, f)

    # compute md5 for .pkl and add to the name of this file
    with open('./data/nb101_kd_dict.pkl', 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    os.rename('./data/nb101_kd_dict.pkl', f'./data/nb101_kd_dict_{md5}.pkl')


def build_nb201_kd_file():

    nb201_kd_dict = extract_nb201_folder()

    # save to pickle
    with open('./data/nb201_kd_dict.pkl', 'wb') as f:
        pickle.dump(nb201_kd_dict, f)

    # compute md5 for .pkl and add to the name of this file
    with open('./data/nb201_kd_dict.pkl', 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    os.rename('./data/nb201_kd_dict.pkl', f'./data/nb201_kd_dict_{md5}.pkl')


# build_nb201_kd_file()
