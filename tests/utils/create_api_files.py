import hashlib
import os
import pickle

file_cls = './data/s0-gt-cls.txt'
file_kd = './data/s1-gt-kd.txt'

diswotv2_dict = {
    'cls': {},
    'kd': {},
}

# open file_cls and convert the content in the dict format
with open(file_cls, 'r') as f:
    for line in f:
        struct, acc = line.split()
        diswotv2_dict['cls'][struct] = float(acc)

# open file_kd and convert the content in the dict format
with open(file_kd, 'r') as f:
    for line in f:
        struct, acc = line.split()
        diswotv2_dict['kd'][struct] = float(acc)

# convert the diswotv2_dict to .pkl file
with open('./data/diswotv2_dict.pkl', 'wb') as f:
    pickle.dump(diswotv2_dict, f)

# compute the md5 of diswotv2_dict.pkl and rename it
with open('./data/diswotv2_dict.pkl', 'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()

os.rename('./data/diswotv2_dict.pkl', f'./data/diswotv2_dict_{md5}.pkl')
