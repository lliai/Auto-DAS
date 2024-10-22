import hashlib
import os
import pickle

from diswotv2.api.nas101_api import nb, query_nb101_acc

nb101_dict = {}
for _hash in nb.hash_iterator():
    acc = query_nb101_acc(_hash)
    nb101_dict[_hash] = acc

# save nb101_dict to pickle
with open('./data/nb101_dict.pkl', 'wb') as f:
    pickle.dump(nb101_dict, f)

# compute the md5 of nb101_dict
with open('./data/nb101_dict.pkl', 'rb') as f:
    data = f.read()
    md5 = hashlib.md5(data).hexdigest()

# rename the file
os.rename('./data/nb101_dict.pkl', f'./data/nb101_dict_{md5}.pkl')
