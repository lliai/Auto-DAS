import pdb

from diswotv2.api.nas101_api import NB101API

# api = NB101API(
#     '/home/stack/project/PicoNAS/data/benchmark/nasbench_only108.tfrecord')

api = NB101API('./data/nb101_dict.pkl')

pdb.set_trace()
rnd_hash = api.random_hash()

model = api.get_nb101_model(rnd_hash)
