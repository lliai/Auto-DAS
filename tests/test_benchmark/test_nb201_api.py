import random

from nas_201_api import NASBench201API

nb201_api = NASBench201API(
    file_path_or_dict='./data/NAS-Bench-201-v1_0-e61699.pth', verbose=False)


def to_arch_str(arch_list: list):
    """convert arch string. """
    assert isinstance(arch_list, list), 'invalid arch_list type : {:}'.format(
        type(arch_list))

    strings = []
    for node_info in arch_list:
        string = '|'.join([x[0] + '~{:}'.format(x[1]) for x in node_info])
        string = f'|{string}|'
        strings.append(string)
    return '+'.join(strings)


def mutate(arch_list):
    """mutate the arch in nas-bench-201
    arch_list = [(('avg_pool_3x3', 0),), (('skip_connect', 0), ('none', 1)), (('none', 0), ('none', 1), ('skip_connect', 2))]

    - random sample a position from six positions, eg: avg_pool_3x3
    - replace it with a randomly sampled operation, eg: none
    - candidate operation is:
        - nor_conv_1x1
        - nor_conv_3x3
        - avg_pool_3x3
        - none
        - skip_connect

    return [(('none', 0),), (('skip_connect', 0), ('none', 1)), (('none', 0), ('none', 1), ('skip_connect', 2))]
    """
    if isinstance(arch_list, str):
        arch_list = nb201_api.str2lists(arch_list)

    # convert items in arch_list to list
    tmp_list = []
    for layer in arch_list:
        tmp_ = []
        for oper in layer:
            tmp_.append(list(oper))
        tmp_list.append(tmp_)

    # candidate position
    operations = [
        'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'none'
    ]

    # sample layer from [0, 1, 2]
    layer_idx = random.randint(0, 2)

    # sample operation from operations
    op_idx = random.randint(0, len(tmp_list[layer_idx]))

    try:
        tmp_list[layer_idx][op_idx][0] = operations[random.randint(
            0,
            len(operations) - 1)]
    except IndexError:
        import pdb
        pdb.set_trace()

    return tmp_list


def crossover(arch_list1, arch_list2):
    """ make cross over between two archs"""
    if isinstance(arch_list1, str):
        arch_list1 = nb201_api.str2lists(arch_list1)
    if isinstance(arch_list2, str):
        arch_list2 = nb201_api.str2lists(arch_list2)

    # convert items in arch_list to list
    tmp_list1 = []
    tmp_list2 = []
    for layer in arch_list1:
        tmp_ = []
        for oper in layer:
            tmp_.append(list(oper))
        tmp_list1.append(tmp_)

    for layer in arch_list2:
        tmp_ = []
        for oper in layer:
            tmp_.append(list(oper))
        tmp_list2.append(tmp_)

    # sample layer from [0, 1, 2]
    layer_idx = random.randint(0, 2)

    # sample operation from operations
    op_idx = random.randint(0, len(tmp_list1[layer_idx]))

    try:
        tmp_list1[layer_idx][op_idx][0] = tmp_list2[layer_idx][op_idx][0]
    except IndexError:
        import pdb
        pdb.set_trace()

    return tmp_list1


# arch = nb201_api.arch(3)
# arch_list = nb201_api.str2lists(arch)
# print(arch_list)
# print(mutate(arch_list))

# arch1 = nb201_api.arch(3)
# arch2 = nb201_api.arch(4)
# print(f'arch1: {arch1}')
# print(f'arch2: {arch2}')
# print(f'cross over: {crossover(arch1, arch2)}')

# arch = nb201_api.arch(3)
# arch_list = nb201_api.str2lists(arch)
# arch_str = to_arch_str(arch_list)
# print(f'arch_str: {arch}')
# print(f'arch_list: {arch_list}')
# print(f'arch_str: {arch_str}')

# searched by diswotv2
index = nb201_api.query_index_by_arch(
    '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_1x1~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'
)
print(f'index: {index}')  # 5292
