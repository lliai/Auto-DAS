import random

from .registry import (_dists_entrypoints, _trans_entrypoints,
                       _weight_entrypoints, dists_entrypoints, is_distance,
                       is_transform, is_weight, trans_entrypoints,
                       weight_entrypoints)

# transform


def show_available_transforms():
    """Displays available transforms"""
    print(list(trans_entrypoints.keys()))


def build_transform(trans_name):
    """Builds a transform"""
    if not is_transform(trans_name):
        raise ValueError(
            f'Unkown transform: {trans_name} not in {list(_trans_entrypoints.keys())}'
        )

    return trans_entrypoints(trans_name)


def sample_transform():
    """Samples a transform randomly"""
    return random.sample(list(_trans_entrypoints.keys()), 1)[0]


# distance
def show_available_distances():
    """Displays available distances"""
    print(list(dists_entrypoints.keys()))


def build_distance(dist_name):
    """Builds a distance"""
    if not is_distance(dist_name):
        raise ValueError(
            f'Unkown distance: {dist_name} not in {list(_dists_entrypoints.keys())}'
        )

    return dists_entrypoints(dist_name)


def sample_distance():
    """Samples a distance randomly"""
    return random.sample(list(_dists_entrypoints.keys()), 1)[0]


# weight


def show_available_weight():
    """Displays available weight"""
    print(list(dists_entrypoints.keys()))


def build_weight(weight_name):
    """Builds a weight"""
    if not is_weight(weight_name):
        raise ValueError(
            f'Unkown weight: {weight_name} not in {list(_weight_entrypoints.keys())}'
        )

    return weight_entrypoints(weight_name)


def sample_weight():
    """Samples a weight randomly"""
    return random.sample(list(_weight_entrypoints.keys()), 1)[0]
