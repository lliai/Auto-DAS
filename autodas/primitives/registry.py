_trans_entrypoints = {}
_dists_entrypoints = {}
_weight_entrypoints = {}


def register_transform(fn):
    """Register a transform function"""
    _trans_entrypoints[fn.__name__] = fn
    return fn


def trans_entrypoints(transform_name):
    """Returns the transform function"""
    return _trans_entrypoints[transform_name]


def is_transform(transform_name):
    """Checks if the transform is registered"""
    return transform_name in _trans_entrypoints


def register_distance(fn):
    """Register a distance function"""
    _dists_entrypoints[fn.__name__] = fn
    return fn


def dists_entrypoints(distance_name):
    """Returns the distance function"""
    return _dists_entrypoints[distance_name]


def is_distance(distance_name):
    """Checks if the distance is registered"""
    return distance_name in _dists_entrypoints


def register_weight(fn):
    """Register a weight function"""
    _weight_entrypoints[fn.__name__] = fn
    return fn


def weight_entrypoints(weight_name):
    """Returns the weight function"""
    return _weight_entrypoints[weight_name]


def is_weight(weight_name):
    """Checks if the weight is registered"""
    return weight_name in _weight_entrypoints
