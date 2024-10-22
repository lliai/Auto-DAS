from abc import ABCMeta, abstractmethod


class BaseInstinct(metaclass=ABCMeta):
    """ Base class for instinct

    Attributes
    ----------
    genotype : str
        the genotype of the structure
    reward_score : float
        the reward score of the structure
    repr_geno : str
        the repr of the genotype

    """

    def __init__(self, *args, **kwargs):
        """ Base class for instinct """
        super().__init__()
        self._genotype = None
        self._reward_score = None
        self._repr_geno = None

    @property
    def genotype(self):
        """ genotype of the structure"""
        return self._genotype

    @genotype.setter
    def genotype(self, _genotype):
        """ set the genotype of the structure"""
        self._genotype = _genotype

    def __repr__(self) -> str:
        """ return the repr of the genotype """
        return self._repr_geno

    @property
    def reward_score(self):
        """ reward score of the structure"""
        return self._reward_score

    @reward_score.setter
    def reward_score(self, _reward_score):
        """ set the reward score of the structure"""
        self._reward_score = _reward_score

    @abstractmethod
    def crossover(self, other):
        """ Cross over two tree structure and return new one """
        pass

    @abstractmethod
    def mutate(self):
        """ Mutate the genotype of the structure """
        pass
