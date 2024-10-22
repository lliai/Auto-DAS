from abc import abstractmethod

from torch.nn.modules.loss import _Loss


class BaseInteraction(_Loss):
    """ Base class for interaction

    Attributes
    ----------
    alleletype : str
        the alleletype of the structure
    reward_score : float
        the reward score of the structure
    repr_geno : str
        the repr of the genotype
    """

    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean',
                 *args,
                 **kwargs):
        super().__init__(size_average, reduce, reduction)
        # self._alleletype = None
        self._reward_score = None
        self._mutable = False  # whether the structure is mutable

    @property
    def alleletype(self):
        """ alleletype of the structure"""
        return self._alleletype

    @alleletype.setter
    def alleletype(self, _alleletype):
        """ set the alleletype of the structure"""
        self._alleletype = _alleletype

    def __repr__(self) -> str:
        """ return the repr of the genotype"""
        return self.__str__()

    @property
    def reward_score(self):
        """ reward score of the structure"""
        return self._reward_score

    @reward_score.setter
    def reward_score(self, _reward_score):
        """ set the reward score of the structure"""
        self._reward_score = _reward_score

    @property
    def mutable(self):
        """ whether the structure is mutable"""
        return self._mutable

    @mutable.setter
    def mutable(self, _mutable):
        """ set the mutable of the structure"""
        self._mutable = _mutable

    @abstractmethod
    def cross_over(self, other):
        """ Cross over two tree structure and return new one """
        pass

    @abstractmethod
    def mutate(self):
        """ Mutate the alleletype of the structure """
        pass
