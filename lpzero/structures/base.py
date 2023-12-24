from abc import ABCMeta, abstractmethod


class BaseStructure(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._genotype = None
        self._sp_score = None
        self._repr_geno = None

    @property
    def genotype(self):
        return self._genotype

    @genotype.setter
    def genotype(self, _genotype):
        self._genotype = _genotype

    def __repr__(self) -> str:
        return self._repr_geno

    @property
    def sp_score(self):
        return self._sp_score

    @sp_score.setter
    def sp_score(self, _sp_score):
        self._sp_score = _sp_score

    @abstractmethod
    def cross_over_by_genotype(self, other):
        """ Cross over two tree structure and return new one """
        pass

    @abstractmethod
    def mutate_by_genotype(self):
        """ Mutate the genotype of the structure """
        pass