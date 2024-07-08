import numpy as np


class BH:

    def __init__(self, dists, fdr=0.05, dependency=True):
        '''
        dist: list of numpy arrays, each array is a calibration set describing the null distribution
        fdr: in our case is equal to the prescribed false positive rate
        dependency: whether the statistics for these tests are dependent with each other (in our case is True)
        '''
        self.dists = dists
        self.K = len(self.dists)
        self.fdr = fdr
        self.dependency = dependency
        self.pv_conformal = None
        if self.dependency:
            self.const = 0
            for j in range(self.K):
                self.const += self.K / (j + 1)
        else:
            self.const = self.K

    def test(self, x):
        self.pv_conformal = self.conformal_pvalue(x)
        pv_ranked = np.sort(self.pv_conformal)
        rank = np.argsort(self.pv_conformal)
        for i in range(self.K):
            if pv_ranked[i] > self.fdr * (i + 1) / self.const:
                break
        rejected = rank[:i]

        return rejected

    def test_v2(self, x):
        self.pv_conformal = self.conformal_pvalue(x)
        max_pv = self.pv_conformal.max()
        min_pv = self.pv_conformal.min()

        if max_pv <= self.fdr or min_pv <= self.fdr / 2:
            return 1
        else:
            return 0

    def conformal_pvalue(self, x):
        pv_conformal = []
        for i in range(len(x)):
            dist = self.dists[i]
            pv_conformal.append((len(np.where(dist >= x[i])[0]) + 1) / (len(dist) + 1))

        return np.asarray(pv_conformal)

def build_bh_procedure(dists, fdr=0.05, dependency=True):
    return BH(dists, fdr, dependency)