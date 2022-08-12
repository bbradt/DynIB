import os
from dynib.estimators import Estimator
from dynib.estimators.NPEET_LNC.lnc import MI
import numpy as np

class DEFAULTS:
    save_folder = os.path.join(Estimator.SAVE_FOLDER, 'npeet')
    normalize = False
    modes = dict(x='continuous',
                 y='continuous',
                 z='continuous')
    base = np.exp(1)
    alpha = 0.25
    k = 5

class NpeetLNCEstimator(Estimator.Estimator):
    def __init__(self,
                 save_folder=DEFAULTS.save_folder,
                 normalize=DEFAULTS.normalize,
                 modes=DEFAULTS.modes,
                 base=DEFAULTS.base,
                 alpha=DEFAULTS.alpha,
                 k=DEFAULTS.k
                 ):
        self.base = base
        self.alpha = alpha
        self.k = k
        if 'discrete' in modes.values():
            raise(ValueError("LNC estimator can only be used for continuous variables"))
        super(NpeetLNCEstimator, self).__init__(save_folder=save_folder, normalize=normalize, modes=modes)

    def compute(self, epoch, acts, x, label):
        ixt = MI.mi_LNC([x,acts], k=self.k, base=self.base, alpha=self.alpha)
        ity = MI.mi_LNC([acts,label], k=self.k, base=self.base, alpha=self.alpha)
        return ixt, ity
