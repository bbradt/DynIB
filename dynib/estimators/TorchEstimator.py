import os
from dynib.estimators import Estimator
from dynib.estimators.npeet.entropy_estimators import entropy, mi, entropyd, midd, micd
import numpy as np
from dynib.estimators.torch_mi.MutualInformation import MutualInformation

class DEFAULTS:
    save_folder = os.path.join(Estimator.SAVE_FOLDER, 'npeet')
    normalize = False
    modes = dict(x='continuous',
                 y='discrete',
                 z='continuous')


class TorchEstimator(Estimator.Estimator):
    def __init__(self,
                 save_folder=DEFAULTS.save_folder,
                 normalize=DEFAULTS.normalize,
                 modes=DEFAULTS.modes,
                 workers=8,
                 device="cpu",
                 num_bins=256,
                 sigma=0.4,
                 ):
        self.modes = modes
        self.normalize = normalize
        self.num_bins = num_bins
        self.sigma = sigma
        self.device=device
        self.engine = MutualInformation(num_bins=num_bins, sigma=sigma, normalize=self.normalize, device=device)#.to(device)
        super(TorchEstimator, self).__init__(save_folder=save_folder)

    def compute(self, epoch, acts, x, label):
        """
            label is the full NxNC matrix of labels
            x is the full NxNC matrix of data
            acts is the activations for epoch - epcoh
        """
        #xm, ym, tm = self.modes['x'], self.modes['y'], self.modes['z']
        #to_measure = [('x', x, xm), ('y', label, ym)]
        xh = self.engine(x, acts)
        hy = self.engine(acts, label)
        return xh, hy
