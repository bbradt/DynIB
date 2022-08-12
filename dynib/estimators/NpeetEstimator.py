import os
from dynib.estimators import Estimator
from dynib.estimators.npeet.entropy_estimators import entropy, mi, entropyd, midd, micd
import numpy as np


class DEFAULTS:
    save_folder = os.path.join(Estimator.SAVE_FOLDER, 'npeet')
    normalize = False
    modes = dict(x='continuous',
                 y='discrete',
                 z='continuous')


class NpeetEstimator(Estimator.Estimator):
    def __init__(self,
                 save_folder=DEFAULTS.save_folder,
                 normalize=DEFAULTS.normalize,
                 modes=DEFAULTS.modes,
                 workers=8,
                 ):
        self.modes = modes
        self.normalize = normalize
        super(NpeetEstimator, self).__init__(save_folder=save_folder)

    def compute(self, epoch, acts, x, label):
        """
            label is the full NxNC matrix of labels
            x is the full NxNC matrix of data
            acts is the activations for epoch - epcoh
        """
        xm, ym, tm = self.modes['x'], self.modes['y'], self.modes['z']
        to_measure = [('x', x, xm), ('y', label, ym)]
        result = {}
        for (l, xy, xym) in to_measure:
            if (tm, xym) == ('continuous', 'continuous'):
                #print('Computing c-c for acts vs %s' % (l))
                information = mi(acts, xy)
                xentropy = entropy(xy)
                tentropy = entropy(acts)
            elif (tm, xym) == ('continuous', 'discrete'):
                #print('Computing c-d for acts vs %s' % (l))
                information = micd(acts, xy)
                xentropy = entropyd(xy)
                tentropy = entropy(acts)
            elif (tm, xym) == ('discrete', 'continuous'):
                #print('Computing d-c for acts vs %s' % (l))
                information = micd(xy, acts)
                xentropy = entropy(xy)
                tentropy = entropyd(acts)
            elif (tm, xym) == ('discrete', 'discrete'):
                #print('Computing d-d for acts vs %s' % (l))
                information = midd(acts, xy)
                xentropy = entropyd(xy)
                tentropy = entropyd(acts)
            if self.normalize:
                norm_factor = abs(min(xentropy, tentropy))
                if norm_factor == 0:
                    norm_factor = 1
                information = information / norm_factor
            result[l] = information
        return result['x'], result['y']
