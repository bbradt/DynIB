import os
from . import Estimator
from dynib.idnns.information import information_process as inn
import numpy as np 

NUM_BINS = 30
SAVE_FOLDER = os.path.join(Estimator.SAVE_FOLDER, 'tishby')


class TishbyEstimator(Estimator.Estimator):
    def __init__(self,
                 save_folder=SAVE_FOLDER,
                 num_bins=NUM_BINS,workers=8):
        super(TishbyEstimator, self).__init__(save_folder=save_folder,workers=workers)
        self.num_bins = NUM_BINS

    def extract_probs(self, label, x):
        """calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
        pys = np.sum(label, axis=0) / float(label.shape[0])
        b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
        unique_array, unique_indices, unique_inverse_x, unique_counts = \
            np.unique(b, return_index=True, return_inverse=True, return_counts=True)
        unique_a = x[unique_indices]
        b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
        pxs = unique_counts / float(np.sum(unique_counts))
        p_y_given_x = []
        for i in range(0, len(unique_array)):
            indexs = unique_inverse_x == i
            py_x_current = np.mean(label[indexs, :], axis=0)
            p_y_given_x.append(py_x_current)
        p_y_given_x = np.array(p_y_given_x).T
        b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
        unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
            np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
        pys1 = unique_counts_y / float(np.sum(unique_counts_y))
        return pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs

    def compute(self, epoch, acts, x, label, save=False):
        print('ok SHAPES', x.shape, label.shape, acts.shape)
        # if len(x.shape) == 1:
        #   x = x.reshape(x,x.size,1)
        #print('ok SHAPES',x.shape,label.shape,acts.shape)
        pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs = self.extract_probs(label, x)
        bins = np.linspace(-1, 1, self.num_bins)
        #x = x.T
        #y = y.T
        #acts = acts.T
        # TODO unpack the inn information computation
        information = inn.calc_information_for_epoch(epoch, 1, [acts], bins, unique_inverse_x, unique_inverse_y,
                                                     label, b, b1, len(unique_a), pys,
                                                     pxs, p_y_given_x, pys1, '', x.shape[1], acts.shape[1])
        if save:
            np.save('information.npy', information, allow_pickle=True)
            print(information)
        return information.item()['local_IXT'], information.item()['local_ITY']
