import os
import numpy as np
import glob
import multiprocessing

SAVE_FOLDER = os.path.join('..', 'results', 'mi_estimations')
LAYER_NAME = 'layers_'


class Estimator:
    def __init__(self, save_folder=SAVE_FOLDER,workers=8):
        self.save_folder = save_folder
        self.workers = workers
        os.makedirs(self.save_folder, exist_ok=True)
        self.save_x = os.path.join(self.save_folder, 'ixt.npy')
        self.save_y = os.path.join(self.save_folder, 'ity.npy')

    def compute(self, epoch, acts, x, label):
        return 0, 0

    def compute_at_layer(self, args):
        """
            args is a 5-tuple with the following:
                epoch - the epoch of computer
                layer - the ID of the layer
                acts - the activations for that sequential layer with size (N,n_hidden)
                x - the input with size (N,n_features)
                y - the labels with size (N,n_classes)
                q - the queue for storing input
        """
        epoch, layer, acts, x, y, q = args
        ixt, ity = self.compute(epoch, acts, x, y)
        q.put((epoch, layer, ixt, ity))

    def compute_at_sequential_layer(self, args):
        """
            args is a 5-tuple with the following:
                epoch - the epoch of computer
                layer - the ID of the layer
                acts - the activations for that sequential layer with size (N,T,n_hidden)
                x - the input with size (N,T,n_features)
                y - the labels with size (N,n_classes)
                q - the queue for storing input
        """
        epoch, layer, acts, x, y, q = args
        T = x.shape[1]
        ixt_over_sequence = np.zeros((T,))
        ity_over_sequence = np.zeros((T,))
        print(acts.shape, x.shape)
        for t in range(T):
            if len(acts.shape) < 3:
                ixt, ity = self.compute(epoch, acts, x[:, t, :], y)
            else:
                ixt, ity = self.compute(epoch, acts[:, t, :], x[:, t, :], y)
            print('Epoch %d T %d ixt %f ity %f' % (epoch, t, ixt, ity))
            ixt_over_sequence[t] = ixt
            ity_over_sequence[t] = ity
        q.put((epoch, layer, ixt_over_sequence, ity_over_sequence))

    def compute_all_epochs(self,
                           x,
                           y,
                           acts,
                           sequential=False,
                           save=True):
        """
            x is a tensor with shape [N_samples,num_timesteps (if any) ,num_features]
            y is a tensor with shape [N_samples, num_classes (or num output features)]
            acts is a tensor with shape [num_epochs, num_layers], where each layer
                contains a tensor of activations for each differing layer size
            mode is either 'standard' for Feed-Forward networks or 'sequential' for RNNs
            save - whether or not to save MI after computing
            workers - number of parallel workers to use
        """
        manager = multiprocessing.Manager()
        q = manager.Queue()
        num_epochs = len(acts)
        num_layers = len(acts[0])
        if self.workers > 0:
            with multiprocessing.Pool(self.workers) as p:
                if sequential:
                    num_layers = 1
                    p.map(self.compute_at_sequential_layer, [(r, 0, acts[r], x, y, q)
                                                            for r in range(num_epochs)
                                                            ])
                    # for r in range(num_epochs):
                    #    for ll in range(num_layers):
                    #        self.compute_at_sequential_layer((r,ll,acts[r][ll],x,y,q))
                else:
                    p.map(self.compute_at_layer, [(r, l, acts[r][l], x, y, q)
                                                for r in range(num_epochs)
                                                for l in range(num_layers)])
        else:
            if sequential:
                num_layers = 1
                for r in range(num_epochs):
                    self.compute_at_sequential_layer((r,0,acts[r],x,y,q))
            else:
                for r in range(num_epochs):
                    for ll in range(num_layers):
                        self.compute_at_layer((r,ll,acts[r][ll],x,y,q))
        ixt_over_epochs = []
        ity_over_epochs = []
        for r in range(num_epochs):
            ixt_over_epochs.append([])
            ity_over_epochs.append([])
            for l in range(num_layers):
                ixt_over_epochs[r].append(None)
                ity_over_epochs[r].append(None)
        while not q.empty():
            epoch, layer, ixt_over_sequence, ity_over_sequence = q.get()
            ixt_over_epochs[epoch][layer] = ixt_over_sequence
            ity_over_epochs[epoch][layer] = ity_over_sequence
            print('At epcoh %d seeing ixt %s, ity %s' % (epoch, ixt_over_sequence, ity_over_sequence))
        if save:
            np.save(os.path.join(self.save_folder, 'ixt.npy'), ixt_over_epochs)
            np.save(os.path.join(self.save_folder, 'ity.npy'), ity_over_epochs)
        return ixt_over_epochs, ity_over_epochs

    def compute_group_epochs(self, x, y, acts, groups={}, sequential=False, save=False, workers=32):
        """
            Compute group indexes for multiple groups of data instances.
            Groups is a dictionary with key-names corresponding to group names, and a list of
                indexes corresponding to that group.
        """
        num_epochs = len(acts)
        num_layers = len(acts[0])
        group_ixt = {}
        group_ity = {}
        for group_name, group_indexes in groups.items():
            group_x = x[group_indexes]
            group_y = y[group_indexes]
            group_acts = [acts[r][l][group_indexes] for r in range(num_epochs) for l in range(num_layers)]
            ixt, ity = self.compute_all_epochs(group_x, group_y, group_acts, sequential=sequential, save=False, workers=workers)
            group_ixt[group_name] = ixt
            group_ity[group_name] = ity
            if save:
                os.makedirs(os.path.join(self.save_folder, group_name), exist_ok=True)
                np.save(os.path.join(self.save_folder, group_name, 'ixt.npy'), ixt)
                np.save(os.path.join(self.save_folder, group_name, 'ity.npy'), ity)
        return group_ixt, group_ity
