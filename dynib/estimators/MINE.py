import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

from dynib.estimators import Estimator


class DEFAULTS:
    save_folder = os.path.join(Estimator.SAVE_FOLDER, 'MINE')
    normalize = False
    H = 20
    n_epochs = 500
    perturb_acts = True
    perturb_x = True
    perturb_y = False


class MINE(Estimator.Estimator):
    def __init__(self,
                 save_folder=DEFAULTS.save_folder,
                 normalize=DEFAULTS.normalize,
                 H=DEFAULTS.H,
                 n_epochs=DEFAULTS.n_epochs,
                 perturb_x=DEFAULTS.perturb_x,
                 perturb_acts=DEFAULTS.perturb_acts,
                 perturb_y=DEFAULTS.perturb_y,
                 workers=0,
                 ):
        self.normalize = normalize
        self.H = H
        self.n_epochs = n_epochs
        self.perturb_x = perturb_x
        self.perturb_acts = perturb_acts
        self.perturb_y = perturb_y
        super(MINE, self).__init__(save_folder=save_folder,workers=0)

    def train(self, x_in, y_in):

        # shuffle and concatenate
        y_shuffle = tf.random_shuffle(y_in)
        x_conc = tf.concat([x_in, x_in], axis=0)
        y_conc = tf.concat([y_in, y_shuffle], axis=0)

        # propagate the forward pass
        layerx = layers.linear(x_conc, self.H)
        layery = layers.linear(y_conc, self.H)
        layer2 = tf.nn.relu(layerx + layery)
        output = layers.linear(layer2, 1)

        # split in T_xy and T_x_y predictions
        N_samples = tf.shape(x_in)[0]
        T_xy = output[:N_samples]
        T_x_y = output[N_samples:]

        # compute the negative loss (maximise loss == minimise -loss)
        neg_loss = -(tf.reduce_mean(T_xy, axis=0) - tf.log(tf.reduce_mean(tf.exp(T_x_y))))
        opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(neg_loss)
    
        return neg_loss, opt

    def _compute(self, x, y, xlabel='x', ylabel='a'):
        x_in = tf.placeholder(tf.float32, [None, x.shape[1]], name='x_in')
        y_in = tf.placeholder(tf.float32, [None, y.shape[1]], name='y_in')
        # make the loss and optimisation graphs
        neg_loss, opt = self.train(x_in, y_in)

        # start the session
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        # train
        MIs = []
        for epoch in range(self.n_epochs):
            if xlabel=='x' and self.perturb_x or xlabel=='y' and self.perturb_y or xlabel=='a' and self.perturb_acts:
                xp = x + np.random.normal(size=x.shape,scale=0.01)
            else:
                xp = x
            if ylabel=='x' and self.perturb_x or ylabel=='y' and self.perturb_y or ylabel=='a' and self.perturb_acts:
                yp = y + np.random.normal(size=y.shape,scale=0.01)
            else:
                yp = y
            # perform the training step
            feed_dict = {x_in: xp, y_in: yp}
            _, neg_l = sess.run([opt, neg_loss], feed_dict=feed_dict)

            # save the loss
            MIs = -neg_l
        tf.reset_default_graph()
        return MIs


    def compute(self, epoch, acts, x, label):
        print('Epoch %d - MINE ixt' % epoch)
        ixt = self._compute(x, acts, xlabel='x', ylabel='a')
        print('Epoch %d - MINE ity' % epoch)
        ity = self._compute(acts, label, xlabel='a', ylabel='y')
        return ixt, ity
