import os
from dynib.datasets.numpy_dataset import NumpyFileDataset

TRAIN_DATAFILE = "X_train.npy"
TEST_DATAFILE = "X_test.npy"
TRAIN_LABELFILE = "y_train.npy"
TEST_LABELFILE = "y_test.npy"

DATASETS = [
    "ArticularyWordRecognition",
    "AtrialFibrilation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "EigenWorms",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PEMS-SF",
    "PenDigits",
    "Phoneme",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
]


class TapnetSet(NumpyFileDataset):
    def __init__(self, name, root="./data/tapnet", train=True):
        datapath = os.path.join(root, name)
        if train:
            xf = os.path.join(datapath, TRAIN_DATAFILE)
            yf = os.path.join(datapath, TRAIN_LABELFILE)
        else:
            xf = os.path.join(datapath, TEST_DATAFILE)
            yf = os.path.join(datapath, TEST_LABELFILE)
        super(TapnetSet, self).__init__(xf, yf, classify=True)


import numpy as np


def get_tapnet_args(name):
    dataset = TapnetSet(name)
    return (list(dataset.x.shape), len(np.unique(dataset.y)))
