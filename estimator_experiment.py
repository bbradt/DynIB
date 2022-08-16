import os
import pandas as pd
import copy
import json
import glob
import torch
import numpy as np
import argparse
from dynib.estimators.NpeetEstimator import NpeetEstimator
from dynib.estimators.TishbyEstimator import TishbyEstimator
from dynib.ibhook import IBHook
from sklearn.model_selection import KFold

#from dynib.estimators.ThothEstimator import ThothEstimator
#from dynib.estimators.MINE import MINE
import torchvision.datasets as td
import torchvision.transforms as tft
from torch.utils.data import Subset, DataLoader
from dynib.models.MLP import MLP
from dynib.datasets.mat_dataset import MatDataset
from tqdm import tqdm
parser = argparse.ArgumentParser()

parser.add_argument("--logdir", default="./results/initial_fmnist_mlp", type=str)
parser.add_argument("--k", default=0, type=int)
parser.add_argument("--estimator", default="npeet", type=str)
parser.add_argument("--estimator_args", default="[]", type=str)
parser.add_argument("--estimator_kwargs", default="{}", type=str)
parser.add_argument("--dataset", default="fmnist", type=str)
parser.add_argument("--model", default="mlp", type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--kf", default=10, type=int)
parser.add_argument("--model_args", default="[dataset[0][0].shape, num_classes]", type=str)
parser.add_argument("--model_kwargs", default='{"hidden_layers":[512,256,128,64,32,16]}', type=str)
parser.add_argument("--epoch", default=1, type=int)
#parser

args = parser.parse_args()

args.estimator_args = json.loads(args.estimator_args)
args.estimator_kwargs = json.loads(args.estimator_kwargs)

if args.estimator == "npeet":
    estimator = NpeetEstimator(*args.estimator_args, **args.estimator_kwargs)
elif args.estimator == "tishby":
    estimator = TishbyEstimator(*args.estimator_args, **args.estimator_kwargs)

fullpath = os.path.join(args.logdir, "kf%d" % args.k, "checkpoints")


# Data loading
if '.mat' in args.dataset:
    dataset = MatDataset(os.path.join("./data", args.dataset))
    num_classes = 2
elif "mnist" == args.dataset:
    dataset = td.MNIST("./data", train=True, download=True, transform=tft.Compose([tft.ToTensor(), tft.Lambda(lambda x: torch.flatten(x))]))
    num_classes = 10
elif "fmnist" == args.dataset:
    dataset = td.FashionMNIST("./data", train=True, download=True, transform=tft.Compose([tft.ToTensor(), tft.Lambda(lambda x: torch.flatten(x))]))
    num_classes = 10
elif "emnist" == args.dataset:
    dataset = td.EMNIST("./data", "letters", train=True, download=True, transform=tft.Compose([tft.ToTensor(), tft.Lambda(lambda x: torch.flatten(x))]))
    num_classes = 26
elif "kmnist" == args.dataset:
    dataset = td.KMNIST("./data", train=True, download=True, transform=tft.Compose([tft.ToTensor(), tft.Lambda(lambda x: torch.flatten(x))]))
    num_classes = 10
data_idx = np.arange(len(dataset))
kf = KFold(n_splits=args.kf, random_state=args.seed, shuffle=True)
for ki, (train_idx, valid_idx) in enumerate(kf.split(data_idx)):
    if ki == args.k:
        break
train_dataset = Subset(dataset, train_idx)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
args.model_args = eval(args.model_args)
args.model_kwargs = eval(args.model_kwargs)
if args.model.lower() == "mlp":
    model = MLP(*args.model_args, flatten=True, **args.model_kwargs)
rows = []
#for i in tqdm(range(1,101)):
i = args.epoch
checkpoint = os.path.join(fullpath,"model.%04d.pth" % i)
mmodel = copy.deepcopy(model)
mmodel.load_state_dict(torch.load(checkpoint, map_location="cpu"))
mmodel.train()
hook = IBHook(mmodel)
xhs = dict()
yhs = dict()
#opt = torch.optim.Adam(mmodel.parameters(), lr=1)

for j, (image, target) in enumerate(tqdm(train_loader)):
    output = mmodel(image)
    if len(target.shape) == 1:
        target = torch.nn.functional.one_hot(target)
    image = image.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    for module_name, act in hook.output_activations.items():
        act = act.detach().cpu().numpy()
        xh, hy = estimator.compute(i, act, image, target)
        if module_name not in xhs.keys():
            xhs[module_name] = xh
            yhs[module_name] = hy
        else:
            xhs[module_name] = (xhs[module_name] + xh)/2
            yhs[module_name] = (yhs[module_name] + hy)/2
    hook.clear()
for module_name in xhs.keys():
    xh = xhs[module_name]
    yh = yhs[module_name]
    row = dict(module=module_name, xh=xh, hy=hy, epoch=i, estimator=args.estimator, k=args.k)
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(fullpath, "information_estimator_epoch%d.csv" % args.epoch))