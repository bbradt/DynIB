import os
from ast import arg
import numpy as np
from dynib.models.MLP import MLP
from dynib.datasets.mat_dataset import MatDataset
from dynib.ibrunner import IBRunner
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import argparse
import shutil
import torch.nn as nn
import torch
from catalyst import dl
import torch.optim.lr_scheduler as lrs 
import torch.optim as opt
import torchvision.datasets as td
import torchvision.transforms as tft
from dynib.ibhook import IBHook
from dynib.callbacks.batch_checkpoint_callback import BatchCheckpointCallback

DEFAULTS = dict(
kf = 10,
k = 0,
criterion = 'CrossEntropyLoss',
scheduler = 'ExponentialLR',
optim = 'Adam',
num_epochs = 10,
lr = 1e-6,
logdir = "./tests/mnist",
model_kwargs = '{"hidden_layers": [4096,2048,1024,512,256,128,64,32,16]}',
model_args = '[dataset[0][0].shape, 10]',
model = "mlp",
dataset = "mnist",
seed = 0,
batch_size = 128
)
parser = argparse.ArgumentParser("dynib")
for k, v in DEFAULTS.items():
    parser.add_argument("--"+k, default=v, type=type(v))
args = parser.parse_args()

# Data loading
if '.mat' in args.dataset:
    dataset = MatDataset(os.path.join("./data", args.dataset))
elif "mnist" == args.dataset:
    dataset = td.MNIST("./data", train=True, download=True, transform=tft.Compose([tft.ToTensor(), tft.Lambda(lambda x: torch.flatten(x))]))
data_idx = np.arange(len(dataset))
kf = KFold(n_splits=args.kf, random_state=args.seed, shuffle=True)
for ki, (train_idx, valid_idx) in enumerate(kf.split(data_idx)):
    if ki == args.k:
        break
train_dataset = Subset(dataset, train_idx)
valid_dataset = Subset(dataset, valid_idx)
loaders = {
    "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True),
    "valid": DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True),
}

# Further eval...
args.optim = eval("opt." + args.optim)
args.scheduler = eval("lrs." + args.scheduler)
args.model_kwargs = eval(args.model_kwargs)
args.model_args = eval(args.model_args)
args.criterion = eval("nn." + args.criterion)


# End Data Loading

# Model
if args.model.lower() == "mlp":
    model = MLP(*args.model_args, flatten=True, **args.model_kwargs)
hook = IBHook(model)
# Criterion
criterion = args.criterion()
optimizer = args.optim(model.parameters(), lr=args.lr)
scheduler = args.scheduler(optimizer, gamma=0.9)
# Callbacks
callbacks = {
    "auc": dl.AUCCallback(input_key="logits", target_key="targets"),
    "precision":  dl.PrecisionRecallF1SupportCallback(input_key="logits", target_key="targets"),
    #"callback": dl.CheckpointCallback(os.path.join(args.logdir, "checkpoints"), loader_key="train", metric_key="auc", topk=100),
    "callback": BatchCheckpointCallback(os.path.join(args.logdir, "checkpoints"), loader_key="train", metric_key="auc"),
    "accuracy": dl.AccuracyCallback(input_key="logits", target_key="targets")
}

#runner = IBRunner(
#    model=model
#)
runner = dl.SupervisedRunner(input_key="logits", target_key="targets")

if os.path.exists(args.logdir):
    shutil.rmtree(args.logdir)

runner.train(
    model=model,
    loaders=loaders,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=args.num_epochs,
    logdir=args.logdir,
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
    callbacks=callbacks,
    scheduler=scheduler
)

print("Finished")

