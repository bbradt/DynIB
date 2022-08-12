import torch
import torch.nn as nn

import torch
import catalyst
from catalyst import dl, metrics, utils
from dynib.ibhook import IBHook
import copy
import pandas as pd
import os

class IBRunner(dl.Runner):
    def __init__(self, model, compute_h2h=False):
        super().__init__()
        self.HY = []
        self.XH = []
        self.HH = []
        self.model = model
        self.hook = IBHook(self.model)
        self.compute_h2h = compute_h2h

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss"]
        }

    def handle_batch(self, batch):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `train()`.
        #x = batch[self._input_key]
        #y = batch[self._target_key]
        x, y = batch
        self.hook.record_input(x, y)        
        y_pred = self.model(x)

     
        loss = self.criterion(y_pred, y)                    
        # Update metrics (includes the metric that tracks the loss)
        self.batch_metrics.update({"loss": loss})
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.HY.append(copy.deepcopy(self.hook.HY))
        hydf = pd.DataFrame(self.HY)
        hydf.to_csv(os.path.join(self._logdir, "hy.csv"))
        self.XH.append(copy.deepcopy(self.hook.XH))
        hydf = pd.DataFrame(self.XH)
        hydf.to_csv(os.path.join(self._logdir, "xh.csv"))
        if self.compute_h2h:
            self.hook.compute_h2h()
        self.HH.append(copy.deepcopy(self.hook.HH_flat))
        
        self.hook.clear()
        self.batch = {"logits": y_pred.clone(), "features": x.clone(), "targets": y.clone()}
    
    def on_loader_end(self, runner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)