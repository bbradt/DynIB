import os
from catalyst.dl import SupervisedRunner

class BatchsaveRunner(SupervisedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_batch_end(self, runner: "IRunner"):
        super().on_batch_end(runner)
        batch = self.batch_step
        epoch = self.epoch_step
        epdir = os.path.join(self._logdir, "epoch-%04d" % epoch)
        os.makedirs(epdir, exist_ok=True)
        savedir = os.path.join(epdir, "b-%d-model.pth" % batch)

        return super().on_batch_end(runner)