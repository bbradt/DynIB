from torch.utils.data.dataset import Dataset
import numpy as np
import torch


class NumpyDataset(Dataset):
    def __init__(self, x, y, indices=None, y_col=None, classify=True):
        self.x = x
        self.y = y
        self.y_col = y_col
        self.classify = classify

        # TypeCasting
        if type(self.x) is np.ndarray:
            self.x = torch.from_numpy(self.x)
            self.y = torch.from_numpy(self.y)
        self.x = self.x.float()

        # Indices
        if indices is not None:
            self.x = x[indices, ...]
            self.y = y[indices, ...]
        if y_col:
            self.y = y[:, y_col]

        # Classification Only
        if classify:
            self.y = self.y.flatten().long()
        else:
            self.y = self.y.float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, k):
        return self.x[k, ...], self.y[k, ...]

    def get_subset_by_condition(self, condition_func, data=True):
        if data:
            indices = torch.where(torch.tensor(condition_func(self.x)))[0]
        else:
            indices = torch.where(torch.tensor(condition_func(self.y)))[0]
        return NumpyDataset(
            self.x, self.y, indices=indices, y_col=self.y_col, classify=self.classify
        )

    def get_subsets_by_conditions(self, conditions, data=True):
        for condition in conditions:
            yield self.get_subsets_by_condition(condition, data=data)

    def get_subset_by_indices(self, indices):
        return NumpyDataset(
            self.x, self.y, indices=indices, y_col=self.y_col, classify=self.classify
        )


class NumpyFileDataset(NumpyDataset):
    def __init__(self, x_filename, y_filename, indices=None, y_col=None, classify=True):
        x = np.load(x_filename)
        y = np.load(y_filename)
        super(NumpyFileDataset, self).__init__(
            x, y, indices=indices, y_col=y_col, classify=classify
        )


class PtFileDataset(NumpyDataset):
    def __init__(self, x_filename, y_filename, indices=None, y_col=None, classify=True):
        x = torch.load(x_filename)
        y = torch.load(y_filename)
        super(PtFileDataset, self).__init__(
            x, y, indices=indices, y_col=y_col, classify=classify
        )
