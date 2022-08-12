import sklearn.datasets as skd
import inspect
import torch
from torch.utils.data import Dataset

def invalid_args(func, argdict):
    sig = inspect.signature(func)
    return {k:v for k,v in argdict.items() if k in sig.parameters.keys()}

class SklearnDataset(Dataset):
    def __init__(self, constructor, *args, return_X_y=True, data_home='./data', xvar="data", yvar="target", classify=False, **kwargs):
        super().__init__()
        self.return_X_y = False
        self.data_home = data_home
        kwargs["return_X_y"] = return_X_y
        kwargs["data_home"] = data_home
        kwargs = invalid_args(constructor, kwargs)
        if 'return_X_y' in kwargs.keys() and kwargs['return_X_y']:
            self.return_X_y = True
        if self.return_X_y:
            self.X, self.y = constructor(*args, **kwargs)
        else:
            self.dataset = constructor(*args, **kwargs)
            self.X = self.dataset[xvar]
            self.y  = self.dataset[yvar]
        try:
            self.X = torch.Tensor(self.X).float()
            self.y = torch.Tensor(self.y).float()
            if classify:
                self.y = self.y.long()
        except Exception:
            pass
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

class SklNewsGroups20(SklearnDataset):
    def __init__(self):
        super().__init__(skd.fetch_20newsgroups, classify=True)

class SklCaliforniaHousing(SklearnDataset):
    def __init__(self):
        super().__init__(skd.fetch_california_housing)

class SklCoverType(SklearnDataset):
    def __init__(self):
        super().__init__(skd.fetch_covtype, classify=True)

class SklKddCup99(SklearnDataset):
    def __init__(self):
        super().__init__(skd.fetch_kddcup99, classify=True)

class SklLfw(SklearnDataset):
    def __init__(self):
        super().__init__(skd.fetch_lfw_pairs, classify=True)

class SklOlivettiFaces(SklearnDataset):
    def __init__(self):
        super().__init__(skd.fetch_olivetti_faces, classify=True)

class SklRcv1(SklearnDataset):
    def __init__(self):
        super().__init__(skd.fetch_rcv1, classify=True)

class SklSpeciesDistributions(SklearnDataset):
    def __init__(self):
        super().__init__(skd.fetch_species_distributions)

class SklBoston(SklearnDataset):
    def __init__(self):
        super().__init__(skd.load_boston)

class SklBreastCancer(SklearnDataset):
    def __init__(self):
        super().__init__(skd.load_breast_cancer, classify=True)

class SklDiabetes(SklearnDataset):
    def __init__(self):
        super().__init__(skd.load_diabetes)

class SklDigits(SklearnDataset):
    def __init__(self):
        super().__init__(skd.load_digits, classify=True)

class SklIris(SklearnDataset):
    def __init__(self):
        super().__init__(skd.load_iris, classify=True)

class SklLinnerud(SklearnDataset):
    def __init__(self):
        super().__init__(skd.load_linnerud)

if __name__ == "__main__":
    test_dataset = CoverType()
    print(test_dataset[0])