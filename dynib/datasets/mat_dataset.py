import scipy.io as sio
from torch.utils.data import Dataset
import torchvision.transforms as tvt

class MatDataset(Dataset):
    def __init__(self, path, xvar="F", yvar="y", transform=tvt.ToTensor(), target_transform=tvt.ToTensor(), classify=True):
        super().__init__()
        self.path = path
        self.data = sio.loadmat(path)
        self.xvar = xvar
        self.yvar = yvar
        self.transform = transform
        self.target_transform = target_transform
        self.classify = classify        
        x = self.data[self.xvar]
        y = self.data[self.yvar]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        if self.classify:
            y = y.long()
        self.x = x.squeeze()
        self.y = y.squeeze()
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, k):
        return self.x[k, ...], self.y[k, ...]

if __name__ == "__main__":
    test_path = "./data/g1.mat"
    dataset = MatDataset(test_path)
    print(dataset[0])