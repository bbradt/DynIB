
from dynib.models.MLP import MLP
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

MODEL_DICT = dict(
    "rnn": RNN,
    "mlp": MLP,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnset152": resnet152,
)

def get_model(name, num_classes, image_size=224, channels=3, patch_size=32, vape=False, **kwargs):
    name = name.lower()
    model = None