"""Python file to instantite the model and the transform that goes with it."""
from model import Net
from model import get_resnet18, get_resnet34
from data import data_transforms,data_transforms_resnet, data_transforms_resnet_rotation_flip


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet18":
          return get_resnet18()
        elif self.model_name == "resnet34":
          return get_resnet34()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return [data_transforms]
        elif self.model_name == "resnet18":
            return [data_transforms_resnet,data_transforms_resnet_rotation_flip]
        elif self.model_name == "resnet34":
            return [data_transforms_resnet,data_transforms_resnet_rotation_flip]
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
