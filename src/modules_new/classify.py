# -*- coding:utf -*-
import torch.nn as nn
import torch.nn.functional as f


class SexClassify_(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1000, 2)
            # nn.Sigmoid()
        )

    def forward(self, inputs):
        # print(f'inputs.shape {inputs.shape}')
        # print(f'classify model {self.classify}')
        x = self.classify(inputs)
        return x


class AgeClassify_(nn.Module):
    def __init__(self, input_size, age_classes):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1000, age_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        x = self.classify(inputs)
        return x



#### original, do not change
class SexClassify(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1000, 2)
        )

    def forward(self, inputs):
        x = self.classify(inputs)
        return x


class AgeClassify(nn.Module):
    def __init__(self, input_size, age_classes):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1000, age_classes)
        )

    def forward(self, inputs):
        x = self.classify(inputs)
        return x


class Classify_(nn.Module):
    def __init__(self, input_size=2048, age_classes=10):
        super().__init__()
        self.gender_classify = SexClassify_(input_size)
        self.age_classify = AgeClassify_(input_size, age_classes)

    def forward(self, inputs):
        gender_out = self.gender_classify(inputs)
        age_out = self.age_classify(inputs)
        return {'gender': gender_out, 'age': age_out}


class Classify(nn.Module):
    def __init__(self, input_size=2048, age_classes=10):
        super().__init__()
        self.gender_classify = SexClassify(input_size)
        self.age_classify = AgeClassify(input_size, age_classes)

    def forward(self, inputs):
        gender_out = self.gender_classify(inputs)
        age_out = self.age_classify(inputs)
        return {'gender': gender_out, 'age': age_out}
