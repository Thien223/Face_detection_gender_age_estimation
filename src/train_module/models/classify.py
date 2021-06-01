# -*- coding:utf -*-
import torch.nn as nn
import torch.nn.functional as f


class GenderClassify_(nn.Module):
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
            nn.Linear(1000, 2),
        )

    def forward(self, inputs):
        x = self.classify(inputs)
        return x


class AgeClassify_(nn.Module):
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
            nn.Linear(1000, 5),
        )

    def forward(self, inputs):
        x = self.classify(inputs)
        return x



#### original, do not change
class GenderClassify(nn.Module):
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
            nn.Linear(1000, 5)
        )

    def forward(self, inputs):
        x = self.classify(inputs)
        return x


class Classify_(nn.Module):
    def __init__(self, input_size=2048):
        super().__init__()
        self.gender_classify = GenderClassify_(input_size)
        self.age_classify = AgeClassify_(input_size)

    def forward(self, inputs):
        gender_out = self.gender_classify(inputs)
        age_out = self.age_classify(inputs)
        return {'gender': gender_out, 'age': age_out}


class Classify(nn.Module):
    def __init__(self, input_size=2048):
        super().__init__()
        self.gender_classify = GenderClassify(input_size)
        self.age_classify = AgeClassify(input_size)

    def forward(self, inputs):
        gender_out = self.gender_classify(inputs)
        age_out = self.age_classify(inputs)
        return {'gender': gender_out, 'age': age_out}
