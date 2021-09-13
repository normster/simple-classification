import pickle
import os
from torchvision import datasets


DATASET = '/datasets01/imagenet_full_size/061417'

traindir = os.path.join(DATASET, 'train')
valdir = os.path.join(DATASET, 'val')

traindata = datasets.ImageFolder(traindir)
valdata = datasets.ImageFolder(valdir)

with open('imagenet_train.pkl', 'wb') as f:
    pickle.dump(traindata.samples, f)

with open('imagenet_val.pkl', 'wb') as f:
    pickle.dump(valdata.samples, f)
