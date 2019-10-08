from os import listdir
from sklearn import manifold, datasets
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader , TensorDataset
import glob
import os
import os.path
import sys
import string
import matplotlib
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import time
