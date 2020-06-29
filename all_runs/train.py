import cv2
import copy
import numpy as np

import torch

from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.optim import Adam

from model import FIG


def train_model(ntrain, dataset, iterations=1):
    net = FIG(ntrain)
    optimizer = Adam({"lr": 1e-4})
    svi = SVI(net.model, net.guide, optimizer, loss=Trace_ELBO())

    loss = 0

    for i in range(iterations):
        x, y = dataset
        loss += svi.step(x, y)

    mean_loss = loss / ntrain

    return net, mean_loss


def classification_run(folder):
    fname_label = 'class_labels.txt'

    # get file names
    with open(folder+'/'+fname_label) as f:
        content = f.read().splitlines()
    pairs = [line.split() for line in content]
    test_files  = [pair[0] for pair in pairs]
    train_files = [pair[1] for pair in pairs]
    answers_files = copy.copy(train_files)
    test_files.sort()
    train_files.sort()	
    ntrain = len(train_files)
    ntest = len(test_files)

    # load the images (and, if needed, extract features)
    train_dataset = load_dataset(train_files, ntrain)
    test_items = np.array([cv2.imread(f, 0) for f in test_files])
    test_items = torch.from_numpy(test_items).unsqueeze(1).float()
    
    model, mean_loss = train_model(ntrain, train_dataset)

    costM = []
    for i in range(ntest):
        img = test_items[i].unsqueeze(0)
        label, _ = model.infer(img)
        costM.append(torch.argmax(label))
	
    # compute the error rate
    correct = 0.0
    for i in range(ntest):
        if train_files[costM[i]] == answers_files[i]:
            correct += 1.0
    pcorrect = 100 * correct / ntest
    perror = 100 - pcorrect
    return perror


def load_dataset(fn, nds):

    data_array = []
    label_array = np.zeros((nds, nds))

    for i, (f) in enumerate(fn):
        image = cv2.imread(f, 0)
        data_array.append(image)
        label_array[i][i] = 1

    data_array = np.array(data_array)
    data_array = torch.from_numpy(data_array).unsqueeze(1).float()
    label_array = torch.from_numpy(label_array).float()
    return data_array, label_array        


if __name__ == "__main__":

    print('One Shot Generalization with a Factored Inverse Graphics Model')

    perror = np.zeros(20)
    for r in range(1, 21):
        rs = str(r)
        if len(rs)==1:
            rs = '0' + rs		
        perror[r-1] = classification_run('run'+rs)
        print(" run " + str(r) + " (error " + str(	perror[r-1] ) + "%)")		
    total = np.mean(perror)
    print(" average error " + str(total) + "%")