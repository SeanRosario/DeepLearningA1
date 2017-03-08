
# coding: utf-8

# In[1]:
from __future__ import print_function

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from skimage.transform import rotate
from skimage.filters import gaussian
#get_ipython().magic(u'pylab inline')


# In[4]:

trainset_imoprt = pickle.load(open("train_labeled.p", "rb"))
trainunl_imoprt = pickle.load(open("train_unlabeled.p", "rb"))
validset_import = pickle.load(open("validation.p", "rb"))
test_import = pickle.load(open("test.p", "rb"))


# In[5]:

train_loader = torch.utils.data.DataLoader(trainset_imoprt, batch_size=64, shuffle=True)
semi_loader = torch.utils.data.DataLoader(trainunl_imoprt, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset_import, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_import, batch_size=64, shuffle=True)


# In[6]:

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(5120, 500)
        self.fcmid = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2))
        x = F.upsample_bilinear(x, size=(16, 16))
        x = F.leaky_relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.upsample_bilinear(x, size=(16, 16))
        x = F.leaky_relu(F.max_pool2d(self.conv3(x), 2))

        x = F.upsample_bilinear(x, size=(16, 16))
        x = x.view(-1, 5120)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.leaky_relu(self.fcmid(x))
        x = F.dropout(x, training=self.training)
        x = F.leaky_relu(self.fc2(x))
        return F.log_softmax(x)

model = Net()


# In[7]:

#optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
#optimizer = optim.SGD(model.parameters(), lr=0.0005)
optimizer = optim.SGD(model.parameters(), lr=0.005,momentum=0.95)


# In[8]:

def train(epoch):
    model.train()
    log_interval = 10.
    for batch_idx, (data, target) in enumerate(train_loader):
        if False:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.squeeze())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return 100. * correct / len(valid_loader.dataset)


# In[11]:

result_train = []
result_valid = []
result_test = []
for epoch in range(1, 400):
    train(epoch)
    if epoch%10 == 0:
        torch.save(model,open('model1.p','wb'))
    result_train.append(test(epoch, train_loader))
    result_valid.append(test(epoch, valid_loader))


# In[14]:

#plt.style.use('ggplot')

#plt.figure(figsize=(10, 6))
#plt.plot(result_train,label = 'Validation')
#plt.title('Training accuracy accross epochs')
#plt.plot(result_train,label = 'Training accuracy')
#plt.plot(result_valid,label = 'Validation accuracy')
#plt.legend(loc=2)
#plt.xlabel('Number of Epochs')
#plt.ylabel('Accuracy')
#plt.show()
#plt.savefig("Base Model with pReLu and UpSampling.png")


# In[27]:

label_predict = np.array([])
model.eval()
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    temp = output.data.max(1)[1].numpy().reshape(-1)
    label_predict = np.concatenate((label_predict, temp))


# In[28]:

predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
predict_label.reset_index(inplace=True)
predict_label.rename(columns={'index': 'ID'}, inplace=True)


# In[29]:

predict_label.to_csv('lrelu_up_submission.csv', index=False)


# In[30]:

#predict_label.head()


# In[31]:

#print("Final validation accuracy",test(1, valid_loader))


# In[ ]:

#print("Final test accuracy",test(1, test_loader))

with open('pickle/best_train.pkl', 'wb') as f1:
    pickle.dump(result_train, f1)

with open('pickle/best_test.pkl', 'wb') as f2:
    pickle.dump(result_valid, f2)

with open('pickle/best_model.pkl', 'wb') as f3:
    pickle.dump(model,f3)

