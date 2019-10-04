import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# set current path
os.chdir(os.path.split(os.path.realpath(__file__))[0])
# clear the file
path = 'image_output/'
for filename in os.listdir(path):
    os.remove(path + filename)

# create a dataset
class Dataset(data.Dataset):
    def __init__(self, clz_num):
        super(Dataset, self).__init__()
        self.clz_num = clz_num
        self.class_num = len(clz_num)

        total_len = 0
        self.data = torch.zeros(0, 2)
        self.label = torch.zeros(0, 1)
        for ii in range(self.class_num):
            total_len = total_len + int(clz_num[ii])
            data_size = torch.ones(int(clz_num[ii]), 2)
            data = torch.normal( 2 * data_size * (ii - self.class_num * 0.5), 0.5)
            clz  = int(clz_num[ii])
            label = torch.full((clz, 1), ii).long()
            self.data = torch.cat((self.data, data), 0).type(torch.FloatTensor)
            self.label = torch.cat((self.label.long(), label.long())).type(torch.LongTensor)
        
        label_ = self.label.clone()
        label_ = torch.zeros_like(torch.max(self.label, 1)[1])
        for ii in range(total_len):
            label_[ii] = self.label[ii]
        self.label = label_.clone()

        plt.scatter(self.data.numpy()[:, 0], self.data.numpy()[:, 1], c=self.label, s=100, lw=0)
        plt.title("dataset")
        plt.show()
        
    def __getitem__(self, item):
        return self.data[item], self.label[item]
    
    def __len__(self):
        return len(self.label)
    
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.linear = nn.Linear(n_input, 10)
        self.relu = F.relu
        self.output = nn.Linear(10, n_output)
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.output(x)
        return x

        
# set up 5 classes
class_num = 5
clz_num = torch.zeros(class_num)
for ii in range(class_num):
    clz_num[ii] = 100

# load data
dataset = Dataset(clz_num)
data_loader = data.DataLoader(dataset=dataset, batch_size=500, shuffle=False)

# net: input = 2, output = 5
net = Net(n_input=2, n_output=class_num).cuda()

# use SGD optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_function = torch.nn.CrossEntropyLoss()

losses = []

for epoch in range(400):
    for _, train_data in enumerate(data_loader):
        data = train_data[0].cuda()
        label = train_data[1].cuda()

        pred = net(data)

        loss = loss_function(pred,  label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)

    if epoch % 25 == 0:
        pred_label = torch.max(pred, 1)[1]
        pred_label = pred_label.cpu().detach().numpy()
        target_label = label.cpu().numpy()
        plt.scatter(data.cpu().numpy()[:,0], data.cpu().numpy()[:,1], c=pred_label, s=100, lw=0)
        accuracy = float((pred_label == target_label).astype(int).sum()) / float(target_label.size)
        plt.text(1.5, -4, "acc=%.2f" % accuracy, fontdict={'size':20, 'color':'red'})
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("the " + str(epoch) + "th training" + ", acc=%.2f" % accuracy)
        plt.savefig(path + str(epoch) + "_plt.jpg")
        plt.show()

batch_nums = range(1, len(losses) + 1)
plt.plot(batch_nums, losses)
plt.title("loss - Batch")
plt.xlabel("batch")
plt.ylabel("loss")
plt.savefig(path + "loss.jpg")
plt.show()