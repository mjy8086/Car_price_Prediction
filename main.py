import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# making model
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        # self.fc3 = nn.Linear(hidden_features, hidden_features)
        self.fc4 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        # h3 = F.leaky_relu(self.fc3(h1))
        out = self.fc4(h2)

        return out


class Dataset(Dataset):
    def __init__(self, dataset):
        self.data = pd.read_csv('data/' + dataset + '.csv')

        encoder = LabelEncoder()

        self.data['model'] = encoder.fit_transform(self.data['model'])
        self.data['transmission'] = encoder.fit_transform(self.data['transmission'])
        self.data['fuelType'] = encoder.fit_transform(self.data['fuelType'])

        self.x_train = self.data.drop('price', axis=1)
        self.y_train = self.data['price']

        scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
        self.x_train = scaler.fit_transform(self.x_train)
        self.y_train = self.y_train.values

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = torch.Tensor(self.x_train[idx])
        y = torch.Tensor(self.y_train)[idx]

        return x, y


batch_size = 256
in_features = 8
hidden_features = 128
out_features = 1
lr = 0.001
weight_decay = 1e-7
n_epochs = 500

if __name__ == '__main__':
    train_data = Dataset('train')
    val_data = Dataset('val')
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    model = MultivariateLinearRegressionModel(in_features, hidden_features, out_features)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs + 1):
        train_loss = 0.0
        train_mae = 0
        for batch_idx, (x_train, y_train) in enumerate(trainloader):
        # for batch_idx, (x_train, y_train) in tqdm(enumerate(trainloader)):

            pred_train = model(x_train.cuda())
            loss = F.mse_loss(pred_train, y_train.cuda()).cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += np.sqrt(loss.item())
            train_mae += torch.abs(pred_train - y_train).sum() / batch_size
            # train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_mae = 0
            for x_val, y_val in validloader:
                pred_val = model(x_val.cuda())
                loss = F.mse_loss(pred_val, y_val.cuda())
                valid_loss += np.sqrt(loss.item())
                valid_mae += torch.abs(pred_val - y_val).sum() / batch_size

        if epoch % 10 == 0:
            print('Epoch {:4d}/{} Train_Loss: {:.6f} Val_Loss: {:.6f}'.format(
                epoch, n_epochs, train_loss / len(trainloader), valid_loss / len(validloader)
            ))
            print('Train_MAE: {:.6f} Val_MAE: {:.6f}'.format(
                train_mae / len(trainloader), valid_mae / len(validloader)
            ))