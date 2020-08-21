
import torch.nn as nn
import torch
from Conf_Training import WAVE_LENGTH,NCHANNEL,fileName,HIDDEN_SIZE,NUM_LSTM_LAYERS,DROP_OUT

class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, common_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out


class ComNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=NCHANNEL+1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LSTM_LAYERS,
            batch_first=True,
            bias=False,
            dropout = DROP_OUT
        )
        self.out1 = nn.Linear(HIDDEN_SIZE, 1, bias=False)
        self.magnifier1 = MLP(NCHANNEL*WAVE_LENGTH, WAVE_LENGTH) # N to one


    def forward(self, x):
        x.dim
        extra1 = self.magnifier1(x.view(x.size(0),NCHANNEL*WAVE_LENGTH))
        extra1 = extra1.unsqueeze(2)
        x = torch.cat((x,extra1), 2)

        lstm_out, (h_n, h_c) = self.lstm(x, None)
        # lstm_out[:, -1, :] is the last lines of the input batches, since it's a many to one IO structure.
        out = (self.out1(lstm_out[:, -1, :]))
        return out


class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv1d( 8 ,256 ,120)
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256*256, 1)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        #x = self.fc1(x)
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

def my_mse_loss(x, y, n): 
    return torch.mean(torch.pow(torch.abs(x - y), n))
