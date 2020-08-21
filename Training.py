
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
import torch

from scipy.stats import norm 
import math

from data_loader import Waveforms
import Model_Training

from Conf_Training import BATCH_SIZE,WEIGHT_DECAY,SEED,LR,LRD,SAMPLE_EPOCH
from Conf_Training import fileName,DATA_VERSION,EPOCHS,outputPath,outputPrefix

from progressbar import ProgressBar

def excute():

    trainWaveformDataset = Waveforms(fileName, train=True)
    testWaveformDataset = Waveforms(fileName, train=False)
    trainData = DataLoader(trainWaveformDataset, batch_size=BATCH_SIZE, shuffle=True)
    testData = DataLoader(testWaveformDataset, batch_size=BATCH_SIZE, shuffle=False)

    comNN = Model_Training.ComNN()

    use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
    if use_gpu:
        comNN = comNN.cuda()
        

    print(comNN)

    optimizer = optim.Adam(comNN.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LRD)
    print(optimizer)
    # loss_func = nn.MSELoss().cuda()
    loss_func = Model_Training.my_mse_loss

    testLoss_his = []
    trainLoss_his = []
    timeResolution_his = []
    mu_his = []
    residualTofList = []
    predictedTofList = []
    labelTofList = []

    print("Data: " + DATA_VERSION + "; SEED = " + str(SEED)+"； Epochs = " + str(EPOCHS))
    progress = ProgressBar()
    for epoch in progress(range(EPOCHS)):
        epoch_loss = 0  # for LR decay rate scheduler
        for index, batch_data in enumerate(trainData):
            input = Variable(batch_data['waveform']).squeeze()#.cuda()
            b_y = Variable(batch_data['tof']).squeeze()#.cuda().squeeze()
            # print(input.size())
            output = comNN(input).squeeze()
            print(len(output),len(b_y),"1")
            loss = loss_func(output, b_y, 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data.cpu().numpy()
        scheduler.step(epoch_loss)

        # to store the sigma, mu, loss
        if (epoch) % SAMPLE_EPOCH == 0:
            trainLoss_his.append(loss.data.cpu().numpy())
            residualTofList.clear()
            predictedTofList.clear()
            labelTofList.clear()
            for index, batch_data in enumerate(testData):
                input = Variable(batch_data['waveform'])#.cuda().cuda()
                b_y = Variable(batch_data['tof']).squeeze()#.cuda().cuda().squeeze()
                output = comNN(input).squeeze()
                loss = loss_func(output, b_y, 2)
                residualTofList.extend(
                    (output-b_y).cpu().detach().numpy().tolist())
                predictedTofList.extend(output.cpu().detach().numpy().tolist())
                labelTofList.extend(b_y.cpu().detach().numpy().tolist())
                # print('output length: ',output.cpu().detach().numpy().tolist().__len__())
            # print('toflist length: ',tofList.__len__())
            (mu, sigma) = norm.fit(residualTofList)
            timeResolution_his.append(sigma/math.sqrt(2))
            mu_his.append(mu)
            testLoss_his.append(loss.data.cpu().numpy())


    # save model
    torch.save(comNN, outputPath+outputPrefix)
    import Plot_output
    Plot_output.plot_output(testLoss_his,trainLoss_his,residualTofList,timeResolution_his,mu_his)

if __name__ == '__main__':

    excute()