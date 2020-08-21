
import time  # for time recording
import matplotlib.pyplot as plt  # for plotting
import matplotlib.mlab as mlab  # for plotting

from torch.utils.data import DataLoader
from torch.autograd import Variable
from scipy.stats import norm
import math 

import data_loader
from Conf_Training import BATCH_SIZE,SAMPLE_EPOCH,outputPath,outputPrefix

import Model_Training



##TODO:WAIT XIANGYU TO FIX THIS PAER##
'''
def testDataset(network, dataVersion, outputPrefix, input_prefix = "./Codes/V0.70/Data/"):
    
    inputFileName = input_prefix+dataVersion+".bin"
    testWaveformDataset = data_loader.Waveforms(inputFileName, testRate = 1.0, train=False)
    testData = DataLoader(testWaveformDataset, batch_size=BATCH_SIZE, shuffle=False)

    residualTofList = []
    predictedTofList = []
    labelTofList = []
    print("Test Data: " + dataVersion)
    mu=sigma=0

    with torch.no_grad( ):
        for index, batch_data in enumerate(testData):
            input = Variable(batch_data['waveform'])#.cuda()
            b_y = Variable(batch_data['tof']).squeeze()#.cuda().squeeze()
            output = Model_Training.comNN(input).squeeze()
            residualTofList.extend(
                (output-b_y).cpu().detach().numpy().tolist())
            predictedTofList.extend(output.cpu().detach().numpy().tolist())
            labelTofList.extend(b_y.cpu().detach().numpy().tolist())

    (mu, sigma) = norm.fit(residualTofList)
    n, bins, patches = plt.hist(residualTofList, 100, range=(-1, 1), density=2)
    # add a 'best fit' line
    y = norm.pdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel('Tof residual [ns]')
    plt.title(
        r'$\mathrm{Tof\ residual:}\ \mu=%.3f ns,\ \sigma/\sqrt{2}=%.3f ns$' % (mu, sigma/math.sqrt(2)))
    plt.grid(True)
    plt.savefig(outputPath + outputPrefix + "Test_" + dataVersion + ".png")
    plt.clf()
'''

def plot_output(testLoss_his,trainLoss_his,residualTofList,timeResolution_his,mu_his):

    plt.plot(testLoss_his, label='testLoss')
    plt.plot(trainLoss_his, label='trainLoss')
    plt.grid(True)
    # plt.ylim([0.01, 0.08])
    # plt.ylabel("MSE [ns^2]")
    plt.title("Loss/MSE Vs Epochs")
    plt.xlabel("sampled every "+str(SAMPLE_EPOCH)+" epochs")
    plt.legend()
    plt.savefig(outputPath+outputPrefix+"_Loss.png")
    plt.clf()

    plt.plot(mu_his)
    plt.grid(True)
    plt.title("Average predicted ToF Vs Epochs")
    plt.ylabel("Average of predicted ToF [ns]")
    plt.ylim([-0.1, 0.1])
    plt.xlabel("sampled every "+str(SAMPLE_EPOCH)+" epochs")
    plt.savefig(outputPath+outputPrefix+"_Mu.png")
    plt.clf()

    plt.plot(timeResolution_his)
    plt.ylabel("Time Resolution [ns]")
    plt.grid(True)
    plt.ylim([0.05, 0.30])
    plt.title("TimeResolution Vs Epochs")
    plt.xlabel("sampled every "+str(SAMPLE_EPOCH)+" epochs")
    plt.savefig(outputPath+outputPrefix+"_TR.png")
    plt.clf()

    (mu, sigma) = norm.fit(residualTofList)
    n, bins, patches = plt.hist(residualTofList, 100, range=(-1, 1), density=1)
    # add a 'best fit' line
    y = norm.pdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel('Tof residual [ns]')
    # plt.ylabel('Probability')
    plt.title(r'$\mathrm{Tof\ residual:}\ \mu=%.3f ns,\ \sigma/\sqrt{2}=%.3f ns$' %(mu, sigma/math.sqrt(2)))
    plt.grid(True)
    plt.savefig(outputPath+outputPrefix+"_TofHist.png")
    plt.clf()