
from torch.utils.data import DataLoader, Dataset
from pathlib import Path 
import struct  # for binary file reading
import torch   
from Conf_Training import WAVE_LENGTH,NCHANNEL,fileName

class Waveforms(Dataset):
    
    def __init__(self, root_dir, testRate = 0.3, train=True):
        self.root_dir = root_dir
        self.train = train
        self.file = open(root_dir, 'rb')
        self.waveLength = WAVE_LENGTH
        self.NChannel = NCHANNEL
        self.NEvents = Path(self.root_dir).stat().st_size / \
            (self.waveLength*self.NChannel+1)/4
        print('NEvents = ', self.NEvents)
        self.testRaio = testRate

    def __len__(self):
        if self.train:
            return int(self.NEvents*(1.0-self.testRaio))
        else:
            return int(self.NEvents*self.testRaio)

    def __getitem__(self, index):
        if self.train:
            self.file.seek((index+int(self.NEvents*self.testRaio))
                           * (self.NChannel*self.waveLength+1)*4, 0)
        else:
            self.file.seek((index)*(self.NChannel*self.waveLength+1)*4, 0)
        data = self.file.read(self.waveLength*self.NChannel*4)
        wave = struct.unpack(str(self.NChannel*self.waveLength)+'f', data)
        wave_tensor = torch.tensor(wave, dtype=torch.float32).reshape(
            self.NChannel, self.waveLength).t()
        data = self.file.read(4)
        tof = struct.unpack('f', data)
        tof_tensor = torch.tensor(tof, dtype=torch.float32)
        sample = {'waveform': wave_tensor, 'tof': tof_tensor}
        return sample

if __name__ == '__main__':
    
    trainWaveformDataset = Waveforms(fileName, train=True)
    print(trainWaveformDataset[0])
