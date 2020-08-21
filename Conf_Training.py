
import os  # for mkdir

#What this for?
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch    # for setting seed to make it reproducible
SEED=5  # reproducible

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
SAMPLE_EPOCH = 1  # test every xx epochs to record mu, timeresolution, and loss

EPOCHS = 3
MODEL_VERSION = '800'

WAVE_LENGTH = 120
NCHANNEL = 8

LR = 0.001 + 0.001*np.random.rand()
LRD = 0.9 + 0.1*np.random.rand()
HIDDEN_SIZE = 60 + int(10*np.random.rand())
NUM_LSTM_LAYERS = 4 #+int(2*np.random.rand())
BATCH_SIZE = 256
DROP_OUT = 0.3
WEIGHT_DECAY = 0.0004 + 0.0004*np.random.rand()  # lambda of L2 regularization


DATA_VERSION = '0710_5'



#############################################
########## INPUT/OUTPUT PATH SETTING ########
#############################################

workspace = 'C:/Users/x/OneDrive/CODING/'
print('workspace:',workspace,os.path.exists(workspace))
fileName = workspace+"/data/"+DATA_VERSION+".bin"
outputPath = workspace+DATA_VERSION+"_"+MODEL_VERSION+"_Aug20th_v1/"
if os.path.exists(outputPath) == False:
    os.mkdir(outputPath)
outputPrefix = DATA_VERSION+"_SEED"+str(SEED).zfill(4) + "_E"+str(EPOCHS)

if __name__ == '__main__':
  print('workspace:',workspace,os.path.exists(workspace))
  print('fileName:',fileName,os.path.exists(fileName))
  print('outputPath:',outputPath,os.path.exists(outputPath))

