#IMPORT LIBRARIES
import torch as _torch
import torch.nn as _nn
import numpy as _np
import os as _os

from ...sqi import _SignalQualityIndicator
# from torch.utils.data import Dataset as _Dataset

FSAMP = 10
LENSECONDS = 20
NSAMP = 200
DN = 50

device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
_path = _os.path.join(_os.path.dirname(__file__))
# print(_path)
WEIGHTSFILE = f'{_path}/_dlweights/weights.pth'

def _normalize(x):
    if _np.std(x) != 0:
        return( (x - _np.mean(x)) / _np.std(x) )
    else:
        return(x - _np.mean(x))
    
class _Simple(_nn.Module):
    def __init__(self, n_classes=2):
        super(_Simple, self).__init__()
        self.n_classes = n_classes
        self.conv_branch= _nn.Sequential(_nn.BatchNorm1d(2),
                                         _nn.Conv1d(2, 64, 21), _nn.BatchNorm1d(64), _nn.ReLU(), _nn.MaxPool1d(2),
                                         _nn.Conv1d(64, 128, 21), _nn.BatchNorm1d(128), _nn.ReLU(), _nn.MaxPool1d(2),
                                         _nn.AdaptiveAvgPool1d(10))
        self.linear= _nn.Sequential(_nn.Linear(1280, 1000), _nn.ReLU(),
                                    _nn.Linear(1000, 1000), _nn.ReLU(),
                                    _nn.Linear(1000, self.n_classes), _nn.Softmax(1))

        state_dict = _torch.load(WEIGHTSFILE, map_location=device)
        self.load_state_dict(state_dict)
        
    def forward(self, x):
        x_feat = self.conv_branch(x)
        x_feat = x_feat.view(x_feat.shape[0], -1)
        x_out = self.linear(x_feat)
        return(x_out)

    def extract_features(self, x):
        x_feat = self.conv_branch(x)
        x_feat = x_feat.view(x_feat.shape[0], -1)
        return(x_feat)
    
#%%
class SignalQualityDeepLearning(_SignalQualityIndicator):
    def __init__(self, threshold=[0.5, 1.5]):
        model = _Simple(n_classes=2)
        model = model.to(device)
        model.eval()
        self.model = model
        _SignalQualityIndicator.__init__(self, threshold=threshold)
        self.dimensions = {'time' : 1, 'component': 1}
        
    
    def algorithm(self, signal):
        assert (signal.p.get_duration() - 20) < 1/FSAMP
        signal = signal.p.resample(10)
        signal_values = signal.p.get_values()[:,0,:]
        signal_values = _normalize(signal_values)
        
        signal_in = signal_values[[-1],:] * _np.ones((200, 2))
        signal_in[:len(signal_values)] = signal_values
        signal_in = _torch.tensor(signal_in.T).float()
        
        output = self.model.forward(signal_in.unsqueeze(0).to(device))
        _, quality = _torch.max(output,1)
        quality = quality.cpu().numpy()
        return(_np.array([[quality]]))
