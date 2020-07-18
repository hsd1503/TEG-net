import torch.nn.functional as F
from torch import nn
from tcn import TemporalConvNet
import torch

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear_T = nn.Linear(num_channels[-1], output_size)
        self.linear_G = nn.Linear(num_channels[-1], 2)
        self.linear_E = nn.Linear(25, output_size)

        self.conv = nn.Sequential(
            nn.Conv1d(25, 25, kernel_size=4, padding=0, stride=4),
            nn.BatchNorm1d(25),
            nn.ReLU(inplace=True),
            nn.Conv1d(25, 25, kernel_size=4, padding=0, stride=4),
            nn.BatchNorm1d(25),
            nn.ReLU(inplace=True),
            nn.Conv1d(25, 2, kernel_size=4, padding=2, stride=4)
            )

    def forward_T(self, inputs, inputs_reverse):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        #y2 = self.tcn(inputs_reverse) 

        y_inp = y1[:, :, -1]
        o = self.linear_T(y_inp)
        return F.log_softmax(o, dim=1)

    def forward_G(self, inputs, inputs_reverse, logits_t, logits_e):
        assert logits_t.shape == logits_e.shape
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        #y2 = self.tcn(inputs_reverse) 

        y_inp = y1[:, :, -1]
        beta = self.linear_G(y_inp)
        beta = torch.sigmoid(beta)
        
        logits_g = F.softmax(beta, dim=1)
        o = logits_t * logits_g[:,0] + logits_e * logits_g[:,1]
        
        #print (beta)
        #print (logits_g)
        return F.log_softmax(o, dim=1), beta, logits_g

    def forward_E(self, inputs, inputs_reverse, feature):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        y2 = self.conv(y1)
        y2 = y2.permute(0, 2, 1)
        #print (y2.shape)

        feature = torch.unsqueeze(feature, dim=0) #torch.Size([1, 9])    
        feature = torch.unsqueeze(feature, dim=0) #torch.Size([1, 1, 9])
        #print (feature.shape)
        
        logit_z = torch.bmm(feature, y2)
        y_inp = logit_z.contiguous().view(logit_z.size(0), -1)
        
        #o = self.linear_E(y_inp)
        #print (y_inp.shape)
        return F.log_softmax(y_inp, dim=1), y2
        
        
