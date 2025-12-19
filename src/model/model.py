import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from src.config.constants import BETA, HIDDEN_SIZE

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class TaskSNN(nn.Module):
    def __init__(self, num_inputs, hidden_size=HIDDEN_SIZE, dropout_p=0.4):
        super().__init__()
        
        spike_grad = surrogate.fast_sigmoid()
        beta = BETA 
        
        self.noise_std = 0.08
        
        self.input_bn = nn.BatchNorm1d(num_inputs)
        
        self.fc_in = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.rlif1 = snn.RLeaky(beta=beta, spike_grad=spike_grad, linear_features=hidden_size)
        self.dropout1 = nn.Dropout(dropout_p)
        
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.rlif2 = snn.RLeaky(beta=beta, spike_grad=spike_grad, linear_features=hidden_size)
        self.dropout2 = nn.Dropout(dropout_p)

        self.fc_out = nn.Linear(hidden_size, 4)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, 
                              output=True, reset_mechanism='none') 

    def forward(self, x):
        x = x.permute(1, 0, 2)
        
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
    
        input_device = x.device
        spk1, mem1 = self.rlif1.init_rleaky()
        spk1, mem1 = spk1.to(input_device), mem1.to(input_device)
        spk2, mem2 = self.rlif2.init_rleaky()
        spk2, mem2 = spk2.to(input_device), mem2.to(input_device)
        mem_out = self.lif_out.init_leaky()
        mem_out = mem_out.to(input_device)
        
        mem_out_rec = []
        
        for step in range(x.size(0)):
            cur_step = x[step]
            cur_step = self.input_bn(cur_step)
            
            cur_input = self.fc_in(cur_step)
            cur_input = self.ln1(cur_input)
            
            spk1, mem1 = self.rlif1(cur_input, spk1, mem1)
            spk1_d = self.dropout1(spk1)
            
            res1 = spk1_d + cur_input
            
            cur2 = self.fc_hidden(res1)
            cur2 = self.ln2(cur2) 
            spk2, mem2 = self.rlif2(cur2, spk2, mem2)
            spk2_d = self.dropout2(spk2)
            
            res2 = spk2_d + res1
            
            cur_out = self.fc_out(res2)
            _, mem_out = self.lif_out(cur_out, mem_out)
            
            mem_out_rec.append(mem_out)
    
        return torch.stack(mem_out_rec, dim=0).mean(dim=0)
