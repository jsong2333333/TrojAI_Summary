import torch
import torch.nn as nn
import math

def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda')
        print("Connected to a GPU")
    else:
        print("Using the CPU")
        device = torch.device('cpu')
    return device

class tauGRU_Cell(nn.Module):
    def __init__(self, ninp, nhid, dt, alpha, beta):
        super(tauGRU_Cell, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.alpha = alpha
        self.beta = beta

        self.dt = dt
        
        self.W1 = nn.Linear(nhid, nhid)
        self.W2 = nn.Linear(nhid, nhid)
        self.W3 = nn.Linear(nhid, nhid)
        self.W4 = nn.Linear(nhid, nhid)

        self.U1 = nn.Linear(ninp, nhid)
        self.U2 = nn.Linear(ninp, nhid)
        self.U3 = nn.Linear(ninp, nhid)       
        self.U4 = nn.Linear(ninp, nhid)       
        
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.nhid)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h, x_delay, h_delay):
        # v1.3
        C = self.dt * torch.sigmoid(self.W3(h) + self.U3(x))
        D = torch.sigmoid(self.W4(h_delay) + self.U4(x_delay))
        A = torch.tanh(self.W1(h) + self.U1(x)) 
        B = torch.tanh(self.W2(h_delay) + self.U2(x_delay))
        
        if self.eval():
            h = (1-C) * h + C * (self.beta * A + self.alpha * D * B)  

        else:
            add_noise = 0.5 * torch.randn(h.shape[0], h.shape[1]).float().to(get_device())
            mult_noise = 0.05 * torch.rand(h.shape[0], h.shape[1]).float().to(get_device()) + (1-0.05)
        
            h = (1-C) * h + mult_noise * C * (self.beta * A + self.alpha * D * B ) + add_noise

        return h

class tauGRU(nn.Module):
    def __init__(self, ninp, nhid, nout, dt=1.0, tau=0, alpha=1, beta=1, drop=0):
        super(tauGRU, self).__init__()
        self.nhid = nhid
        self.ninp = ninp    
        self.tau = tau

        self.cell = tauGRU_Cell(ninp, nhid, dt, alpha, beta)
        self.classifier = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(drop)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)

    def forward(self, input):
        ## initialize hidden states
        h = input.data.new(input.size(1), self.nhid).zero_()
        h_delay = input.data.new(input.size(1), self.nhid).zero_()
        x_delay = input.data.new(input.size(1), self.ninp).zero_()
        x_history = []
        h_history = []
        
        for t, x in enumerate(input):
            
            if self.tau == 0:
                h = self.cell(self.dropout(x), self.dropout(h), self.dropout(x), self.dropout(h))            
            else:
                h = self.cell(self.dropout(x), self.dropout(h), self.dropout(x_delay), self.dropout(h_delay))



            h_history.append(h)
            x_history.append(x)
            if t >= self.tau:
                h_delay = h_history[-self.tau]
                x_delay = x_history[-self.tau]

        out = self.classifier(h)
        return out       

         
        
class LEMCell(nn.Module):
    def __init__(self, ninp, nhid, dt):
        super(LEMCell, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.dt = dt
        self.inp2hid = nn.Linear(ninp, 4 * nhid)
        self.hid2hid = nn.Linear(nhid, 3 * nhid)
        self.transform_z = nn.Linear(nhid, nhid)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.nhid)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, y, z):
        transformed_inp = self.inp2hid(x)
        transformed_hid = self.hid2hid(y)
        i_dt1, i_dt2, i_z, i_y = transformed_inp.chunk(4, 1)
        h_dt1, h_dt2, h_y = transformed_hid.chunk(3, 1)

        ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1)
        ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2)

        z = (1.-ms_dt) * z + ms_dt * torch.tanh(i_y + h_y)
        y = (1.-ms_dt_bar)* y + ms_dt_bar * torch.tanh(self.transform_z(z)+i_z)

        return y, z

class LEM(nn.Module):
    def __init__(self, ninp, nhid, nout, dt=1.):
        super(LEM, self).__init__()
        self.nhid = nhid
        self.cell = LEMCell(ninp,nhid,dt)
        self.classifier = nn.Linear(nhid, nout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)

    def forward(self, input):
        ## initialize hidden states
        y = input.data.new(input.size(1), self.nhid).zero_()
        z = input.data.new(input.size(1), self.nhid).zero_()
        for x in input:
            y, z = self.cell(x,y,z)
        out = self.classifier(y)
        return out             
