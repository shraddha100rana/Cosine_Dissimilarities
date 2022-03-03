import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# X is input features matrix
X = np.load('X_Input_Data.npy')
periods = X.shape[0]
nodes = X.shape[1]
features_X = X.shape[2]

# y is output features matrix
y = np.load('Y_Output_Daily.npy')
features_y = y.shape[2]

# Set device as GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Send data to device
x_tensor = torch.from_numpy(X).float().to(device)
y_tensor = torch.from_numpy(y).float().to(device)

train_data = TensorDataset(x_tensor, y_tensor)
train_loader = DataLoader(dataset = train_data, batch_size = 1)

# The Encoder captures the latent representation of the inputs
# X_{t} --> Z_{t}
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(nodes*features_X, nodes*16)
        self.fc2 = nn.Linear(nodes*16, nodes*8)
        self.fc3 = nn.Linear(nodes*8, nodes*features_y)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim = 1)
    
# The Dynamic part captures the temporal relationship of the latent representation
# We also specify a spatial relationship between the nodes which is learnt
# Z_{t+1} = g(Z_{t}*Theta0 + W*Z_{t}*Theta1)
class Dynamic(nn.Module):
    def __init__(self, hidden_size):
        super(Dynamic, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.randn(nodes, nodes, requires_grad = True, dtype = torch.float))
        self.theta0 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad = True, dtype = torch.float))
        self.theta1 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad = True, dtype = torch.float))
        
        self.gru = nn.GRU(nodes*hidden_size, nodes*hidden_size)

    def forward(self, inp, hidden):
        actual_inp = torch.matmul(inp.view(nodes, self.hidden_size), self.theta0) + torch.matmul(torch.matmul(self.W, inp.view(nodes, self.hidden_size)), self.theta1) 
        actual_inp = actual_inp.view(1, 1, nodes*self.hidden_size)
        output, hidden = self.gru(actual_inp, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, nodes*self.hidden_size, device = device)

# The Decoder predicts the output using the latent representation
# Y_{t} = d(Z_{t})
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(nodes*features_y, nodes*features_y)
        
    def forward(self, x):
        x = self.fc1(x)
        return x  
    
# The Dynamic process and the Decoding process, both affect the overall loss on a pass  

n_epochs = 100
rate = 0.01

encoder = Encoder().to(device)
dynamic = Dynamic(hidden_size = features_y).to(device)
decoder = Decoder().to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr = rate)
dynamic_optimizer = optim.SGD(dynamic.parameters(), lr = rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr = rate)

criterion = nn.MSELoss()
lam = 0.01

for epoch in range(n_epochs):
    
    # Initializing gradients to 0
    encoder_optimizer.zero_grad()
    dynamic_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    # Encoding process
    encoder_outputs = []
    encoder.train()
    
    for x_batch, y_batch in train_loader:
        
        x_batch = x_batch.to(device)
        encoder_output = encoder(x_batch.view(-1,nodes*features_X))
    
        # List of true Z_{t} 
        encoder_outputs.append(encoder_output)
    
    # Dynamic process
    Z = encoder_outputs.copy()
    
    # True Z_{t}
    Z_t = torch.stack(Z[0:-1])
    # True Z_{t+1}
    Z_t1 = torch.stack(Z[1:])
    
    Z_data = TensorDataset(Z_t, Z_t1)
    Z_loader = DataLoader(dataset = Z_data, batch_size = 1)
       
    dynamic.train()
    dynamic_loss = 0
    # Initializing hidden state of Dynamic process
    dynamic_hidden = dynamic.initHidden()
    
    for Z_t_batch, Z_t1_batch in Z_loader:
        
        Z_t_batch = Z_t_batch.to(device)
        Z_t1_batch = Z_t1_batch.to(device)

        # Calculated Z_{t+1}
        dynamic_output, dynamic_hidden = dynamic(Z_t_batch, dynamic_hidden, disaster_indicator)
    
        # Adding MSE at each time step
        dynamic_loss += criterion(Z_t1_batch, dynamic_output)
       
    # Taking average of MSE for all time steps
    dynamic_loss /= t

    # Decoder process
    decoder.train()
    decoder_loss = 0
    
    # time step
    t = 0
    for x_batch, y_batch in train_loader:
        y_batch = y_batch.to(device)

        decoder_output = decoder(encoder_outputs[t])
        t = t+1
    
        # Adding error at each time step
        decoder_loss += criterion(decoder_output.view(-1,nodes,features_y), y_batch)
    
    # Taking average of error for all time steps
    decoder_loss /= t
    
    # Both losses contribute to the updates
    loss = decoder_loss + (lam*dynamic_loss)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    dynamic_optimizer.step() 
    
    if (epoch%10 == 0):
        print(loss)

W_final = dynamic.W.detach().numpy()
np.save('Spatial_Relation_Matrix', W_final)
