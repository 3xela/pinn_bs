import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(6,50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50,10)
        self.fc5 = nn.Linear(10,1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x)) + x
        x = self.relu(self.fc3(x)) + x
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))

        return x
class PinnLoss(nn.Module):
    def __init__(self):
        super(PinnLoss, self).__init__()
    def pinn_loss(self, V,a):
        V_S = torch.autograd.grad(V, a[0], create_graph=True)[0]
        V_SS = torch.autograd.grad(V_S, a[0])
        V_T = torch.autograd.grad(V, a[3])

        pde_error = V_T + 0.5*a[2]**2*a[0]**2*V_SS + a[4]*a[0]*V_S - a[4]*V
        return pde_error

    def forward(self, predicted, target, model,a):
        # Define your custom loss computation here
        loss = torch.norm((predicted - target), p=2) +  torch.norm(self.pinn_loss(model,a ), p = 2)
        return loss

#model_t + 0.5*sigma**2 * S**2 *model_SS + r*S* V_S - r*model