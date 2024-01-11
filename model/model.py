import torch.nn as nn
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(5,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,256)
        self.fc5 = nn.Linear(256,10)
        self.fc6 = nn.Linear(10,1)
        self.fc7 = nn.Linear(1,1)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x)) + x
        x = self.tanh(self.fc3(x)) + x
        x = self.tanh(self.fc4(x)) + x
        x = self.tanh(self.fc5(x))
        x = self.tanh(self.fc6(x))
        x = self.fc7(x)
        return x.squeeze()
    def __call__(self, x):
        return self.forward(x)

class PinnLoss(nn.Module):
    def __init__(self,model):
        super(PinnLoss, self).__init__()
        self.model = model

    def pinn_loss(self, grid):
        num_samples_per_batch = 1000
        indices = torch.randint(0, grid.size(1), (grid.size(0), num_samples_per_batch))
        a = torch.stack([grid[i, idx] for i, idx in enumerate(indices)], dim=0)

        V_predicted = self.model(a)
        gradient = torch.autograd.grad(V_predicted.sum(), a, create_graph=True, retain_graph=True)[0]

        V_predicted_s = gradient[:,:,0]
        V_predicted_t = gradient[:,:,3]


        gradient_2 = torch.autograd.grad(V_predicted_s.sum(), a , create_graph=True, retain_graph=True)[0]

        V_predicted_ss =gradient_2[:,:,0]

        pde_error = V_predicted_t + 0.5 * a[:,:,2]**2 * a[:,:,0]**2 * V_predicted_ss + a[:,:,4] * a[:,:,0] * V_predicted_s - a[:,:,4] * V_predicted
        pde_error = torch.norm(pde_error, dim = 1)
        return pde_error

    def boundary_loss(self, batch_size):

        return 0

    def forward(self, predicted, target, grid):

        mse_loss = nn.MSELoss()(predicted, target)
        pinn_loss = self.pinn_loss(grid)
        pinn_loss = torch.mean(pinn_loss)
        boundary_loss = torch.tensor(0.0, dtype=predicted.dtype)

        loss = pinn_loss + mse_loss + boundary_loss

        return loss