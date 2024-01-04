import torch.nn as nn
import torch
from torch.autograd.functional import jacobian
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(5,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,100)
        self.fc4 = nn.Linear(100,10)
        self.fc5 = nn.Linear(10,1)
        self.fc6 = nn.Linear(1,1)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x)) + x
        x = self.tanh(self.fc3(x)) + x
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
        x = self.fc6(x)
        return x.squeeze()
    def __call__(self, x):
        return self.forward(x)

class PinnLoss(nn.Module):
    def __init__(self,model):
        super(PinnLoss, self).__init__()
        self.model = model

    def pinn_loss(self, V, a):

        jacobian_matrix = jacobian(V, inputs=a, create_graph=True)
        summed_matrix = torch.sum(jacobian_matrix, dim=1)

        V_s = summed_matrix[:,0]
        V_t = summed_matrix[:,3]

        c = 0.01
        b = a.clone()
        b[:, 0] += c

        jacobian_matrix_2 = jacobian(V,inputs = b, create_graph=True)

        summed_matrix_2 = torch.sum(jacobian_matrix_2, dim = 1)
        V_s_ds = summed_matrix_2[:,0]

        V_ss = (V_s_ds - V_s)/c

        pde_error = V_t+0.5*a[:,2]**2*a[:,0]**2*V_ss + a[:,4]*a[:,0]*V_s - a[:,4]*self.model(a)
        print(pde_error.shape)
        return pde_error
    def integrate_squared_norm(self, V, domain):

        batch_size, num_points, _ = domain.size()
        pinn_loss_values = self.pinn_loss(V, domain)
        print(pinn_loss_values.shape)
        integral_norm = torch.sum(pinn_loss_values ** 2)
        integral = integral_norm / (batch_size * num_points)
        return integral
    def forward(self, predicted, target,domain):

        mse_loss = nn.MSELoss()(predicted, target)

        pinn_loss_integral = self.integrate_squared_norm(self.model, domain)
        boundary_loss = torch.tensor(0.0, dtype=predicted.dtype)

        loss = pinn_loss_integral + mse_loss + boundary_loss
        loss = torch.mean(loss)
        return loss