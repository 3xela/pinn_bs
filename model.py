import torch.nn as nn
import torch
from torch.autograd.functional import jacobian
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(5,50)
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
        return x.view(-1)
class PinnLoss(nn.Module):
    def __init__(self):
        super(PinnLoss, self).__init__()

    def pinn_loss(self, V, a):

        jacobian_matrix = jacobian(V,inputs = a, create_graph=True)

        V_s = jacobian_matrix[:,:,:,0]
        V_t = jacobian_matrix[:,:,:,3]

        jacobian_matrix_V_ss = jacobian(lambda x: x, inputs=(V_s,))
        V_ss = jacobian_matrix_V_ss[:,:,:,0]

        pde_error = V_t+0.5*a[:,2]**2*a[:,0]**2*V_ss + a[:,4]*a[:,0]*V_s - a[:,4]*V

        return pde_error


    def integrate_squared_norm(self, V, a, domain):
        pinn_loss_values = self.pinn_loss(V, a)
        delta_x = (domain[:, -1] - domain[:, 0]) / (len(domain) - 1)
        integral = delta_x * (
                    (pinn_loss_values ** 2) - 0.5 * (pinn_loss_values[0] ** 2 + pinn_loss_values[-1] ** 2))
        return integral
    def forward(self, predicted, target, model,a,domain, batch_size):
        # Define your custom loss computation here
        mse_loss = nn.MSELoss()(predicted, target)

        squared_norm_integral = self.integrate_squared_norm(model(a), a, domain)

        boundary_loss = torch.tensor(0.0, dtype=predicted.dtype)

        loss = squared_norm_integral + mse_loss + boundary_loss
        loss = torch.mean(loss)
        return loss

#model_t + 0.5*sigma**2 * S**2 *model_SS + r*S* V_S - r*model