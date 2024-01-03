import torch.nn as nn
import torch
from torch.autograd.functional import jacobian
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(5,50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50,10)
        self.fc5 = nn.Linear(10,1)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x)) + x
        x = self.tanh(self.fc3(x)) + x
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
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

        return pde_error
    def integrate_squared_norm(self, V, domain):
        num_samples = 100
        # 50 = number of points along each dimension
        sampled_indices = [torch.randint(0, 50, (num_samples,), dtype=torch.long) for _ in range(5)]

        sampled_points = [domain[i][sampled_indices[i]] for i in range(5)]

        sampled_points = torch.stack(sampled_points, dim=1)

        integral = nn.MSELoss()(self.model(sampled_points) , 0)

        return integral
    def forward(self, predicted, target,domain):

        mse_loss = nn.MSELoss()(predicted, target)

        squared_norm_integral = self.integrate_squared_norm(self.model, domain)
        boundary_loss = torch.tensor(0.0, dtype=predicted.dtype)

        loss = squared_norm_integral + mse_loss + boundary_loss
        loss = torch.mean(loss)
        return loss

#model_t + 0.5*sigma**2 * S**2 *model_SS + r*S* V_S - r*model