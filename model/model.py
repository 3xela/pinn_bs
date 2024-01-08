import torch.nn as nn
import torch

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

    def pinn_loss(self, batch_size):

        num_points = 10
        random_points = torch.rand(batch_size*num_points, 5)
        endpoint_1 = torch.tensor([0.0, 0.0, 0.0, 30.0, 0.0])
        endpoint_2 = torch.tensor([300.0, 300.0, 0.4, 180.0, 0.1])
        a = random_points * (endpoint_2 - endpoint_1) + endpoint_1
        a = a.view(batch_size, num_points, 5)
        V_predicted = self.model(a)

        gradient = torch.autograd.grad(V_predicted.sum(), a, create_graph=True, retain_graph=True)[0]

        V_predicted_s = gradient[:,0]
        V_predicted_t = gradient[:,3]


        gradient_2 = torch.autograd.grad(V_predicted_s.sum(), a , create_graph=True, retain_graph=True)[0]

        V_predicted_ss =gradient_2[:,0]

        pde_error = V_predicted_t+0.5*a[:,2]**2*a[:,0]**2*V_predicted_ss + a[:,4]*a[:,0]*V_predicted - a[:,4]*V_predicted
        return pde_error

    def forward(self, predicted, target, batch_size):

        mse_loss = nn.MSELoss()(predicted, target)
        pinn_loss_integral = 0

        boundary_loss = torch.tensor(0.0, dtype=predicted.dtype)

        loss = pinn_loss_integral + mse_loss + boundary_loss

        return loss