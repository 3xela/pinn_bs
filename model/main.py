from model import Model, PinnLoss
import torch.optim as optim
from data_generation import MyDataset, data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

epochs = 2000
batch_size =5

my_model = Model().to(device)
pinnloss = PinnLoss(my_model).to(device)
optimizer = optim.Adam(my_model.parameters(), lr=0.001)

training_data = MyDataset(data)
dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)

num_points_per_dim = 10

endpoint_1 = torch.tensor([0.0, 0.0, 0.0, 30.0, 0.0])
endpoint_2 = torch.tensor([300.0, 300.0, 0.4, 180.0, 0.1])

dim_points = [
    torch.linspace(start, end, num_points_per_dim) for start, end in zip(endpoint_1, endpoint_2)
]
grid_points = torch.meshgrid(*dim_points)
grid = torch.stack(grid_points, dim=-1).view(batch_size, -1, 5)
grid.requires_grad_(True)
grid = grid.to(device)


loss_plot = []

model_folder = 'C:/Users/alexa/PycharmProjects/pinn_bs/trained models'
model_filename = 'my_model.pt'
model_path = os.path.join(model_folder, model_filename)


for epoch in range(epochs):
    running_loss = 0.0
    for x_batch, y_batch in dataloader:
        prediction = my_model(x_batch)

        my_loss = pinnloss(prediction, y_batch, grid)

        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
        loss_plot.append(my_loss.item())
        running_loss += my_loss.item()

    average_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}')
    if epoch % 10 == 0:
        torch.save(my_model.state_dict(), model_path)

plt.plot(range(1, epochs + 1), loss_plot)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()