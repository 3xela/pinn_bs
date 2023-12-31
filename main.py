from model import Model, PinnLoss
import torch
import torch.optim as optim
from data_generation import MyDataset, data
from torch.utils.data import DataLoader

endpoint_1 = torch.tensor([0.0, 0.0, 0.0, 30.0, 0.0])
endpoint_2 = torch.tensor([300.0, 300.0, 0.4, 180.0, 0.1])

num_points = 10
grid = torch.meshgrid([torch.linspace(endpoint_1[i], endpoint_2[i], num_points) for i in range(5)])

domain = torch.stack([tensor.flatten() for tensor in grid], dim=1)

model = Model()
pinnloss = PinnLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
batch_size =10

training_data = MyDataset(data)
dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    running_loss = 0.0
    for x_batch, y_batch in dataloader:
        prediction = model(x_batch)

        loss = pinnloss(prediction, y_batch, model, x_batch, domain, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    average_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}')