from model import Model, PinnLoss
import torch
import torch.optim as optim
from data_generation import MyDataset, data
from torch.utils.data import DataLoader

epochs = 1000
batch_size =10

endpoint_1 = torch.tensor([0.0, 0.0, 0.0, 30.0, 0.0])
endpoint_2 = torch.tensor([300.0, 300.0, 0.4, 180.0, 0.1])

num_points = 50
domain = torch.meshgrid([torch.linspace(endpoint_1[i], endpoint_2[i], num_points) for i in range(5)])

print(domain)

my_model = Model()
pinnloss = PinnLoss(my_model)
optimizer = optim.Adam(my_model.parameters(), lr=0.001)

training_data = MyDataset(data)
dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    running_loss = 0.0
    for x_batch, y_batch in dataloader:
        prediction = my_model(x_batch)

        my_loss = pinnloss(prediction, y_batch, domain)

        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()

        running_loss += my_loss.item()

    average_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}')