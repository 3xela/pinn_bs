from model import Model, PinnLoss
import torch.optim as optim
from data_generation import MyDataset, data, values
from torch.utils.data import Dataset, DataLoader


model = Model()
Loss = PinnLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
batch_size =10
training_data = MyDataset(data, values)
dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)


for epoch in range(epochs):
    for batch_data, batch_targets in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_data)

        loss = Loss(outputs, batch_targets)
        Loss.forward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {Loss.item()}')