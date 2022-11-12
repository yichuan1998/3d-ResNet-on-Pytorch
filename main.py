from tqdm import tqdm
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch
from model.ResNet3D import BasicBlock, Bottleneck, ResNet3D
from time import time
from dataset.Toy_example import Toy_example
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

ResNet_layers = {"ResNet18":[2, 2, 2, 2], "ResNet34":[3, 4, 6, 3], "ResNet50":[3, 4, 6, 3],
                 "ResNet101":[3, 4, 23, 3], "ResNet152":[3, 8, 36, 3]}


X, y = make_circles(n_samples=10000, noise=0.05, random_state=26)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=26)
train_data = Toy_example(X_train, y_train)
dataset_train = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)

test_data = Toy_example(X_test, y_test)
dataset_test = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)


num_epochs = 100
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = ResNet3D(block=BasicBlock, layers=ResNet_layers["ResNet18"], data_in_channels=1, num_classes=2)
model.to(device)

softmax_func = nn.Softmax(dim=1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.005, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1, last_epoch=-1)
scaler = GradScaler()
writer = SummaryWriter(log_dir=r"E:\Python_file\3d_ResNet\log", flush_secs=30)

# Train Process
for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Training Now:"):
    model.train()
    Train_loss = 0.0
    for i, (cubes, labels) in enumerate(dataset_train):
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outs = model(cubes.to(device))
            losses = criterion(outs, labels.to(device))

        Train_loss += losses.item()

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
        del cubes, labels, losses
        torch.cuda.empty_cache()
    tqdm.write("epoch: {}  loss: {}".format(epoch, Train_loss))
    writer.add_scalar(tag='Loss/train', scalar_value=Train_loss, global_step=epoch)
    scheduler.step()

    model.eval()
    correct_num = 0.0
    for i, (cubes, labels) in enumerate(dataset_test):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(cubes.to(device))
            outputs = softmax_func(outputs)
            _, predicted = torch.max(outputs.data, 1)
            correct_num += (labels.to(device) == predicted).sum().item()
            del cubes, labels, outputs

    tqdm.write('Accuracy of the network on the {} validation images: {} %'.format(3300, 100 * correct_num / 3300.0))
    writer.add_scalar(tag='Acc/val', scalar_value=correct_num / 3300.0, global_step=epoch)
writer.close()