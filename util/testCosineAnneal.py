import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

max_lr = 1
min_lr = 0.001

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=max_lr)
scheduler_steps = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_steps)
losslist = []

for epoch in range(200):
    for idx in range(20):
        pass
    scheduler.step()
    print(scheduler.get_lr())
    if (epoch + 1) % 50 == 0:
        print('Reset scheduler')
        optimizer = optim.SGD(model.parameters(), lr=max_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, scheduler_steps)
    losslist.append(scheduler.get_lr())

    plt.figure(figsize=(15, 5))
    name = "train"
    plt.plot(range(len(losslist)), losslist, label=f'train {name}')
    plt.plot(range(200), range(1), label=f'val {name}')
    plt.title(f'{name} plot')
    plt.xlabel('Epoch')
    plt.ylabel(f'{name}')
    plt.legend()
    plt.savefig("./my.png")
    plt.close()
