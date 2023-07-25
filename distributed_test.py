import torch as th
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
import torchvision.datasets as datasets
import torch.utils.data as data_utils
import torchvision.transforms as transforms

class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1,stride=2)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3,padding=1,stride=2)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32,64)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(64,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = th.nn.functional.avg_pool2d(x,8)
        x = self.fc1(th.squeeze(x))
        x = self.act1(x)
        x = self.fc2(x)
        return x

def get_CIFAR10_dataloader(image_size, batch_size):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load data
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

# Package it up in batches
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

class Image_Classification_Module(pl.LightningModule):
        def __init__(self, model, loss_fun, optimizer):
            super().__init__()
            self.model = model
            self.loss_fun = loss_fun
            self.optimizer = optimizer

        def training_step(self, batch, idx):
            x, y = batch
            logits = self.model(x)
            # y = F.one_hot(y, VOCAB_SIZE).float()
            loss = self.loss_fun(logits, y)
            print(f"\rtrain_loss : {loss}", end="")
            return loss

        def validation_step(self, batch, idx):
            x, y = batch
            logits = self.model(x)
            # y = F.one_hot(y, VOCAB_SIZE).float()
            loss = self.loss_fun(logits, y)
            print(f"\rvalidation_loss : {loss}", end="")

            return loss

        def test_step(self, batch, idx):
            x, y = batch
            logits = self.model(x)
            # y = F.one_hot(y, VOCAB_SIZE).float()
            loss = self.loss_fun(logits, y)
            print(f"\rtest_loss : {loss}", end="")

            return loss

        def configure_optimizers(self):
            return self.optimizer

BSZ = 5
lr = .01
def experiment(gpus = 0):
    print("beginning experiment")
    train_loader, validation_loader, test_loader =  get_CIFAR10_dataloader(32, BSZ)
    loss_fun = nn.CrossEntropyLoss()
    model = TinyModel()
    optimizer = th.optim.Adam(model.parameters(),lr=lr)
    train_module = Image_Classification_Module(model,loss_fun,optimizer)
    if gpus == 0:
        accelerator = "cpu"
        devices = "auto"
    else:
        accelerator = "cuda"
        devices = gpus
    if gpus > 1:
        strategy = pl.strategies.DDPStrategy(static_graph=True)
    else:
        strategy = "auto"
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, max_epochs=2, strategy=strategy,
                         num_nodes=1, log_every_n_steps=50)
    print("Trainer system parameters:")
    print(f"\t trainer.world_size : {trainer.world_size}")
    print(f"\t trainer.num_nodes : {trainer.num_nodes}")
    print(f"\t trainer.accelerator : {trainer.accelerator}")
    print(f"\t trainer.device_ids {trainer.device_ids}")
    print(f"\t train_loader.num_workers : {train_loader.num_workers}")

    print("Training Begins")
    trainer.fit(train_module,train_dataloaders=train_loader, val_dataloaders=validation_loader)
    print("training is over")


if __name__ == "__main__":
    experiment()
