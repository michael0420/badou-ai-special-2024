import torch
from paper_dataset import data_loader_train, data_loader_test
from alexnet import AlexNet

net = AlexNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = net.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    temp_loss = 0

    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        temp_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i % 100 == 99:
            print(f"epoch:{epoch},iteration:{i + 1},loss:{temp_loss / 100:.3f}")
            temp_loss = 0

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


if __name__ == '__main__':
    import os

    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    model.load_state_dict(torch.load("logs/ep010-loss0.196-val_loss0.368.pth"))

    best_acc = 0.0
    patience = 10
    trigger_times = 0

    start_epoch = 10
    num_epochs = 50

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train(model, data_loader_train, criterion, optimizer, device)
        val_loss, val_acc = validate(model, data_loader_test, criterion, device)

        print(f'Epoch {epoch + 1}/{50}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 模型保存方式，5个epoch保存一次
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       log_dir + f'ep{epoch + 1:03d}-loss{train_loss:.3f}-val_loss{val_loss:.3f}.pth')

        # 学习率下降方式
        if val_acc > best_acc:
            best_acc = val_acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f'Reduce learning rate to {optimizer.param_groups[0]["lr"]:.6f}')
                trigger_times = 0

        # 提前停止
        if trigger_times >= patience:
            print("Early stopping")
            break

    # 保存最终模型权重
    torch.save(model.state_dict(), log_dir + 'last1.pth')
