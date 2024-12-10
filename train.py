import copy
import time
import torch
import torch.nn as nn
import numpy as np
from tools.mixup import mixup
from PrunedAttentionGRU import prunedAttentionGRU

def set_train_model(device, input_size,hidden_size,attention_dim,num_classes,learningrate ,epochs):
    model = prunedAttentionGRU(input_size, hidden_size, attention_dim,num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = learningrate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return model,criterion,optimizer,scheduler
def train_model(device, model, criterion, optimizer, scheduler, num_epochs, trainloader, testloader, p=[0.7, 0.3]):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # History of acc and loss
    loss_hist = {"train": [], "test": []}
    acc_hist = {"train": [], "test": []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = trainloader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = testloader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                labels = labels.type(torch.FloatTensor)
                inputs = inputs.type(torch.FloatTensor)  # Ensure inputs are float32

                optimizer.zero_grad()

                # Option selection
                option = np.random.choice(['mixup', 'naive'], p=p)

                if option == "mixup":
                    inputs, label1, label2, lam = mixup(inputs, labels, 1.0)
                    inputs = inputs.to(device)
                    label1, label2 = label1.to(device), label2.to(device)
                    lam = float(lam)
                else:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward
                    prediction = model(inputs)
                    _, preds = torch.max(prediction, 1)
                    # print(prediction.shape, preds.shape, labels.shape)
                    if option == "naive":
                        loss = criterion(prediction, labels)
                        _, onehotlabels = torch.max(labels, 1)
                        acc = float(torch.sum(preds == onehotlabels)) / len(preds)
                    else:
                        loss = criterion(prediction, label1) * lam + criterion(prediction, label2) * (1 - lam)
                        _, onehotlabels1 = torch.max(label1, 1)
                        _, onehotlabels2 = torch.max(label2, 1)
                        acc = (lam * (preds == onehotlabels1).cpu().sum().float() +
                               (1 - lam) * (preds == onehotlabels2).cpu().sum().float()) / len(preds)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += acc * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = 100 * running_corrects / len(dataloader.dataset)

            loss_hist[phase].append(epoch_loss)
            acc_hist[phase].append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}')

            # Deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_hist, acc_hist

def cross_validate(device, model_class, criterion, optimizer_class, scheduler_class, dataset, num_epochs=50, n_splits=5, batch_size=32, p=[0.7,0.3]):
    """
    Perform 5-fold cross validation.
    model_class: A callable that returns a new instance of the model (uninitialized)
    optimizer_class: A callable that takes model.parameters() and returns an optimizer
    scheduler_class: A callable that takes optimizer and returns a scheduler
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"Fold {fold+1}/{n_splits}")

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # Instantiate a new model for each fold
        model = model_class().to(device)
        optimizer = optimizer_class(model.parameters())
        scheduler = scheduler_class(optimizer)

        model, loss_hist, acc_hist = train_model(device, model, criterion, optimizer, scheduler, num_epochs, trainloader, testloader, p=p)

        # Best accuracy for this fold is the max test accuracy recorded.
        best_acc = max(acc_hist['test'])
        fold_accuracies.append(best_acc)
        print(f"Fold {fold+1} Best Accuracy: {best_acc:.2f}")

    print(f"Average accuracy across {n_splits} folds: {np.mean(fold_accuracies):.2f}")
    return fold_accuracies
    
