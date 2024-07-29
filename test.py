import torch
import time
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def test_model(model,device,testloader):# Initialize lists to collect true labels and predictions
    all_labels = []
    all_predictions = []
    model.eval()
    with torch.no_grad():
        since = time.time()
        test_correct = 0
        test_total = 0
        for currx, labels in testloader:
            currx = currx.to(device)
            labels = labels.to(device)
            _, labels_index = torch.max(labels, 1)
            outputs = model(currx).to(device)
            _, predicted = torch.max(outputs, 1)

            # Collect true labels and predictions
            all_labels.extend(labels_index.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            test_total += labels.size(0)
            test_correct += (predicted == labels_index).sum().item()
            time_elapsed = time.time() - since
        print(f'testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60}s')
        # Calculate accuracy
        accuracy = 100 * test_correct / test_total
        print('Test Accuracy: {:.3f} %'.format(accuracy))

        # Calculate F1 score
        f1 = f1_score(all_labels, all_predictions, average='macro')
        print('F1 Score: {:.3f}'.format(f1))

        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        # Normalize confusion matrix by true class counts
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] *100

        # Plot normalized confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix(%)')
        plt.show()

