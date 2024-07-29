import matplotlib.pyplot as plt

def plot_training_history(loss_hist, acc_hist):
    epochs = range(len(loss_hist['train']))

    plt.figure(figsize=(12, 5))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_hist['train'], label='Train Loss')
    plt.plot(epochs, loss_hist['test'], label='Test Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_hist['train'], label='Train Accuracy')
    plt.plot(epochs, acc_hist['test'], label='Test Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()