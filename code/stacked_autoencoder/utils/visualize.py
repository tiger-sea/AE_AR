import matplotlib.pyplot as plt

def visualize_loss(history, title: str):
    """
    Visualize history of keras.callbacks
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    # mae = history.history["mae"]
    epochs = range(1, len(loss)+1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, loss, "b", label="Training loss MSE")
    # plt.plot(epochs, mae, "g", label="Training loss MAE")
    plt.plot(epochs, val_loss, "r", label="Validation loss MSE")
    plt.title(title)
    if len(loss) < 15:
        plt.xticks(list(range(1, len(loss)+1)))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# def visualize_prediction():
