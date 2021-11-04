import numpy as np
import matplotlib.pyplot as plt


def show(in_imgs, out_imgs, save=False, path=None):
    fig, axes = plt.subplots(2, figsize=(20, 10))
    npimgs_in = in_imgs.numpy()
    npimgs_out = out_imgs.numpy()
    axes[0].imshow(np.transpose(npimgs_out, (1,2,0)), interpolation='nearest')
    axes[1].imshow(np.transpose(npimgs_in, (1,2,0)), interpolation='nearest')
    axes[0].title.set_text('Original')
    axes[1].title.set_text('Compressed')
    if save:
        plt.savefig( path / 'validation_compare.png')
    
    
def plot_loss(criterion, train_history, val_history, path):
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(train_history.keys(), train_history.values(), label="train loss")
    plt.plot(val_history.keys(), val_history.values(), label="validation loss")
    plt.title("loss curves")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(str(criterion))
    plt.savefig(path / "loss_curves.png")
    
def write_losses(path, train_history, val_history):  
    with open(path / "loss_values.txt", "w") as f:
        for i in train_history.keys():
            f.write(f"epoch {i}: train loss is {train_history[i]:.6f}, validation loss is {val_history[i]:.6f}\n")
