import numpy as np
import matplotlib.pyplot as plt


def show(in_imgs, out_imgs):
    fig, axes = plt.subplots(2, figsize=(20, 10))
    npimgs_in = in_imgs.numpy()
    npimgs_out = out_imgs.numpy()
    axes[0].imshow(np.transpose(npimgs_out, (1,2,0)), interpolation='nearest')
    axes[1].imshow(np.transpose(npimgs_in, (1,2,0)), interpolation='nearest')
    axes[0].title.set_text('Output')
    axes[1].title.set_text('Input')