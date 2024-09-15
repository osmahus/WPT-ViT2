import numpy as np
import matplotlib.pyplot as plt
import pywt
from pywt import WaveletPacket2D as wp2d
import ptwt
import torch
from einops import rearrange
from mpl_toolkits.axes_grid1 import ImageGrid

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


####################################################################################
#                    Tensor to Numpy (Get Ready for Plot)
####################################################################################
def tensor_to_numpy(img):

    # Make sure the used Image is numpy array with (Height,Width,Channel) shape
    if torch.is_tensor(img):
        img = img.numpy()

    if img.shape[0] == 3:
        img = rearrange(img, 'c h w -> h w c')

    return img


####################################################################################
#                    Norm to Plot
####################################################################################
def norm_to_plot(img):

    # Make sure the used Image is numpy array with (Height,Width,Channel) shape
    img = tensor_to_numpy(img)

    img_out = (img-img.min())/(img.max()-img.min())

    return img_out


####################################################################################
#                    Plot the Image and its Extracted WPT Features
####################################################################################
def plot_img_grid(holder, data, main_title, sub_title=None, Grid2D=True, setticks=None, normalize=False, axes_pad=0.3):

    # If Grid2D = False , means that the Grid is one dimensional (row of pictures)

    data_rows = data.shape[0]
    data_cols = data.shape[1]
    n = data[0, 0].shape[0]
    extent = [0, n, n, 0]

    if Grid2D:
        nrows = data_rows
        ncols = data_cols
    else:
        nrows = 1
        ncols = data_rows*data_cols

    x_label = range(ncols)
    y_label = range(nrows)

    grid = ImageGrid(holder, 111, (nrows, ncols), axes_pad=axes_pad)

    for i, ax in enumerate(grid):
        r, c = i//data_cols, i % data_cols

        if normalize:
            ax.imshow(norm_to_plot(data[r, c]), extent=extent)
        else:
            ax.imshow(data[r, c], extent=extent)

        if setticks is not None:
            ax.set(xticks=np.arange(0, data[r, c].shape[1]+1, step=setticks),
                   yticks=np.arange(0, data[r, c].shape[0]+1, step=setticks))
        else:
            ax.set(xticks=[], yticks=[])

        if Grid2D:
            if r == nrows - 1:
                ax.set_xlabel(x_label[c], rotation=0, fontsize=10, labelpad=20)
            if c == 0:
                ax.set_ylabel(y_label[r], rotation=0, fontsize=10, labelpad=20)
        else:
            ax.set_xlabel(x_label[i], rotation=0, fontsize=10, labelpad=20)
            if i == 0:
                ax.set_ylabel(y_label[i], rotation=0, fontsize=10, labelpad=20)

        if sub_title is not None:
            ax.set_title(sub_title[i], fontsize=10)

    holder.suptitle(main_title)
    plt.show()


####################################################################################
#      Decompose the Image using Wavelet Packet Transform(Keep Features Only)
####################################################################################
def wpt_dec(img, wavelet_fun, level, mode='symmetric', slice=None):
    # This function decomposes the input 2D image into (2**level)**2 2D features
    # hence if we have only one level the output 2D features will be 4
    # The 2D features could be arranged as a matrix of (2**level) rows and (2**level) cols
    # ranging from the most approximate feature at location (0,0) to the most detailed
    # feature at location ((2**level) - 1,(2**level) - 1)
    # The original 2D wave_packet_decompose function down samples the input by 2 after each filter

    # Make sure the used Image is numpy array with (Height,Width,Channel) shape.
    isBatch = True if len(img.shape) == 4 else False

    if level == 0:

        # Although there is no WPT decomposition when level = 0 , however will add extra dimension in position 0 as an index of decomposition
        nodes_tensor = img.unsqueeze(0)
        paths = ['original']

    else:
        paths = [0]*3
        features_rows = 2**level

        ptwp = ptwt.WaveletPacket2D( img.float(), wavelet_fun, mode=mode, maxlevel=level)
        paths = [key for key in ptwp if len(key) == level]
        nodes = [ptwp[path] for path in paths]
        nodes_tensor = torch.stack(nodes)

        # # Arrange the paths in a 2D matrix shape, useful to visualize the wavelet packet features
        paths_rows = []
        paths_matrix = []
        for i, path in enumerate(paths):
            if (i+1) % features_rows == 0:
                paths_rows.append(path)
                paths_matrix.append(paths_rows)
                paths_rows = []
            else:
                paths_rows.append(path)

    if isBatch:
        # rearrange batch to go to first position and decomposition to the second position
        nodes_tensor = nodes_tensor.permute(1, 0, 2, 3, 4)
        nodes_array = nodes_tensor.permute(0, 1, 3, 4, 2).to('cpu').numpy()
        if slice != None:
            nodes_tensor = nodes_tensor[:, slice, :, :, :]
            sliced_nodes_array = np.zeros(nodes_array.shape)
            sliced_nodes_array[:, slice, :, :, :] = nodes_array[:, slice, :, :, :]
            nodes_array = sliced_nodes_array

    else:
        nodes_array = nodes_tensor.permute(0, 2, 3, 1).to('cpu').numpy()
        if slice != None:
            nodes_tensor = nodes_tensor[slice, :, :, :]
            sliced_nodes_array = np.zeros(nodes_array.shape)
            sliced_nodes_array[slice, :, :, :] = nodes_array[slice, :, :, :]
            nodes_array = sliced_nodes_array

    return paths, nodes_array, nodes_tensor


####################################################################################
#                    Plot the Image and its Extracted WPT Features
####################################################################################
def plot_wpt_nodes(image, wavelet_fun, level, setticks=None, slice=None, figsize=(10, 10)):
    paths, nodes, _ = wpt_dec(image, wavelet_fun, level, mode='symmetric', slice=slice)
    plt.rcParams['figure.constrained_layout.use'] = True

    fig = plt.figure(figsize=figsize)

    grid_text = "Features extracted using Wavelet Packet Transform"

    # nodes shape: (features,features_height,features_weight,channels)

    nodes_grid = np.reshape(nodes, (int(np.sqrt(nodes.shape[0])), -1, nodes.shape[1], nodes.shape[2], nodes.shape[3]))

    plot_img_grid(fig, nodes_grid, grid_text, paths, Grid2D=True, normalize=True, setticks=setticks)




####################################################################################
#                     Plot the Wavelet Impulse function
####################################################################################

def adj_plot(axes, x, y, step, title, zoom, lim):

    axes.plot(x, y)
    axes.margins(zoom)

    if lim[0] != None:
        axes.set_xlim(lim[0])
    xmin0, xmax0 = axes.get_xlim()
    xmin0 = ((xmin0//step[0])/(1/step[0]))+step[0]
    xmax0 = ((xmax0//step[0])/(1/step[0]))+step[0]
    axes.set_xticks(np.arange(xmin0, xmax0, step=step[0]))

    if lim[1] != None:
        axes.set_ylim(lim[1])
    ymin0, ymax0 = axes.get_ylim()
    ymin0 = ((ymin0//step[1])/(1/step[1]))+step[1]
    ymax0 = ((ymax0//step[1])/(1/step[1]))+step[1]
    axes.set_yticks(np.arange(ymin0, ymax0, step=step[1]))

    axes.axhline(0, linestyle='dashed', color='r', alpha=0.2)
    axes.axvline(0, linestyle='dashed', color='r', alpha=0.2)
    axes.set_xlabel("(t)")
    axes.set_title(title)
    axes.grid(True)
    

def plot_wpt_fun(wavelet_fun, zoom, step1=(1, 1), step2=(1, 1), lim1=(None, None), lim2=(None, None), figsize=(6, 2)):

    wavelet = pywt.Wavelet(wavelet_fun)
    [phi, psi, x] = wavelet.wavefun()

    fig, axs = plt.subplots(1, 2, figsize=figsize, layout='constrained')

    # axs[0].plot(x, phi)
    title1 = f'{wavelet_fun} scaling function\nΦ(t)'
    adj_plot(axs[0], x, phi, step1, title1, zoom[0], lim1)

    # axs[1].plot(x, psi)
    title2 = f'{wavelet_fun} wavelet function\n{chr(936)}(t)'
    adj_plot(axs[1], x, psi, step2, title2, zoom[1], lim2)

    plt.show()


####################################################################################
#                     Print Wavelet Families
####################################################################################
def wavelet_families():
    w_family = pywt.families()[:7]
    wavelet_lst = []
    for i, w_fun in enumerate(w_family):
        wavelet_lst.append(pywt.wavelist(w_fun))
        formatted_w_fun = f'{i}) {w_fun}: {", ".join(pywt.wavelist(w_fun))}'
        print(formatted_w_fun)
    print(" \n")