# On PyTorch
import torch
from scipy.signal import convolve2d
import itertools
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import imageio
import os
from tqdm.auto import tqdm
from mpl_toolkits.mplot3d import Axes3D # 3D plotting
from functools import wraps

def init_map_2d(N, C=2):
    """
    Args:
        N: Square map size
        C: Number of different populations. Default: 2
        empty: % of field left empty. Default: 10%

    Returns:
        result: Array of shape (C, N, N)
    """
    board = torch.rand((N, N))
    game_map = torch.stack([((1 / C) * c < board) * (board <= (1 / C) * (c + 1)) for c in range(C)])
    return game_map.to(torch.float)

def compress_2d(game_map):
    """
    Args:
        game_map: Array of shape (C, N, N)
    
    Returns:
        compressed_map: Array of shape (N, N)
    """
    C, N, _ = game_map.shape
    result = torch.zeros(N, N)
    for c in range(C):
        result += game_map[c] * c
    return result.to(game_map)

def decompress_2d(game_map_2d, C):
    """
    Args:
        game_map_2d: Compressed map of shape (N, N)
        C: Number of channels

    Returns:
        decompress_map: Array of shape (C, N, N)
    """
    N, _ = game_map_2d.shape
    result = torch.zeros(C, N, N)
    for c in range(C):
        result[c] = (game_map_2d == c)
    return result.to(game_map_2d)

def get_kernel_2d(distance, kernel_size):

    if distance == 'L2':
        kernel_2d = torch.ones(kernel_size, kernel_size)
        kernel_2d[kernel_size // 2, kernel_size // 2] = 0 # 2D square with 0 in center
    elif distance == 'L1':
        if kernel_size == 3:
            kernel_2d = torch.tensor([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]])
        elif kernel_size == 5:
            kernel_2d = torch.tensor([
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ])
        else:
            raise ValueError("Kernel size not supported")
    else:
        raise ValueError("Only `L2` and `L1` distance is supported")
    return kernel_2d
    

def game_step_2d(game_map: torch.Tensor, r: float, distance: str = 'L2', kernel_size: int = 3):
    """
    Args:
        game_map: Array of shape (C, N, N)
        r: Tolerance value. If less than r% of neighbours are of the same class, cell whants to move.
        distance: One of {`L2`, `L1`}.
        kernel_size: Size of neighbourhood to look at.
        
    """
    C, N, N = game_map.shape
    
    kernel_2d = get_kernel_2d(distance, kernel_size).to(game_map)
    
    compressed_map = compress_2d(game_map)
    neighbours_3d = torch.zeros(C, N, N).to(game_map)
    for c in range(C):
        neighbours_3d[c] = torch.nn.functional.conv2d(
            game_map[c][None, None, ...], kernel_2d[None, None, ...], padding=kernel_size // 2, stride=1)

    neighbours_3d *= game_map
    neighbours = neighbours_3d.max(axis=0).values
    moving = neighbours < int(kernel_2d.sum() * r)
    num_moving = moving.sum().item()
    
    moving_colors = compressed_map[moving]
    idx = torch.randperm(moving_colors.nelement())
    compressed_map[moving] = moving_colors.view(-1)[idx].view(moving_colors.size())

    updated_map = decompress_2d(compressed_map, C)
    return updated_map, num_moving

def game_2d(N, C, r=0.3, game_length=30, distance: str = 'L2', kernel_size: int = 3, create_gif=False, device="cpu"):
    """
    Start an iterative moving process. 
    
    Args:
        N: Size of game board
        C: Number of different populations
        game_length: Number of iterations
        r: Tolerance value. If less than r% of neighbours are of the same class, cell whants to move.
        distance: One of {`L2`, `L1`}.
        kernel_size: Size of neighbourhood to look at.
    
    Returns:
        move_hist: Number of cells that moved at each iteration
    """
    assert r <= 1, "Wrong r value! 0 <= r <= 1"
    game_map = init_map_2d(N, C).to(device)
    name = f'schelling2d_size-{N}_C-{C}_dist-{distance}_ks-{kernel_size}_neigh-{int(r*8)}'
    move_hist = []
    images = []
    for i in tqdm(range(game_length),desc=f'Number of neighbours={int(r*8)}', leave=False):
        # Return torch.tensor
        game_map, moved = game_step_2d(game_map, r, distance='L2', kernel_size=3)
        
        if create_gif:
            game_map_2d = compress_2d(game_map)
            fname = f'imgs/{name}.png'
            plot_2d(game_map_2d, C=C, r=r, i=i, save_image=True, name=fname, figsize=(14,11))
            images.append(imageio.imread(fname))
            os.remove(fname)

        move_hist.append(moved)
        
    if create_gif:
        fname = f'imgs/{name}.gif'
        imageio.mimsave(fname, images, fps = 10)
    return move_hist
        
def prepare_2d_plot(game_map):
    """
    Takes CxNxN matrix and return 
    2D array for plotting
    """
    game_map_2d = compress_2d(game_map)
    return game_map_2d

def plot_2d(game_map_2d, C=2, r=None, i=None, save_image=False, name=None, figsize=(14,11)):
    plt.figure(figsize=figsize)
    
    plt.imshow(game_map_2d.numpy())
#     if C > 2:
#         plt.imshow(game_map_2d)
#     else:
#         plt.imshow(game_map_2d, cmap=plt.cm.gray)
    
    if r:
        plt.title(f'Schelling’s: r = {r}, iteration = {i}', fontsize=20)
    plt.axis('off')
    if save_image:
        plt.savefig(name)
        plt.close()
