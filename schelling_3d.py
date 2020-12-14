import os
from scipy.signal import convolve
import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
from tqdm.auto import tqdm
from mpl_toolkits.mplot3d import Axes3D # 3D plotting

# Same functions for 3D case
def init_map_3d(N, C=2):
    """
    Args:
        N: Cubic map size
        C: Number of different populations. Default: 2
        empty: % of field left empty. Default: 10%

    Returns:
        result: Array of shape (C, N, N, N)
    """
    board = torch.rand((N, N, N))
    game_map = torch.stack([((1 / C) * c < board) * (board <= (1 / C) * (c + 1)) for c in range(C)])
    return game_map.to(torch.float)

def compress_3d(game_map):
    """
    Args:
        game_map: Array of shape (C, N, N, N)
    
    Returns:
        compressed_map: Array of shape (N, N, N)
    """
    C, N, _, _ = game_map.shape
    result = torch.zeros(N, N, N)
    for c in range(C):
        result += game_map[c] * c
    return result.to(game_map)


def decompress_3d(game_map_3d, C):
    """
    Args:
        game_map_3d: Compressed map of shape (N, N, N)
        C: Number of channels

    Returns:
        decompress_map: Array of shape (C, N, N, N)
    """
    N = game_map_3d.shape[0]
    result = torch.zeros(C, N, N, N)
    for c in range(C):
        result[c] = (game_map_3d == c)
    return result.to(game_map_3d)


def get_kernel_3d(distance, kernel_size):
    if distance == 'L2':
        kernel_3d = torch.ones(kernel_size, kernel_size, kernel_size)
        kernel_3d[kernel_size // 2, kernel_size // 2, kernel_size // 2] = 0 # 3D cube with 0 in center
    elif distance == 'L1':
        if kernel_size == 3:
            kernel_3d = torch.tensor([
                [[0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],

                [[0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]],

                [[0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]]])

        elif kernel_size == 5:
            kernel_3d = torch.tensor([
                [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
                
                [[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]],
                
                [[0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0]],
                
                [[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]],
                
                [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]])
        else:
            raise ValueError("Kernel size not supported")
    else:
        raise ValueError("Only `L2` and `L1` distance is supported")
        
    return kernel_3d
    
    
def game_step_3d(game_map: torch.Tensor, r: float, distance: str = 'L2', kernel_size: int = 3):
    """
    Args:
        game_map: Array of shape (C, N, N, N)
        r: Tolerance value. If less than r% of neighbours are of the same class, cell whants to move.
        distance: One of {`L2`, `L1`}.
        kernel_size: Size of neighbourhood to look at.
    Returns:
        updated_map: New game map
        num_moving: Number of cells moved on this step
    """
    
    C, N, N, N = game_map.shape
    
    kernel_3d = get_kernel_3d(distance, kernel_size).to(game_map)
    compressed_map = compress_3d(game_map)

    # max is just fancy way to compress 4D array into 3D
    neighbours_4d = torch.zeros(C, N, N, N).to(game_map)
    for c in range(C):
        neighbours_4d[c] = torch.nn.functional.conv3d(
            game_map[c][None, None, ...], kernel_3d[None, None, ...], padding=kernel_size // 2, stride=1)
        
#         neighbours_4d[c] = torch.nn.functional.conv2d(
#             game_map[c][None, ...], kernel_3d[None, ...], padding=kernel_size // 2, stride=1)
        
    neighbours_4d *= game_map
    neighbours = neighbours_4d.max(axis=0)

    moving = neighbours < int(kernel_3d.sum() * r)
    num_moving = moving.sum()
    
    compressed_map = compress_3d(game_map)
    
    moving_colors = compressed_map[moving]
    idx = torch.randperm(moving_colors.nelement())
    compressed_map[moving] = moving_colors.view(-1)[idx].view(moving_colors.size())
    
    updated_map = decompress_3d(compressed_map, N, C)
    return updated_map, num_moving

def game_3d(
        N, C, r=0.3, game_length=30, distance: str = 'L2', kernel_size: int = 3,
        create_gif=False, proj=False, device="cpu"):
    """
    Start an iterative moving process. 
    
    Args:
        N: Size of game board
        C: Number of different populations
        r: Tolerance value. If less than r% of neighbours are of the same class, cell whants to move.
        game_length: Number of iterations
        distance: One of {`L2`, `L1`}.
        kernel_size: Size of neighbourhood to look at.
        create_gif: Flag to save game as GIF
        proj: Flag to save projections on 3 axes instead of 3D plane
    
    Returns:
        game_map:
        move_hist: Number of cells that moved at each iteration
    """
    assert r <= 1, "Wrong r value! 0 <= r <= 1"
    game_map = init_map_3d(N, C).to(device)
    
    FIGSIZE = (14, 11)
    if create_gif:
        # Plot inital conditions
        x, y, z, c, proj_x, proj_y, proj_z = prepare_3d_plot(game_map, projections=True)
        if N < 20: # With big N makes no sence
            plot_3d(x, y, z, c, r=r, save_image=False, figsize=FIGSIZE)
        plot_projections(proj_x, proj_y, proj_z)
        
    name = f'schelling3d_size-{N}_C-{C}_dist-{distance}_ks-{kernel_size}_neigh-{int(r*27)}'
    fname = 'imgs/schelling3d_tmp.png'

    move_hist = []
    images = []
    
    for i in tqdm(range(game_length), desc=f'Number of neighbours={int(r*27)}', leave=False):
        game_map, moved = game_step_3d(game_map, r)
        x, y, z, c, proj_x, proj_y, proj_z = prepare_3d_plot(game_map, projections=True)
        if proj:
            plot_projections(proj_x, proj_y, proj_z, save_image=True, name=fname)
        else:
            plot_3d(x, y, z, c, r=r, save_image=True, name=fname, figsize=FIGSIZE)
        images.append(imageio.imread(fname))
        os.remove(fname)
        move_hist.append(moved)

    imageio.mimsave(f'imgs/{name}.gif', images, fps = 10)
    
    if create_gif:
        # Plot final conditions
        x, y, z, c, proj_x, proj_y, proj_z = prepare_3d_plot(game_map, projections=True)
        if N < 20: # With big N makes no sence
            plot_3d(x, y, z, c, r=r, save_image=False, figsize=FIGSIZE)
        plot_projections(proj_x, proj_y, proj_z)
    
    return game_map, move_hist

def prepare_3d_plot(game_map, projections=False):
    """
    Args:
        game_map: Array of shape (C, N, N, N)
    
    Return:
        
    Takes CxNxNxN matrix and return 
    N^3 x 4 matrix (x, y, z and colour channels)
    """
    C, N, _, _ = game_map.shape
    assert C == 2, "Only 2 colours are supported!"
    game_map_3d = compress_3d(game_map)
    xyz = torch.tensor(list(itertools.product(range(N), range(N), range(N))))
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    c = game_map_3d.reshape(-1)
    
    if C == 2:
        # delete entries for one of the colours
        mask = (c ==1)
        c = c[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if projections:
            ## Return also 3 projections on X, Y and Z planes
            proj_x, proj_y, proj_z = game_map_3d.sum(axis=0), game_map_3d.sum(axis=1), game_map_3d.sum(axis=2)
            return x, y, z, c, proj_x, proj_y, proj_z
            
    return x, y, z, c
 
def plot_projections(proj_x, proj_y, proj_z, save_image=False, name=None):
    fig, ax = plt.subplots(ncols=3, figsize=(15,5))
#     fig.suptitle('test title', fontsize=10)
    ax[0].imshow(proj_x)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[1].imshow(proj_y)
    ax[2].imshow(proj_z)
    ax[0].set_title('Projection on X')
    ax[1].set_title('Projection on Y')
    ax[2].set_title('Projection on Z')
    plt.axis('off')
    if save_image:
        plt.savefig(name)
        plt.close()
    
def plot_3d(x, y, z, c, r=None, save_image=False, name=None, figsize=(14, 11)):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
#     ax = plt.axes()
    if c.max() <= 1:
#         ax.scatter(x, y, z, c=c, linewidth=0.5, marker='.', cmap=plt.cm.gray)
#         ax.scatter(x, y, z, linewidth=0.5, marker='o', cmap=plt.cm.gray)
        ax.scatter(x, y, z, c='red', linewidth=0.5, marker='.', s=5, depthshade=False)
    else:
        ax.scatter(x, y, z, c=c, linewidth=0.5, marker='.')
    ax.set_axis_off()
    if r:
        ax.set_title(f'3D model with r={r}', fontsize=20)
        
    if save_image:
        plt.savefig(name)
        plt.close()
