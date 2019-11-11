"""
This is a collection of utility functions.
"""

def dprint(content):
    print(content)
    log_file = open('log.txt', 'a')
    print(content, file=log_file)
    log_file.close()

def rgb2gray(rgb):
    import numpy as np
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def colorbar(mappable):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cax.tick_params(labelsize=7)
    return fig.colorbar(mappable, cax=cax)

def cli_progress_test(end_val, bar_length=20):
    import sys
    for i in range(0, end_val):
        percent = float(i) / end_val
        hashes = '#' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(hashes))
        sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
        sys.stdout.flush()
    sys.stdout.write("\n")