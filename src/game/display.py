import numpy as np

def show_state(ax, state):
    """Display the game state in a matplotlib plot.
    
    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object to create the plot in.
    state : array_like
        A 4x4 array containing integers.
    """
    ax.imshow(state)
    for (i, j), v in np.ndenumerate(state):
        ax.text(j, i, v, ha="center", va="center", color="w", size=20)
