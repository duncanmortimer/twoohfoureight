import numpy as np


def identity(a): return a
def rotCW(a): return np.fliplr(a.T)
def rotCCW(a): return np.flipud(a.T)


LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


moves = {
    LEFT: [rotCCW, rotCW],
    UP: [identity, identity],
    RIGHT: [rotCW, rotCCW],
    DOWN: [np.fliplr, np.fliplr],
}


def next_state(state, move):
    transform, inv_transform = moves[move]
    nst = np.copy(transform(state))
    nst = nst[np.indices((4,4))[0], np.argsort(nst==0)]
    for ind in range(3):
        cmp = (nst[:, ind] == nst[:, ind+1]) * (nst[:, ind] > 0)
        nst[:, ind] += cmp
        nst[cmp, ind+1] = -1
    nst = nst[np.indices((4,4))[0], np.argsort(nst == -1)]
    nst[nst==-1] = 0
    return inv_transform(nst)


def perturb(state):
    zeros = np.argwhere(state == 0)
    if len(zeros) > 0:
        nst = np.copy(state)
        loc = np.random.randint(zeros.shape[0])
        row, col = zeros[loc, :]
        nst[row, col] = 1
        return nst
    else:
        return state


def is_done(state):
    num_zeros = np.sum(state == 0)
    if num_zeros > 0:
        return False
    else:
        up = next_state(state, UP)
        left = next_state(state, LEFT)
        return np.all(up == left)


def play_to_completion(strategy, n_games, progress_callback=None):
    init_state = np.array([
        [1,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
    ])
    state = np.copy(init_state)
    record = []
    for game in range(n_games):
        move_count = 0
        state = np.copy(init_state)
        moves_taken = []
        while not is_done(state):
            move = strategy(state)
            new_state = next_state(state, move)
            if not np.all(new_state == state):
                state = perturb(new_state)
                move_count += 1
                moves_taken += [move]
        record += [(move_count, np.max(state))]
        if progress_callback: progress_callback(game)
    return record