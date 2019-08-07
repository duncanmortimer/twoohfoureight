import numpy as np
from .rules import (UP, LEFT, DOWN, RIGHT, identity)

def rotCWMulti(st): return np.rot90(st, axes=(1,2))
def rotCCWMulti(st): return np.rot90(st, axes=(2,1))
def flipMulti(st): return np.flip(st, axis=1)

moves = {
    UP: [identity, identity],
    LEFT: [rotCCWMulti, rotCWMulti],
    DOWN: [flipMulti, flipMulti],
    RIGHT: [rotCWMulti, rotCCWMulti]
}

def next_state(state, move):
    transform, inv_transform = moves[move]
    nst = np.copy(transform(state))
    nst = nst[np.indices(nst.shape)[0], np.argsort(nst==0, axis=1), np.indices(nst.shape)[2]]
    for ind in range(3):
        cmp = (nst[:, ind] == nst[:, ind+1]) * (nst[:, ind] > 0)
        nst[:, ind, :] += cmp
        nst[:, ind+1, :][cmp] = -1
    nst = nst[np.indices(nst.shape)[0], np.argsort(nst==-1, axis=1), np.indices(nst.shape)[2]]
    nst[nst==-1] = 0
    return inv_transform(nst)

def is_full(state):
    n_games = state.shape[0]
    return np.sum(state.reshape(n_games, -1) == 0, axis=1) == 0

def is_done(state):
    n_games = state.shape[0]
    return is_full(state) & np.all((next_state(state, 'left') == next_state(state, 'up')).reshape(n_games, -1), axis=1)

def has_changed(old_state, new_state):
    n_games = old_state.shape[0]
    return np.invert(np.all((old_state == new_state).reshape(n_games, -1), axis=1))

def perturb(state):
    not_full = np.invert(is_full(state))
    tmp = state[not_full]
    for ind, game in enumerate(tmp):
        zeros = np.argwhere(game == 0)
        loc = np.random.randint(zeros.shape[0])
        row, col = zeros[loc, :]
        game[row, col] = 1
    state[not_full] = tmp

def play_to_completion(strategy, n_games, max_turns = 5, progress_callback=None):
    init_state = np.array([[
        [1,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
    ]])
    state = np.repeat(init_state, repeats=n_games, axis=0)
    complete = np.repeat(False, repeats=n_games)
    move_count = np.zeros(n_games)
    for turn in range(max_turns):
        if np.all(complete):
            print(f"all games complete after {turn} turns")
            break
        incomplete = np.invert(complete)
        
        live_state = state[incomplete]
        moves_to_make = strategy(live_state)
        for m in [UP, LEFT, DOWN, RIGHT]:
            games_to_update = moves_to_make == m
            live_state[games_to_update] = next_state(live_state[games_to_update], m)
            
        did_change = has_changed(state[incomplete], live_state)
        if np.any(did_change):
            moves_made = np.zeros(np.sum(incomplete))
            moves_made[did_change] = 1
            move_count[incomplete] += moves_made
            changed = live_state[did_change]
            perturb(changed)
            live_state[did_change] = changed
            state[incomplete] = live_state
            complete[incomplete] = is_done(live_state)
            if progress_callback is not None:
                n_done = np.sum(complete)
                progress_callback(n_done)
    return move_count, complete, state