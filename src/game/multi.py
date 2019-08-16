import numpy as np
from .rules import (UP, LEFT, DOWN, RIGHT, identity)

def rotCWMulti(st): return np.rot90(st, axes=(1,2))
def rotCCWMulti(st): return np.rot90(st, axes=(2,1))
def flipMulti(st): return np.flip(st, axis=1)

moves = {
    LEFT: [rotCCWMulti, rotCWMulti],
    UP: [identity, identity],
    RIGHT: [rotCWMulti, rotCCWMulti],
    DOWN: [flipMulti, flipMulti],
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
    return np.sum(state == 0, axis=(1,2)) == 0

def is_done(state):
    n_games = state.shape[0]
    return is_full(state) & np.all((next_state(state, LEFT) == next_state(state, UP)).reshape(n_games, -1), axis=1)

def has_changed(old_state, new_state):
    n_games = old_state.shape[0]
    return np.invert(np.all((old_state == new_state).reshape(n_games, -1), axis=1))

def perturb(state):
    not_full = np.invert(is_full(state))
    tmp = state[not_full]
    for _, game in enumerate(tmp):
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


def env(n_games, reward_on_complete_fn):
    def init_state():
        s = np.array([[
            [1,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
        ]])
        return np.repeat(s, repeats=n_games, axis=0)

    def select_action(state, Q):
        """Given a (multi-)game state S and expected rewards Q for each action when taken from each state. This 
        will not allow the agent to select moves that will not update the game state (i.e. that are invalid).
        
        Parameters:
        -----------
        state : array_like
            Dimensions are (n_games, 4, 4, 1), where (i, :, :, 0) represents the state of game i
        Q : array_like
            Dimensions are (n_games, 4), where (i, j) is the expected reward for taking action j in game i
        
        Returns:
        --------
        array_like
            An (n_games,) dimensional array selecting the move that maximises the expected reward for each game
        """
        valid_moves = np.array([
            np.any(state != next_state(state, m), axis=(1,2))
            for m in [0, 1, 2, 3]
        ]).T
        choose_by = np.where(valid_moves, Q, -1e5)
        return np.argmax(choose_by, axis=1)

    def act(state, actions, live=np.full(n_games, True)):
        """Update the state(s) of the games on the basis of the selected actions, and calculate respective
        rewards.
        
        Parameters:
        -----------
        state : array_like
            The (n_games, 4, 4, 1) dimensional array of game states
        live : array_like
            An (n_games,) dimensional boolean array, where live[i] is True when game i has not yet completed
        actions : array_like
            The (n_games,) array of selected actions
            
        Returns:
        --------
        new_state : array_like
            An (n_games, 4, 4, 1) dimensional array of updated game states
        reward : array_like
            An (n_games,) dimensional array of rewards
        became_complete : array_like
            An (n_games,) dimensional array where became_complete[i] is True if this move was the last 
            for game i (i.e. it has no more valid moves), and False otherwise
        live : array_like
            An (n_games,) dimensional array where live[i] is True if game i is live after this move"""
        
        if np.all(~live):
            return state, np.zeros(n_games), ~live, live
        
        n_live = np.sum(live)
        live_reward = np.zeros(n_live)

        new_state = state[live]
        for m in [UP, LEFT, DOWN, RIGHT]:
            games_to_update = actions == m
            new_state[games_to_update] = next_state(new_state[games_to_update], m)
        
        did_change = has_changed(state[live], new_state)

        perturbed = new_state[did_change]
        perturb(perturbed)
        new_state[did_change] = perturbed        
        
        reward_invalid = 0.0
        live_reward[~did_change] += reward_invalid
        
        complete = is_done(new_state).reshape((n_live,))
        reward_complete = reward_on_complete_fn(new_state[complete])
        live_reward[complete] += reward_complete

        reward_move = 1
        live_reward[~complete & did_change] += reward_move
        
        result = np.copy(state)
        result[live] = new_state
        
        reward = np.zeros(n_games)
        reward[live] = live_reward
        
        became_complete = np.full((n_games,), False)
        became_complete[live] = complete

        return result, reward, became_complete, (live & ~became_complete)
    
    return init_state, select_action, act