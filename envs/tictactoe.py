from collections import defaultdict
import functools
import random
from typing import *
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

import gym
from gym import spaces

def rotate(sliceable, n):
    '''Rotates any sliceable to the left by n.'''
    n %= len(sliceable)
    return sliceable[n:] + sliceable[:n]

def replace(tup, row, col, value):
    '''Updating multi-dimension tuples in python is kind of annoying...'''
    new_tup = list(list(row) for row in tup)
    new_tup[row][col] = value
    return tuple(tuple(row) for row in new_tup)

class BoardState(NamedTuple):
    '''An immutable (and hashable) state object for tic-tac-toe-like games.'''
    board: Tuple[Tuple[int, ...], ...]
    player_order: Tuple[int, ...]
        
    @classmethod
    def initial(cls, size, players):
        board = tuple([(0,) * size] * size)
        player_order = tuple(range(1, players+1))
        return cls(board, player_order)
    
    @property
    def size(self):
        return len(self.board)
                    
    @property
    def last_player(self):
        return self.player_order[-1]
    
    @property
    def next_player(self):
        return self.player_order[0]
        
    def __getitem__(self, key):
        if type(key) == tuple:
            row, col = key
            return self.board[row][col]
        else:
            return tuple.__getitem__(self, key)
        
    @property
    @functools.lru_cache(maxsize=None)
    def valid_actions(self):
        actions = []
        for r in range(self.size):
            for c in range(self.size):
                if self[r, c] == 0:
                    actions.append((r, c))
        return actions
    
    @property
    def random_action(self):
        actions = list(self.valid_actions)
        return random.choice(actions)
    
    @property
    def is_full(self):
        return not any(self.valid_actions)
    
    @property
    def is_won(self):
        player = self.last_player
        
        for r in range(self.size):
            if all([self[r, c] == player for c in range(self.size)]):
                return True

        for c in range(self.size):
            if all([self[r, c] == player for r in range(self.size)]):
                return True

        return all([self[i, i] == player for i in range(self.size)]) or \
            all([self[i, self.size - i - 1] == player for i in range(self.size)])
    
    @property
    def is_tied(self):
        return self.is_full and not self.is_won
    
    @property
    def is_done(self):
        return self.is_full or self.is_won
    
    def __call__(self, action):
        new_player_order = rotate(self.player_order, 1)
        
        row, col = action
        proposed = self[row, col]
        if proposed != 0:
            return self._replace(player_order=new_player_order)
        
        new_board = replace(self.board, row, col, self.next_player)
        return BoardState(new_board, new_player_order)
    

class Reward(Enum):
    WIN = 1
    TIE = 0
    LOSS = -1

    
class Observation(NamedTuple):
    state: BoardState
    reward: Reward
    done: bool
    info: Any


class BoardStatePrinter(object):
    def __init__(self, markers='XOABCDEFGHIJKLMNPQRSTUVWYZ'):
        self.markers = markers
    
    def render_player(self, player):
        return ' ' if player < 1 else self.markers[player-1]
                
    def render_board(self, state):
        size = state.size
        for row in range(size):
            if row > 0:
                print('─' * ((size*2)-1))
            
            for col in range(size):
                if col > 0:
                      print("|", end="")
                print(self.render_player(state[row, col]), end="")
            print()
    
    def print(self, state):
        self.render_board(state)
        print("Last Play: " + self.render_player(state.last_player))
        

class BoardStatePlotter(object):
    def get_marker_size(self, ax, state):
        box = ax.get_window_extent()
        width = (box.x1 - box.x0) / state.size
        height = (box.y1 - box.y0) / state.size
        size = min(width, height)
        return size * size * 0.33
    
    def setup_axes(self, ax, state):
        ax.set_aspect('equal')
        ax.set_prop_cycle(None)
        
        # Square plot with axis grids, but no tick labels.
        ax.grid(True, c='k', lw=1.5, aa=True)
        ax.tick_params(axis='both',
                       left='off', top='off', right='off', bottom='off',
                       labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        
        ticks = np.arange(0.5, state.size-1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
        lim = (-0.5, state.size - 0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.invert_yaxis() # So the plotted board matches a printed one.
        
    def get_data(self, state):
        n_players = len(state.player_order)
        xs = [list() for i in range(n_players)]
        ys = [list() for i in range(n_players)]
        for r in range(state.size):
            for c in range(state.size):
                player = state[r, c]
                if player > 0:
                    ys[player-1].append(r)
                    xs[player-1].append(c)
        return xs, ys
    
    def plot(self, state, ax=None):
        if ax is None:
            ax = plt.gca() # Get current Axes object if unset.
        
        self.setup_axes(ax, state)
        
        xs, ys = self.get_data(state)
        marker_size = self.get_marker_size(ax, state)
        return [ax.scatter(xs[m], ys[m], s=marker_size) for m in range(len(xs))]

            
class TicTacToeEnv(gym.Env):
    renderers = {
        'human': BoardStatePrinter().print,
        'matplotlib': BoardStatePlotter().plot
    }
    
    metadata = {'render.modes': [renderers.keys()]}
    
    def __init__(self, size=3, n_players=2):
        self.size = size
        self.n_players = n_players
        
        self.reward_range = (Reward.LOSS, Reward.WIN)
        
        # An action is simply specifying which space on the board to play next.
        self.action_space = spaces.Tuple([spaces.Discrete(size)] * 2)
        
        # The observation space is a tuple with two elements, the board and the player order.
        # The board is a tuple, one element for each space on the board, which can contain
        # 0 for empty or {player number}.
        size_space = spaces.Discrete(size)
        board_space = spaces.Tuple([spaces.Discrete(n_players + 1)] * (size ** 2))
        next_player_space = spaces.Tuple([spaces.Discrete(n_players)] * self.n_players)
        self.observation_space = spaces.Tuple([size_space, board_space, next_player_space])
        
        self.reset()
        
    def reset(self):
        self.state = BoardState.initial(self.size, self.n_players)
        return self.state
            
    def step(self, action):
        self.state = self.state(action)
        
        if self.state.is_won:
            return Observation(self.state, 1, True, {})
        elif self.state.is_tied:
            return Observation(self.state, 0, True, {})
        else:
            return Observation(self.state, None, False, {})
                               
    def render(self, mode='human'):
        self.renderers[mode](self.state)