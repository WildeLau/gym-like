# Reference:
#   https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np


class Easy21(gym.Env):
    """
    Easy21 is a card game corresponds to David Silver's RL course's assignment.
    http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf
    This environment is designed for model-free reinforcement learning.

    The game is played with an infinite deck of cards (i.e. cards are sampled
    with replacement).
    Each draw from the deck results in a value between 1 and 10 (uniformly
    distributed) with a colour of red (probability 1/3) or black (probability
    2/3).
    There are no aces or picture (face) cards in this game (i.e. [2, 10]).

    At the start of the game both the player and the dealer draw one black
    card (fully observed).

    Each turn the player may either stick (=0) or hit (=1).
    If the player hits then she draws another card from the deck.
    If the player sticks she receives no further cards.

    The values of the player’s cards are added (black cards) or subtracted (red
    cards).
    If the player’s sum exceeds 21, or becomes less than 1, then she “goes
    bust” and loses the game (reward -1).

    If the player sticks then the dealer starts taking turns. The dealer always
    sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes
    bust, then the player wins; otherwise, the outcome – win (reward +1),
    lose (reward -1), or draw (reward 0) – is the player with the largest sum.
    """

    def __init__(self):
        self.metadata = {'render.modes': ['ansi']}
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiDiscrete([22, 11])
        self.reward_range = [-1, 1]
        self.seed()
        self.score = {"player": None, "dealer": None}

        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)
        if action:   # hit
            self.score["player"] += self._draw_card()
            reward, done = self._get_reward_and_terminator("player")
        else:   # stick
            done = True
            while 1 <= self.score["dealer"] < 17:
                self.score["dealer"] += self._draw_card()
            reward, done = self._get_reward_and_terminator("dealer")
        return self._get_observation(), reward, done, {}

    def reset(self):
        return self._draw_hand()

    def render(self, mode='ansi'):
        if mode == 'ansi':
            state = "Easy21: Player {} vs {} Dealer ".format(
                self.score["player"], self.score["dealer"]
            )
            print(state)
            return state
        else:
            super(Easy21, self).render(mode=mode)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _draw_hand(self):
        self.score["player"] = self.np_random.randint(2, 11)
        self.score["dealer"] = self.np_random.randint(2, 11)
        return self._get_observation()

    def _draw_card(self):
        card_type = self.np_random.choice([-1, 1], p=[0.3333, 0.6667])
        card_idx = self.np_random.randint(2, 11)
        return card_type * card_idx

    def _get_observation(self):
        return (self.score["player"], self.score["dealer"])

    def _is_bust(self, score):
        return score < 1 or score > 21

    def _get_reward_and_terminator(self, caller):
        if caller == "player":
            if self._is_bust(self.score["player"]):
                reward, done = -1, True
            else:
                reward, done = 0, False
        else:
            assert self._is_bust(self.score["player"]) is False, \
                "Player already goes bust."
            done = True
            if self._is_bust(self.score["dealer"]):
                reward = 1
            else:
                reward = np.sign(
                    self.score["player"] - self.score["dealer"]
                )
        return reward, done


# -------test----------
if __name__ == "__main__":
    env = Easy21()

    try:
        list(map(env.step, [env.action_space.sample(), 0, 1, 3]))
    except AssertionError as e:
        print("Invalid action error caught!")

    ob, r, d, _ = env.step(1)
    assert type(ob) is tuple
    assert env.reward_range[0] <= r <= env.reward_range[1]
    assert type(d) is bool

    assert type(env.reset()) is tuple

    try:
        env.render()
        env.render(mode="human")
    except NotImplementedError as e:
        print("Not implemented error caught!")

    assert type(env.seed()) is list

    assert type(env._draw_hand()) is tuple

    assert type(env._draw_card()) is np.int64

    for item in env._get_observation():
        assert type(item) is int

    assert env._is_bust(1) is False
    assert env._is_bust(21) is False
    assert env._is_bust(0) is True
    assert env._is_bust(22) is True

    env.score["player"], env.score["dealer"] = 5, 5
    reward, done = env._get_reward_and_terminator("player")
    assert reward == 0 and done is False

    env.score["player"], env.score["dealer"] = 5, 5
    reward, done = env._get_reward_and_terminator("dealer")
    assert reward == 0 and done is True

    env.score["player"], env.score["dealer"] = 22, 5
    reward, done = env._get_reward_and_terminator("player")
    assert reward == -1 and done is True

    env.score["player"], env.score["dealer"] = 22, 5
    try:
        reward, done = env._get_reward_and_terminator("dealer")
    except AssertionError as e:
        print("Player goes bust error caught!")

    env.score["player"], env.score["dealer"] = 14, 5
    reward, done = env._get_reward_and_terminator("dealer")
    assert reward == 1 and done is True

    env.score["player"], env.score["dealer"] = 14, 19
    reward, done = env._get_reward_and_terminator("dealer")
    assert reward == -1 and done is True

    env.score["player"], env.score["dealer"] = 14, 0
    reward, done = env._get_reward_and_terminator("dealer")
    assert reward == 1 and done is True

    print("All test passed!")
