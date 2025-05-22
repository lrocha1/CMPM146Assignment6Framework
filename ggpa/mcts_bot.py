from __future__ import annotations
import math
import random
import time
from agent import Agent
from battle import BattleState
from card import Card
from action.action import EndAgentTurn, PlayCard
from game import GameState
from ggpa.ggpa import GGPA
from config import Verbose

# You only need to modify the TreeNode!
class TreeNode:
    def __init__(self, param, parent=None):
        # children: list of (GameAction, TreeNode)
        self.children: list[tuple] = []
        self.parent = parent
        self.results: list[float] = []
        self.param = param

    def step(self, state: BattleState):
        self.select(state)

    def get_best(self, state: BattleState):
        best_action = None
        best_avg = -float('inf')
        for ga, node in self.children:
            if not node.results:
                continue
            avg = sum(node.results) / len(node.results)
            if avg > best_avg:
                best_avg = avg
                best_action = ga
        return best_action or random.choice(state.get_actions())

    def print_tree(self, indent=0):
        spacer = ' ' * indent
        for ga, node in self.children:
            visits = len(node.results)
            avg = sum(node.results) / visits if visits else 0.0
            print(f"{spacer}{ga}: visits={visits}, avg={avg:.3f}")
            node.print_tree(indent + 2)

    def select(self, state: BattleState):
        # 1) terminal? record and stop
        if state.ended():
            self.backpropagate(self.score(state))
            return

        actions = state.get_actions()
        explored = [ga for ga, _ in self.children]
        unexplored = [ga for ga in actions if ga not in explored]

        if unexplored:
            self.expand(state, unexplored)
        else:
            # UCB-1 selection
            total = sum(len(n.results) for _, n in self.children)
            best_ucb = -float('inf')
            best_ga = best_node = None

            for ga, node in self.children:
                v = len(node.results)
                mean = sum(node.results) / v
                ucb = mean + self.param * math.sqrt(2 * math.log(total) / v)
                if ucb > best_ucb:
                    best_ucb, best_ga, best_node = ucb, ga, node

            # **apply** that action to the sample‐state
            self._apply(state, best_ga)
            best_node.select(state)

    def expand(self, state: BattleState, available: list):
        ga = random.choice(available)
        child = TreeNode(self.param, parent=self)
        self.children.append((ga, child))

        # **apply** that action
        self._apply(state, ga)
        result = child.rollout(state)
        child.backpropagate(result)

    def rollout(self, state: BattleState):
        while not state.ended():
            ga = random.choice(state.get_actions())
            self._apply(state, ga)
        return self.score(state)

    def backpropagate(self, result: float):
        self.results.append(result)
        if self.parent:
            self.parent.backpropagate(result)

    def score(self, state: BattleState):
        return state.score()

    def _apply(self, state: BattleState, ga):
        """ Convert GameAction → PlayCard/EndAgentTurn and mutate `state`. """
        # end‐turn
        if ga.card is None:
            state.tick_player(EndAgentTurn())
            return

        name, upg = ga.card
        for idx, c in enumerate(state.hand):
            if c.name == name and c.upgrade_count == upg:
                state.tick_player(PlayCard(idx))
                return
        # (This really should never happen, but just in case…)
        state.tick_player(EndAgentTurn())


# You do not have to modify the MCTSAgent (but you can)
class MCTSAgent(GGPA):
    def __init__(self, iterations: int, verbose: bool, param: float):
        self.iterations = iterations
        self.verbose = verbose
        self.param = param

    # REQUIRED METHOD
    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        actions = battle_state.get_actions()
        if len(actions) == 1:
            return actions[0].to_action(battle_state)

        root = TreeNode(self.param)
        for _ in range(self.iterations):
            sample = battle_state.copy_undeterministic()
            root.step(sample)

        best = root.get_best(battle_state)
        if self.verbose:
            root.print_tree()
        return best.to_action(battle_state)

    # REQUIRED METHOD: All our scenarios only have one enemy
    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return agent_list[0]

    # REQUIRED METHOD: Our scenarios do not involve targeting cards
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return card_list[0]
