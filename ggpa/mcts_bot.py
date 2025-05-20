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
        # store children as a list of (GameAction, TreeNode)
        self.children: list[tuple] = []
        self.parent = parent
        self.results: list[float] = []
        self.param = param

    # one MCTS iteration
    def step(self, state: BattleState):
        self.select(state)

    # pick the action with highest average score
    def get_best(self, state: BattleState):
        best_action = None
        best_avg = -float('inf')
        for action, node in self.children:
            if not node.results:
                continue
            avg = sum(node.results) / len(node.results)
            if avg > best_avg:
                best_avg = avg
                best_action = action
        return best_action or random.choice(state.get_actions())

    # debug-print
    def print_tree(self, indent=0):
        spacer = ' ' * indent
        for action, node in self.children:
            visits = len(node.results)
            avg = sum(node.results) / visits if visits else 0.0
            print(f"{spacer}{action}: visits={visits}, avg={avg:.3f}")
            node.print_tree(indent + 2)

    # selection → expansion or recurse
    def select(self, state: BattleState):
        # 1) if terminal, backpropagate and stop
        if state.ended():
            self.backpropagate(self.score(state))
            return

        actions = state.get_actions()
        # 2) find unexplored actions
        explored = [a for a, _ in self.children]
        unexplored = [a for a in actions if a not in explored]

        if unexplored:
            # expand one new child
            self.expand(state, unexplored)
        else:
            # fully expanded: pick via UCB-1
            total_visits = sum(len(n.results) for _, n in self.children)
            best_val = -float('inf')
            best_act = best_node = None

            for action, node in self.children:
                visits = len(node.results)
                avg = sum(node.results) / visits
                ucb = avg + self.param * math.sqrt(2 * math.log(total_visits) / visits)
                if ucb > best_val:
                    best_val = ucb
                    best_act, best_node = action, node

            # descend into best child
            state.step(best_act)
            best_node.select(state)

    # add a child for one unexplored action, rollout, backpropagate
    def expand(self, state: BattleState, available: list):
        action = random.choice(available)
        child = TreeNode(self.param, parent=self)
        self.children.append((action, child))

        state.step(action)
        result = child.rollout(state)
        child.backpropagate(result)

    # random playout to terminal
    def rollout(self, state: BattleState):
        while not state.ended():
            action = random.choice(state.get_actions())
            state.step(action)
        return self.score(state)

    # record result here and bubble up
    def backpropagate(self, result: float):
        self.results.append(result)
        if self.parent:
            self.parent.backpropagate(result)

    # default evaluation
    def score(self, state: BattleState):
        return state.score()


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
