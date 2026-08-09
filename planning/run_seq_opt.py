import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import networkx as nx
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import json
from tqdm import tqdm
from functools import total_ordering
from scipy.spatial.transform import Rotation as R
import random


@total_ordering
class GraspScore:
    def __init__(self, scores: list, maximize: list):
        self.scores = scores
        self.maximize = maximize
        assert len(scores) == len(maximize)

    def __eq__(self, other):
        return all(score == other_score for score, other_score in zip(self.scores, other.scores))

    def __gt__(self, other):
        assert len(self.scores) == len(other.scores)
        assert self.maximize == other.maximize
        for score, other_score, maximize in zip(self.scores, other.scores, self.maximize):
            if score > other_score:
                return maximize
            elif score < other_score:
                return not maximize
        return False
    
    def __iter__(self):
        return iter(self.scores)
    
    def __getitem__(self, index):
        return self.scores[index]

    def __setitem__(self, index, value):
        self.scores[index] = value

    def copy(self):
        return GraspScore(self.scores.copy(), self.maximize.copy())


class SequenceOptimizer:

    def __init__(self, G_preced, grasps):
        self.G_preced = G_preced
        self.grasps = grasps['grasps']
        self.parts = list(self.grasps.keys())
        for part in self.parts: # convert list to dict for faster search
            self.grasps[part]['move'] = {grasp[0].grasp_id: grasp for grasp in self.grasps[part]['move']}
            self.grasps[part]['hold'] = {grasp.grasp_id: grasp for grasp in self.grasps[part]['hold']}
        self.grasp_id_pairs = grasps['grasp_id_pairs']
        self.gripper = grasps['gripper']
        self.arm = grasps['arm']
        self.offset_delta = grasps['settings']['offset_delta']

    def _find_root_node(self, tree):
        for node in tree.nodes:
            if tree.in_degree(node) == 0:
                return node
        raise ValueError(f'[run_seq_opt] Root node not found')

    def _check_grasp_pair_valid(self, grasp_id_pair, parts_remain, part_move, part_hold):
        if grasp_id_pair not in self.grasp_id_pairs[(part_move, part_hold)]:
            return False
        grasp_move_id, grasp_hold_id = grasp_id_pair
        grasp_move = self.grasps[part_move]['move'][grasp_move_id][0]
        grasp_hold = self.grasps[part_hold]['hold'][grasp_hold_id]
        parts_in_collision_move = grasp_move.parts_in_collision_move
        parts_in_collision_hold = grasp_hold.parts_in_collision_hold
        move_valid = set(parts_remain).intersection(parts_in_collision_move) == set() 
        hold_valid = set(parts_remain).intersection(parts_in_collision_hold['fix']) == set() and part_move not in parts_in_collision_hold['move']
        return move_valid and hold_valid
    
    def _check_sequence_valid(self, sequence, grasps_sequence):
        remaining_parts = list(self.parts)
        for (move_part, hold_part), (move_grasp, hold_grasp) in zip(sequence, grasps_sequence):
            grasp_id_pair = (move_grasp[0].grasp_id, hold_grasp.grasp_id)
            if not self._check_grasp_pair_valid(grasp_id_pair, remaining_parts, move_part, hold_part):
                return False
            remaining_parts.remove(move_part)
        return True
    
    def _convert_part_tree_to_grasp_tree(self, part_tree, verbose=False):
        '''
        Convert part-based tree (parts as nodes and edges have grasps) to grasp-based tree (grasps as nodes)
        '''
        grasp_tree = nx.DiGraph() # node format: (assembled_parts, part_id, grasp_id)

        part_root_node = self._find_root_node(part_tree)
        parent_nodes = {part_root_node}
        num_layers = len(self.parts) - 1 # number of move-hold layers

        for i in tqdm(range(2 * num_layers - 1), desc='tree conversion', disable=not verbose):
            child_nodes = set()

            for parent_node in parent_nodes:
                if not part_tree.nodes[parent_node]['feasible']: continue

                for child_node in part_tree.successors(parent_node):
                    if not part_tree.nodes[child_node]['feasible']: continue
                    parent_edge_info = part_tree.edges[parent_node, child_node]
                    if not parent_edge_info['feasible']: continue

                    child_nodes.add(child_node)

                    parent_part = parent_edge_info['part']
                    parent_grasp_ids = parent_edge_info['grasp_ids']

                    if i == 0: # root node
                        for parent_grasp_id in parent_grasp_ids:
                            grasp_tree.add_edge((parent_node[0], None, None), (child_node[0], parent_part, parent_grasp_id))

                    for grandchild_node in part_tree.successors(child_node):
                        if not part_tree.nodes[grandchild_node]['feasible']: continue
                        child_edge_info = part_tree.edges[child_node, grandchild_node]
                        if not child_edge_info['feasible']: continue

                        child_part = child_edge_info['part']
                        child_grasp_ids = child_edge_info['grasp_ids']

                        parts_remain = list(grandchild_node[0])

                        for parent_grasp_id in parent_grasp_ids:
                            for child_grasp_id in child_grasp_ids:

                                if child_node[1] == 'move':
                                    part_move, part_hold = parent_part, child_part
                                    grasp_move_id, grasp_hold_id = parent_grasp_id, child_grasp_id
                                elif child_node[1] == 'hold':
                                    part_move, part_hold = child_part, parent_part
                                    grasp_move_id, grasp_hold_id = child_grasp_id, parent_grasp_id
                                else:
                                    raise NotImplementedError

                                if not self._check_grasp_pair_valid((grasp_move_id, grasp_hold_id), parts_remain, part_move, part_hold):
                                    continue

                                grasp_tree.add_edge((child_node[0], parent_part, parent_grasp_id), (grandchild_node[0], child_part, child_grasp_id))

                        if i == 2 * num_layers - 2:
                            assert part_tree.out_degree(grandchild_node) == 0 # last layer should be leaf nodes

            parent_nodes = child_nodes
            if len(parent_nodes) == 0: # tree is not complete
                return None

        return grasp_tree
    
    def _get_move_grasp_score(self, grasp_move):
        move_part = grasp_move.part_id
        part_contact_points = []
        for predecessor in self.G_preced.predecessors(move_part):
            part_contact_points.extend(self.G_preced.edges[predecessor, move_part]['contact_points'])
        part_contact_points = np.array(part_contact_points)
        if len(part_contact_points) == 0:
            return 0
        grasp_contact_points = np.array(grasp_move.contact_points)
        part_com = self.G_preced.nodes[move_part]['com']
        path = self.G_preced.nodes[move_part]['path']
        contact_direction = path[-1][:3] - path[0][:3]
        contact_direction /= np.linalg.norm(contact_direction)
        part_torque = np.sum(np.cross(part_contact_points - part_com, contact_direction / len(part_contact_points)), axis=0)
        grasp_torque = np.sum(np.cross(grasp_contact_points - part_com, -contact_direction / len(grasp_contact_points)), axis=0)
        net_torque = part_torque + grasp_torque
        grasp_score = np.linalg.norm(net_torque) / len(grasp_contact_points)
        return grasp_score

    def _get_hold_grasp_score(self, grasp_hold, grasp_move):
        grasp_score = (R.from_quat(grasp_hold.quat[[1, 2, 3, 0]]).inv() * R.from_quat(grasp_move.quat[[1, 2, 3, 0]])).magnitude()
        grasp_score *= len(grasp_hold.contact_points)
        return grasp_score
    
    def _get_all_move_grasp_scores(self):
        grasp_scores = {}
        for part in self.parts:
            if self.G_preced.nodes[part]['path'] is None:
                continue # base part
            grasp_scores[part] = {}
            for grasp_id, grasp in self.grasps[part]['move'].items():
                grasp_score = self._get_move_grasp_score(grasp[0])
                grasp_scores[part][grasp_id] = grasp_score
        return grasp_scores
    
    def _calculate_average_random_sequence_score(self, grasp_tree, verbose=False):
        
        def generate_random_sequence(root_node, num_layers):
            sequence = [root_node]
            current_node = root_node
            for _ in range(num_layers):
                successors = list(grasp_tree.successors(current_node))
                if not successors:
                    break  # No more moves possible
                next_move = random.choice(successors)
                sequence.append(next_move)
                current_node = next_move

            return sequence
        
        def calculate_grasp_score(sequence):
            sum_move_score = 0
            sum_hold_score = 0
            num_stable_part_hold = 0
            num_grasp_switch = 0

            move_grasp_scores = self._get_all_move_grasp_scores()  # Assuming this function is defined

            for i in range(1, len(sequence) - 1, 2):  # Iterate over move-hold pairs
                parent_node = sequence[i - 1]
                move_node = sequence[i]
                hold_node = sequence[i + 1]

                _, move_part, move_grasp_id = move_node
                _, hold_part, hold_grasp_id = hold_node

                move_grasp = self.grasps[move_part]['move'][move_grasp_id][0]
                hold_grasp = self.grasps[hold_part]['hold'][hold_grasp_id]

                move_score = move_grasp_scores[move_part][move_grasp_id]
                hold_score = self._get_hold_grasp_score(hold_grasp, move_grasp)  # Assuming this is defined

                sum_move_score += move_score
                sum_hold_score += hold_score

                # Check stability
                G_preced_curr = self.G_preced.subgraph(parent_node[0])
                if nx.has_path(G_preced_curr, hold_part, move_part):
                    if G_preced_curr.has_edge(hold_part, move_part):
                        if_stable_part_hold = True
                    else:
                        if_stable_part_hold = all(
                            G_preced_curr.out_degree(base_part) != 1
                            for base_part in G_preced_curr.predecessors(move_part)
                        )
                else:
                    if_stable_part_hold = False

                num_stable_part_hold += int(if_stable_part_hold)

                # Check grasp switch
                if_grasp_switch = not (hold_node[1] == parent_node[1] and hold_node[2] == parent_node[2])
                num_grasp_switch += int(if_grasp_switch)

            return sum_move_score, sum_hold_score, num_stable_part_hold, num_grasp_switch
        
        root_node = self._find_root_node(grasp_tree)  # Assuming this function is defined
        num_layers = len(self.parts) - 1

        move_grasp_scores = []
        hold_grasp_scores = []
        stable_holds = []
        grasp_switches = []

        # for _ in tqdm(range(100), desc='Random Grasp Sequences'):
        for _ in range(100):
            sequence = generate_random_sequence(root_node, num_layers)
            move_score, hold_score, stable_hold, grasp_switch = calculate_grasp_score(sequence)
            move_grasp_scores.append(move_score)
            hold_grasp_scores.append(hold_score)
            stable_holds.append(stable_hold)
            grasp_switches.append(grasp_switch)

        average_move_score = np.mean(move_grasp_scores)
        average_hold_score = np.mean(hold_grasp_scores)
        average_stable_hold = np.mean(stable_holds)
        average_grasp_switch = np.mean(grasp_switches)

        return [average_stable_hold, average_grasp_switch, average_move_score, average_hold_score]

    def _optimize_part_tree_from_grasp_tree(self, grasp_tree, verbose=False): 
        '''
        Goal: find optimal paths in the tree with the best grasp stability
        '''
        root_node = self._find_root_node(grasp_tree)

        num_layers = len(self.parts) - 1
        DP = [{} for _ in range(num_layers + 1)] # DP[layer][node] = GraspScore(num_stable_part_hold, num_grasp_switch, sum_move_score, sum_hold_score)
        prev = [{} for _ in range(num_layers + 1)] # prev[layer][node] = (curr_move_node, prev_hold_node)
        
        ABLATION = 'original'
        # ABLATION = 'woseq'
        # ABLATION = 'wograsp'
        # ABLATION = 'asap'
        
        if ABLATION == 'original':
            maximize = [True, False, False, True]
        elif ABLATION == 'woseq':
            maximize = [False, True]
        elif ABLATION == 'wograsp':
            maximize = [True, False]
        elif ABLATION == 'asap':
            maximize = [True, True]
        else:
            raise NotImplementedError

        if ABLATION == 'original':
            root_grasp_score = GraspScore([0, 0, 0, 0], maximize)
        else:
            root_grasp_score = GraspScore([0, 0], maximize)

        if ABLATION == 'original':
            worst_grasp_score = GraspScore([0, np.inf, np.inf, -np.inf], maximize)
        elif ABLATION == 'woseq':
            worst_grasp_score = GraspScore([np.inf, -np.inf], maximize)
        elif ABLATION == 'wograsp':
            worst_grasp_score = GraspScore([0, np.inf], maximize)
        elif ABLATION == 'asap':
            worst_grasp_score = GraspScore([-np.inf, -np.inf], maximize)
        else:
            raise NotImplementedError
        np.random.seed(42)

        DP[0] = {root_node: root_grasp_score}
        prev[0] = {root_node: None}
        parent_nodes = [root_node]
        
        move_grasp_scores = self._get_all_move_grasp_scores()

        # calculate DP values
        for layer in tqdm(range(1, num_layers + 1), desc='dynamic programming', disable=not verbose):

            for parent_node in parent_nodes: # parent_node: last hold, child_node: current move, grandchild_node: current hold
                G_preced_curr = self.G_preced.subgraph(parent_node[0]) # including current move and current hold

                for child_node in grasp_tree.successors(parent_node):
                    for grandchild_node in grasp_tree.successors(child_node):
                        if grandchild_node not in DP[layer]:
                            DP[layer][grandchild_node] = worst_grasp_score.copy()
                            prev[layer][grandchild_node] = None

                        _, move_part, move_grasp_id = child_node
                        _, hold_part, hold_grasp_id = grandchild_node
                        move_grasp, hold_grasp = self.grasps[move_part]['move'][move_grasp_id][0], self.grasps[hold_part]['hold'][hold_grasp_id]
                        # move_score = self._get_move_grasp_score(move_grasp)
                        move_score = move_grasp_scores[move_part][move_grasp_id]
                        hold_score = self._get_hold_grasp_score(hold_grasp, move_grasp)
                        
                        # calculate new cost, update DP and prev
                        if nx.has_path(G_preced_curr, hold_part, move_part):
                            if G_preced_curr.has_edge(hold_part, move_part):
                                if_stable_part_hold = True # hold_part is directly connected to move_part, stable
                            else:
                                for base_part in G_preced_curr.predecessors(move_part):
                                    if G_preced_curr.out_degree(base_part) == 1:
                                        if_stable_part_hold = False # unstable base_part connected to move_part and not being held, not stable
                                        break
                                else:
                                    if_stable_part_hold = True # all base_parts connected to move_part are stable, stable
                        else:
                            if_stable_part_hold = False # hold_part is not directly or indirectly connected to move_part, not stable
                        if_grasp_switch = False if grandchild_node[1] == parent_node[1] and grandchild_node[2] == parent_node[2] else True

                        if ABLATION == 'original' or ABLATION == 'wograsp':
                            num_stable_part_hold = DP[layer - 1][parent_node][0] + int(if_stable_part_hold)
                            num_grasp_switch = DP[layer - 1][parent_node][1] + int(if_grasp_switch)
                        if ABLATION == 'original':
                            sum_move_score = DP[layer - 1][parent_node][2] + move_score
                            sum_hold_score = DP[layer - 1][parent_node][3] + hold_score
                        if ABLATION == 'woseq':
                            sum_move_score = DP[layer - 1][parent_node][0] + move_score
                            sum_hold_score = DP[layer - 1][parent_node][1] + hold_score

                        if ABLATION == 'original':
                            new_score = GraspScore([num_stable_part_hold, num_grasp_switch, sum_move_score, sum_hold_score], maximize)
                        elif ABLATION == 'woseq':
                            new_score = GraspScore([sum_move_score, sum_hold_score], maximize)
                        elif ABLATION == 'wograsp':
                            new_score = GraspScore([num_stable_part_hold, num_grasp_switch], maximize)
                        elif ABLATION == 'asap':
                            new_score = GraspScore([np.random.random(), np.random.random()], maximize)
                        else:
                            raise NotImplementedError
                        if new_score > DP[layer][grandchild_node]:
                            DP[layer][grandchild_node] = new_score
                            prev[layer][grandchild_node] = (child_node, parent_node)
            
            parent_nodes = list(DP[layer].keys())

            if len(parent_nodes) == 0: # tree is not complete
                return None
            if layer == num_layers:
                for parent_node in parent_nodes:
                    assert grasp_tree.out_degree(parent_node) == 0 # last layer should be leaf nodes

        # find best leaf node and optimal cost
        best_leaf_node = None
        best_score = None
        for leaf_node in DP[num_layers]:
            if best_leaf_node is None:
                best_leaf_node = leaf_node
                best_score = DP[num_layers][leaf_node]
            else:
                if DP[num_layers][leaf_node] > DP[num_layers][best_leaf_node]:
                    best_leaf_node = leaf_node
                    best_score = DP[num_layers][leaf_node]

        # backtrace to find optimal path
        optimal_path = [best_leaf_node]
        for layer in range(num_layers, 0, -1):
            optimal_path.extend(list(prev[layer][optimal_path[-1]]))
        optimal_path = optimal_path[::-1]
        
        # build optimal part tree
        optimal_tree = nx.DiGraph()
        for i in range(0, len(optimal_path) - 2, 2):
            optimal_tree.add_edge(optimal_path[i], optimal_path[i + 2], 
                move_part=optimal_path[i + 1][1], hold_part=optimal_path[i + 2][1],
                move_grasp_id=optimal_path[i + 1][2], hold_grasp_id=optimal_path[i + 2][2]
            )
        return optimal_tree, best_score.scores

    def optimize(self, tree, verbose=False):
        grasp_tree = self._convert_part_tree_to_grasp_tree(tree, verbose=verbose)
        if grasp_tree is None: return None
        optimal_tree, optimal_scores = self._optimize_part_tree_from_grasp_tree(grasp_tree, verbose=verbose)

        # random_scores = self._calculate_average_random_sequence_score(grasp_tree, verbose=verbose)
        # latex_str = ''
        # for rs, os in zip(random_scores, optimal_scores):
        #     latex_str += f'& {os:.2f} / {rs:.2f} '
        # latex_str += r'\\'

        return optimal_tree
    
    @staticmethod
    def plot_tree(tree, save_path=None):
        from networkx.drawing.nx_agraph import graphviz_layout
        plt.figure(figsize=(15, 15))
        node_colors = ['g' for node in tree.nodes]
        node_labels = {node: ','.join([str(x) for x in node[0]]) for node in tree.nodes}
        edge_colors = ['g' for edge in tree.edges]
        edge_labels = {edge: f'm {tree.edges[edge]["move_part"]} h {tree.edges[edge]["hold_part"]}' for edge in tree.edges}
        pos = graphviz_layout(tree, prog='dot')
        nx.draw(tree, pos, labels=node_labels, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, font_size=10)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    def get_sequence(self, tree): # get optimized sequence from tree
        root_node = self._find_root_node(tree)
        sequence = []
        grasps_sequence = []
        parent_node = root_node
        while tree.out_degree(parent_node) > 0:
            child_nodes = list(tree.successors(parent_node))
            assert len(child_nodes) == 1 # tree should be an optimized chain
            child_node = child_nodes[0]
            edge_info = tree.edges[parent_node, child_node]
            move_part, hold_part = edge_info['move_part'], edge_info['hold_part']
            move_grasp_id, hold_grasp_id = edge_info['move_grasp_id'], edge_info['hold_grasp_id']
            move_grasp = self.grasps[move_part]['move'][move_grasp_id]
            hold_grasp = self.grasps[hold_part]['hold'][hold_grasp_id]
            sequence.append((move_part, hold_part))
            grasps_sequence.append((move_grasp, hold_grasp))
            parent_node = child_node
        assert self._check_sequence_valid(sequence, grasps_sequence)
        return sequence, grasps_sequence
    

def run_seq_opt(log_dir, plot=False, verbose=False):

    tree_path = os.path.join(log_dir, 'tree.pkl')
    if not os.path.exists(tree_path):
        print(f'[run_seq_opt] {tree_path} not found')
        return
    precedence_path = os.path.join(log_dir, 'precedence.pkl')
    if not os.path.exists(precedence_path):
        print(f'[run_seq_opt] {precedence_path} not found')
        return
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not os.path.exists(grasps_path):
        print(f'[run_seq_opt] {grasps_path} not found')
        return
    
    with open(tree_path, 'rb') as f:
        tree = pickle.load(f)
    with open(precedence_path, 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(grasps_path, 'rb') as fp:
        grasps = pickle.load(fp)
    
    t_start = time()
    seq_optimizer = SequenceOptimizer(G_preced, grasps)
    new_tree = seq_optimizer.optimize(tree, verbose=verbose)

    if new_tree is None:
        print(f'[run_seq_opt] Failed sequence planning in {log_dir}')
        return
    
    with open(os.path.join(log_dir, 'tree_opt.pkl'), 'wb') as f:
        pickle.dump(new_tree, f)
    
    stats_path = os.path.join(log_dir, 'stats.json')
    with open(stats_path, 'r') as fp:
        stats = json.load(fp)
    stats['seq_opt'] = {'time': round(time() - t_start, 2)}
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp)
    
    if plot:
        seq_optimizer.plot_tree(new_tree, save_path=os.path.join(log_dir, 'tree_opt.png'))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True, help='directory to load precedence and save generated grasps')
    parser.add_argument('--plot', action='store_true', default=False, help='plot tree')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    run_seq_opt(args.log_dir, plot=args.plot, verbose=args.verbose)
