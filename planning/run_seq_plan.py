import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import json
import pickle
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import traceback

from assets.load import load_part_ids
from planning.sequence.sim_string import get_body_color_dict
from planning.sequence.physics_planner import MultiPartPathPlanner, get_contact_graph


DEBUG = False


class SequencePlanner:
    '''
    Disassembly sequence planning (with ground, fixed pose, one part at a time, dual arms)
    '''
    def __init__(self, asset_folder, assembly_dir, G_preced, grasps, save_sdf=False, contact_eps=None): # TODO: add parallel search

        self.asset_folder = asset_folder
        self.assembly_dir = assembly_dir
        self.parts = sorted(load_part_ids(self.assembly_dir))
        assert len(self.parts) >= 2

        self.G_preced = G_preced
        self.G_preced_undirected = G_preced.to_undirected()
        self.G_contact = get_contact_graph(self.asset_folder, self.assembly_dir, self.parts, save_sdf=save_sdf, contact_eps=contact_eps)
        self.save_sdf = save_sdf

        self.grasps = grasps['grasps']
        for part in self.parts: # convert list to dict for faster search
            self.grasps[part]['move'] = {grasp[0].grasp_id: grasp for grasp in self.grasps[part]['move']}
            self.grasps[part]['hold'] = {grasp.grasp_id: grasp for grasp in self.grasps[part]['hold']}
        self.grasp_id_pairs = grasps['grasp_id_pairs']
        self.gripper = grasps['gripper']
        self.arm = grasps['arm']

        self.t_start = None
        self.stop_msg = None
        self.N_path = None # for DFS

    def _select_node(self, tree):
        N_last = self.N_path[-1]
        if len(N_last[0]) == 1 and N_last[1] == 'hold': # leaf node
            if not tree.edges[self.N_path[-2], self.N_path[-1]]['feasible']:
                tree.nodes[self.N_path[-1]]['feasible'] = False
            self.N_path.pop()
            return self._select_node(tree)
        elif tree.out_degree(N_last) < len(N_last[0]): # current node has unexplored child
            return N_last
        else: # fully explored, backtrack
            if len(self.N_path) == 1:
                return None # root node, no more nodes to explore
            else:
                for edge in tree.out_edges(self.N_path[-1]):
                    if tree.edges[edge]['feasible']:
                        break
                else:
                    tree.edges[self.N_path[-2], self.N_path[-1]]['feasible'] = False # fully explored, mark infeasible
                    tree.nodes[self.N_path[-1]]['feasible'] = False
                self.N_path.pop()
                return self._select_node(tree)
            
    def _check_grasp_move_valid(self, grasp_move_id, parts_remain, part_move):
        grasp_move = self.grasps[part_move]['move'][grasp_move_id][0]
        parts_in_collision_move = grasp_move.parts_in_collision_move
        return set(parts_remain).intersection(parts_in_collision_move) == set()
        
    def _check_grasp_hold_valid(self, grasp_hold_id, parts_remain, part_hold):
        grasp_hold = self.grasps[part_hold]['hold'][grasp_hold_id]
        parts_in_collision_hold = grasp_hold.parts_in_collision_hold
        return set(parts_remain).intersection(parts_in_collision_hold['fix']) == set()

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
    
    def _select_edge(self, tree, node):
        node_action = node[1]
        if node_action == 'hold':
            return self._select_edge_move(tree, node)
        elif node_action == 'move':
            return self._select_edge_hold(tree, node)
        else:
            raise ValueError(f'[run_seq_plan] Invalid node action {node_action} ({self.assembly_dir})')

    def _select_edge_move(self, tree, node):
        is_root = len(node[0]) == len(self.parts)
        if is_root:
            assert len(self.N_path) == 1
            E_last = None
            part_hold = None
        else:
            assert len(self.N_path) >= 3 and len(self.N_path) % 2 == 1 # need to have at least 2 edge to select next move node
            E_last = tree.edges[self.N_path[-2], self.N_path[-1]]
            part_hold = E_last['part']
        
        # find parts that can be moved
        parts_move_cand = []
        for part_move in node[0]:
            next_node = (tuple([part for part in node[0] if part != part_move]), 'move', part_move)
            if is_root and tree.has_node(next_node): # whole branch is tried before
                if DEBUG: print(f'[run_seq_plan] {part_move} tried before ({self.assembly_dir})')
                continue
            if set(self.G_preced.nodes[part_move]['parts_before']).intersection(node[0]) != set(): # some parts in node are not disassembled before part_move
                if DEBUG: print(f'[run_seq_plan] {part_move} has parts before ({self.assembly_dir})')
                continue
            G_contact_rest = self.G_contact.subgraph([part for part in node[0] if part != part_move])
            if not nx.is_connected(G_contact_rest): # removing part_move will lead to disconnected graph
                if DEBUG: print(f'[run_seq_plan] {part_move} removal leads to disconnected graph ({self.assembly_dir})')
                continue
            parts_move_cand.append(part_move)
        if len(parts_move_cand) == 0:
            return {}
        
        # find valid grasp pairs considering last part_hold (or single grasps if root)
        n_valid_grasp_pairs = {}
        valid_grasp_move_ids = {}
        if is_root:
            for part_move in parts_move_cand:
                valid_grasp_move_ids[part_move] = set()
                for grasp in self.grasps[part_move]['move'].values():
                    grasp_move_id = grasp[0].grasp_id
                    if self._check_grasp_move_valid(grasp_move_id, node[0], part_move):
                        valid_grasp_move_ids[part_move].add(grasp_move_id)
                n_valid_grasp_pairs[part_move] = len(valid_grasp_move_ids[part_move])
        else:
            valid_grasp_hold_ids = E_last['grasp_ids']
            for part_move in parts_move_cand:
                next_node = (tuple([part for part in node[0] if part != part_move]), 'move', part_move)
                valid_grasp_id_pairs = []
                for (grasp_move_id, grasp_hold_id) in self.grasp_id_pairs[(part_move, part_hold)]:
                    if grasp_hold_id in valid_grasp_hold_ids:
                        if self._check_grasp_pair_valid((grasp_move_id, grasp_hold_id), node[0], part_move, part_hold):
                            valid_grasp_id_pairs.append((grasp_move_id, grasp_hold_id))
                valid_grasp_move_ids[part_move] = set([grasp_id_pair[0] for grasp_id_pair in valid_grasp_id_pairs])
                if tree.has_edge(node, next_node): # check if part_move has been tried to move before
                    last_grasp_ids = set(tree.edges[node, next_node]['grasp_ids'])
                    valid_grasp_move_ids[part_move] = valid_grasp_move_ids[part_move].difference(last_grasp_ids)
                    n_valid_grasp_pairs[part_move] = 0
                    for valid_grasp_id_pair in valid_grasp_id_pairs:
                        if valid_grasp_id_pair[0] in valid_grasp_move_ids[part_move]:
                            n_valid_grasp_pairs[part_move] += 1
                else:
                    n_valid_grasp_pairs[part_move] = len(valid_grasp_id_pairs) # part_move has not been tried to move before
        
        for part_move in parts_move_cand.copy():
            if n_valid_grasp_pairs[part_move] == 0:
                valid_grasp_move_ids.pop(part_move)
                parts_move_cand.remove(part_move)
        
        # sort parts by number of valid grasp pairs
        parts_move_cand = sorted(parts_move_cand, key=lambda part_move: -n_valid_grasp_pairs[part_move])
        return {part_move: valid_grasp_move_ids[part_move] for part_move in parts_move_cand}
    
    def _select_edge_hold(self, tree, node):
        assert len(self.N_path) >= 2 and len(self.N_path) % 2 == 0 # need to have at least 1 edge to select next hold node
        E_last = tree.edges[self.N_path[-2], self.N_path[-1]]
        part_move = E_last['part']

        # find parts that can be held
        parts_hold_cand = []
        G_preced_curr = self.G_preced.subgraph(node[0] + (part_move,))
        for part_hold in node[0]:
            if part_move in self.G_preced.nodes[part_hold]['parts_after']: # part_hold cannot block part_move
                if DEBUG: print(f'[run_seq_plan] {part_hold} cannot block {part_move} ({self.assembly_dir})')
                continue
            # NOTE: should be turned off when doing ablation study on part sequencing
            # if not self.G_contact.has_edge(part_move, part_hold): # part_move needs to be in contact with part_hold
            #     if DEBUG: print(f'[run_seq_plan] {part_move} not in contact with {part_hold} ({self.assembly_dir})')
            #     continue
            if not nx.has_path(self.G_preced, part_hold, part_move) and G_preced_curr.in_degree(part_move) > 0: # hold parts that precede (depend on) part_move if possible
                if DEBUG: print(f'[run_seq_plan] {part_hold} does not precede {part_move} ({self.assembly_dir})')
                continue
            parts_hold_cand.append(part_hold)
        if len(parts_hold_cand) == 0:
            return {}
        
        # find valid grasp pairs considering part_move
        n_valid_grasp_pairs = {}
        valid_grasp_hold_ids = {}
        valid_grasp_move_ids = E_last['grasp_ids']
        for part_hold in parts_hold_cand:
            next_node = (tuple([part for part in node[0]]), 'hold', part_hold)
            valid_grasp_id_pairs = []
            for (grasp_move_id, grasp_hold_id) in self.grasp_id_pairs[(part_move, part_hold)]:
                if grasp_move_id in valid_grasp_move_ids:
                    if self._check_grasp_pair_valid((grasp_move_id, grasp_hold_id), node[0], part_move, part_hold):
                        valid_grasp_id_pairs.append((grasp_move_id, grasp_hold_id))
            valid_grasp_hold_ids[part_hold] = set([grasp_id_pair[1] for grasp_id_pair in valid_grasp_id_pairs])
            if tree.has_edge(node, next_node): # check if part_hold has been tried to hold before
                last_grasp_ids = set(tree.edges[node, next_node]['grasp_ids'])
                valid_grasp_hold_ids[part_hold] = valid_grasp_hold_ids[part_hold].difference(last_grasp_ids)
                n_valid_grasp_pairs[part_hold] = 0
                for valid_grasp_id_pair in valid_grasp_id_pairs:
                    if valid_grasp_id_pair[1] in valid_grasp_hold_ids[part_hold]:
                        n_valid_grasp_pairs[part_hold] += 1
            else:
                n_valid_grasp_pairs[part_hold] = len(valid_grasp_id_pairs) # part_hold has not been tried to hold before
        
        for part_hold in parts_hold_cand.copy():
            if n_valid_grasp_pairs[part_hold] == 0:
                valid_grasp_hold_ids.pop(part_hold)
                parts_hold_cand.remove(part_hold)
        
        # sort parts by number of valid grasp pairs
        parts_hold_cand = sorted(parts_hold_cand, key=lambda part_hold: -n_valid_grasp_pairs[part_hold])
        return {part_hold: valid_grasp_hold_ids[part_hold] for part_hold in parts_hold_cand}

    def plan(self, early_term=False, timeout=None, debug=False):
        '''
        Main planning function
        Output:
            tree: disassembly tree including all disassembly attempts
            - node: (parts, action, part_active), {feasible}
                    action means the next action to take at the current node: 'move' or 'hold' (assembly order)
                    part_active is the part on which the next action is taken
            - edge: (node, node), {part, grasp_ids, feasible}
        '''
        self.t_start = time()
        self.stop_msg = None
        solution_found = False

        N0 = (tuple(self.parts.copy()), 'hold', None)
        tree = nx.DiGraph()
        tree.add_node(N0, feasible=True)
        self.N_path = [N0]

        try:
            while True:

                if timeout is not None and (time() - self.t_start) > timeout:
                    self.stop_msg = 'timeout'
                    break

                N = self._select_node(tree)
                if N is None:
                    self.stop_msg = f'tree fully explored (solution found: {solution_found})' # should have been checked above
                    break
                if debug:
                    print(f'[run_seq_plan] selected node: {N}')

                edge_cands = self._select_edge(tree, N)
                parts_infeasible = set(N[0]).difference(set(edge_cands.keys()))
                
                for part_cand, grasp_ids in edge_cands.items():
                    assert len(grasp_ids) > 0
                    if debug:
                        print(f'[run_seq_plan] selected part: {part_cand}, grasps: {len(grasp_ids)}')

                    if N[1] == 'move':
                        N_next = (N[0], 'hold', part_cand)
                    elif N[1] == 'hold':
                        N_next = (tuple([part for part in N[0] if part != part_cand]), 'move', part_cand)
                    else:
                        raise NotImplementedError

                    if tree.has_edge(N, N_next):
                        E = tree.edges[N, N_next]
                        if E['feasible'] or len(grasp_ids.difference(E['grasp_ids'])) > 0:
                            E['feasible'] = True
                            E['grasp_ids'] = E['grasp_ids'].union(grasp_ids)
                        else:
                            parts_infeasible.add(part_cand)
                            continue
                    else:
                        if not tree.has_node(N_next):
                            tree.add_node(N_next, feasible=True)
                        else:
                            tree.nodes[N_next]['feasible'] = True
                        tree.add_edge(N, N_next, part=part_cand, grasp_ids=grasp_ids, feasible=True)
                        if debug:
                            print(f'[run_seq_plan] added feasible edge from {N} to {N_next}')
                    
                    self.N_path.append(N_next)
                    break

                for part_infeasible in parts_infeasible: # add edges for infeasible parts
                    if N[1] == 'move':
                        N_next = (N[0], 'hold', part_infeasible)
                    elif N[1] == 'hold':
                        N_next = (tuple([part for part in N[0] if part != part_infeasible]), 'move', part_infeasible)
                    else:
                        raise NotImplementedError

                    if tree.has_edge(N, N_next):
                        pass # there is other feasible way
                    else:
                        if not tree.has_node(N_next):
                            tree.add_node(N_next, feasible=False)
                        tree.add_edge(N, N_next, part=part_infeasible, grasp_ids=set(), feasible=False)
                        if debug:
                            print(f'[run_seq_plan] added infeasible edge from {N} to {N_next}')

                # self.plot_tree(tree)

                # successfully reached leaf node
                if len(self.N_path[-1][0]) == 1 and self.N_path[-1][1] == 'hold':
                    solution_found = True
                    if early_term:
                        self.stop_msg = 'solution found'
                        break

        except (Exception, KeyboardInterrupt) as e:
            if type(e) == KeyboardInterrupt:
                self.stop_msg = 'interrupt'
            else:
                self.stop_msg = 'exception'
            print(e, f'from {self.assembly_dir}')
            print(traceback.format_exc())

        assert self.stop_msg is not None, '[run_seq_plan] bug: unexpectedly stopped'
        if debug:
            print(f'[run_seq_plan] stopped: {self.stop_msg}')

        return tree

    @staticmethod
    def plot_tree(tree, save_path=None):
        from networkx.drawing.nx_agraph import graphviz_layout
        plt.figure(figsize=(15, 15))
        node_colors = ['g' if tree.nodes[node]['feasible'] else 'r' for node in tree.nodes]
        # node_labels = {node: f'{node[0]}' for node in tree.nodes}
        node_labels = {}
        for node in tree.nodes:
            if len(node[0]) == 1: # leaf node
                node_labels[node] = f'{node[0][0]}'
            elif tree.in_degree(node) == 0: # root node
                node_labels[node] = ','.join([str(x) for x in node[0]])
        edge_colors = ['g' if tree.edges[edge]['feasible'] else 'r' for edge in tree.edges]
        # edge_labels = {edge: f'{edge[1][1][0]} {tree.edges[edge]["part"]} ({len(tree.edges[edge]["grasp_ids"])})' for edge in tree.edges}
        edge_labels = {edge: f'{edge[1][1][0]} {tree.edges[edge]["part"]}' for edge in tree.edges}
        pos = graphviz_layout(tree, prog='dot')
        nx.draw(tree, pos, labels=node_labels, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, font_size=10)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    def sample_sequence(self, tree, seed):
        '''
        [(part_move, part_hold), ...], [(grasp_move, grasp_hold), ...]
        NOTE: only find one random sequence
        '''
        np.random.seed(seed)

        # find leaf node
        leaf_node = None
        tree_edges = list(tree.edges)
        np.random.shuffle(tree_edges)
        for edge in tree_edges:
            edge_info = tree.edges[edge]
            if edge_info['feasible'] and len(edge[0][0]) == 1:
                leaf_node = edge[1]
                break
        else:
            return None, None

        # find sequence in tree from bottom to top
        sequence = []
        node = leaf_node
        while tree.in_degree(node) > 0:
            parent_nodes = list(tree.predecessors(node))
            np.random.shuffle(parent_nodes)
            for parent_node in parent_nodes:
                edge_info = tree.edges[parent_node, node]
                if edge_info['feasible']:
                    sequence.insert(0, edge_info['part'])
                    node = parent_node
                    break
        assert len(sequence) % 2 == 0, f'invalid sequence: {sequence}'

        # reformat sequence (assembly order)
        sequence = [(sequence[i], sequence[i + 1]) for i in range(0, len(sequence), 2)][::-1]

        grasps_sequence = []
        parts_assembled = tuple((sequence[0][1],)) # initial part_hold
        for i, (part_move, part_hold) in enumerate(sequence):
            next_part_hold = sequence[i + 1][1] if i + 1 < len(sequence) else None
            node_initial = (parts_assembled, 'hold', part_hold)
            node_after_hold = (parts_assembled, 'move', part_move)
            node_after_move = (tuple(sorted(parts_assembled + (part_move,))), 'hold', next_part_hold)
            edge_hold = tree.edges[node_after_hold, node_initial]
            edge_move = tree.edges[node_after_move, node_after_hold]
            grasp_hold_ids = edge_hold['grasp_ids']
            grasp_move_ids = edge_move['grasp_ids']
            grasp_id_pairs = list(self.grasp_id_pairs[(part_move, part_hold)])
            np.random.shuffle(grasp_id_pairs)
            for grasp_id_pair in grasp_id_pairs:
                if grasp_id_pair[0] in grasp_move_ids and grasp_id_pair[1] in grasp_hold_ids:
                    if self._check_grasp_pair_valid(grasp_id_pair, node_after_move[0], part_move, part_hold):
                        grasp_move = self.grasps[part_move]['move'][grasp_id_pair[0]]
                        grasp_hold = self.grasps[part_hold]['hold'][grasp_id_pair[1]]
                        grasps_sequence.append((grasp_move, grasp_hold))
                        break
            else:
                raise ValueError(f'[run_seq_plan] No valid grasp pair found for {part_move}-{part_hold} ({self.assembly_dir})')
            parts_assembled += (part_move,)
            parts_assembled = tuple(sorted(parts_assembled))

        sequence, grasps_sequence = sequence[::-1], grasps_sequence[::-1] # reverse sequence to disassembly sequence
        return sequence, grasps_sequence

    @staticmethod
    def check_success(tree):
        success = False
        for edge in tree.edges:
            edge_info = tree.edges[edge]
            if edge_info['feasible'] and len(edge[1][0]) == 1:
                success = True
        return success

    def log(self, tree, log_dir, plot=False):
        '''
        Log planned disassembly sequence and gripper statistics
        '''
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'tree.pkl'), 'wb') as fp:
            pickle.dump(tree, fp)
        
        stats_path = os.path.join(log_dir, 'stats.json')
        with open(stats_path, 'r') as fp:
            stats = json.load(fp)
        stats['seq_plan'] = {
            'success': SequencePlanner.check_success(tree),
            'time': round(time() - self.t_start, 2),
            'stop_msg': self.stop_msg,
        }
        with open(stats_path, 'w') as fp:
            json.dump(stats, fp)
        
        if plot:
            self.plot_tree(tree, save_path=os.path.join(log_dir, 'tree.png'))

    def render(self, sequence, record_dir=None):
        '''
        Render planned disassembly sequence
        '''
        parts_assembled = self.parts.copy()
        parts_removed = []

        if record_dir is not None:
            os.makedirs(record_dir, exist_ok=True)

        for i, (part_move, part_hold) in enumerate(sequence):
            parts_rest = parts_assembled.copy()
            parts_rest.remove(part_move)
            parts_free = [part_i for part_i in parts_rest if part_i != part_hold] + [part_move]
            path = np.array(self.G_preced.nodes[part_move]['path'])

            if record_dir is not None:
                record_path = os.path.join(record_dir, f'{i}_{part_move}.gif')
            else:
                record_path = None

            path_planner = MultiPartPathPlanner(self.asset_folder, self.assembly_dir, parts_rest, part_move, parts_removed=parts_removed, pose=np.eye(4), save_sdf=self.save_sdf)

            body_color_dict = get_body_color_dict([part_hold], parts_free) # visualize fixes
            path_planner.sim.set_body_color_map(body_color_dict)
            path_planner.render(path=path, record_path=record_path)

            parts_assembled = parts_rest
            parts_removed.append(part_move)


def run_seq_plan(assembly_dir, log_dir, early_term=False, plot=False, verbose=False):
    asset_folder = os.path.join(project_base_dir, './assets')

    precedence_path = os.path.join(log_dir, 'precedence.pkl')
    if not os.path.exists(precedence_path):
        print(f'[run_seq_plan] {precedence_path} not found')
        return
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not os.path.exists(grasps_path):
        print(f'[run_seq_plan] {grasps_path} not found')
        return
    
    with open(precedence_path, 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(grasps_path, 'rb') as fp:
        grasps = pickle.load(fp)

    seq_planner = SequencePlanner(asset_folder, assembly_dir, G_preced, grasps, save_sdf=True, contact_eps=None)
    tree = seq_planner.plan(early_term=early_term, timeout=None, debug=verbose)
    seq_planner.log(tree, log_dir, plot=plot)
    # seq_planner.render(stats['sequence'], record_dir=None)

    contact_graph_path = os.path.join(log_dir, 'contact.pkl')
    with open(contact_graph_path, 'wb') as fp:
        pickle.dump(seq_planner.G_contact, fp)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True, help='directory of assembly')
    parser.add_argument('--log-dir', type=str, required=True, help='directory to load precedence and save generated grasps')
    parser.add_argument('--early-term', action='store_true', default=False, help='early termination')
    parser.add_argument('--plot', action='store_true', default=False, help='plot tree')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
    args = parser.parse_args()

    run_seq_plan(args.assembly_dir, args.log_dir, early_term=args.early_term, plot=args.plot, verbose=args.verbose)
