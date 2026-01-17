import slaythespire
import sys

class MapEvaluator:
    def __init__(self, gc):
        self.gc = gc
        self.map_info = gc.get_map_info()
        self.nodes = self.map_info['nodes'] # List of floors, each floor list of nodes
        self.max_floor = 14 # Floor 14 is last normal floor, 15 is Boss
        
        # Room Type Enums (based on observation/headers)
        # SHOP=0, REST=1, EVENT=2, ELITE=3, MONSTER=4, TREASURE=5, BOSS=6, BOSS_TREASURE=7, NONE=8, INVALID=9
        self.ROOM_SHOP = 0
        self.ROOM_REST = 1
        self.ROOM_EVENT = 2
        self.ROOM_ELITE = 3
        self.ROOM_MONSTER = 4
        self.ROOM_TREASURE = 5
        self.ROOM_BOSS = 6
        
    def evaluate_path(self):
        """
        Evaluates best next node from current position.
        Returns: next_node_x (int)
        """
        cur_x = self.map_info['current_x']
        cur_y = self.map_info['current_y']
        
        # Identify valid next nodes
        next_nodes = []
        if cur_y == -1:
            # Start of act, any node on floor 0 is valid?
            # Actually usually you pick from floor 0.
            # Let's check map structure. Floor 0 nodes usually have no parents?
            # Or are we selecting for Floor 0?
            # Yes, if cur_y is -1, we choose from Floor 0.
            # All nodes on floor 0 are candidates.
            for n in self.nodes[0]:
                next_nodes.append(n)
        else:
            # We are at cur_y, choose for cur_y + 1
            # Current node object
            # Find current node in list
            # We can just look at children of current node.
            # But wait, map_info doesn't easily give "current node object" without searching.
            # Let's search.
            if cur_y < 14:
                curr_node_obj = None
                for n in self.nodes[cur_y]:
                    if n['x'] == cur_x:
                        curr_node_obj = n
                        break
                
                if curr_node_obj:
                    # Children are X coordinates on next floor
                    children_xs = curr_node_obj['children']
                    # Find these nodes on next floor
                    next_floor = self.nodes[cur_y + 1]
                    for nx in children_xs:
                        for n in next_floor:
                            if n['x'] == nx:
                                next_nodes.append(n)
        
        if not next_nodes:
            # Should not happen unless at boss
            return 0
            
        # Score each next node by simulating best path from it
        best_score = -9999.0
        best_node = next_nodes[0]
        
        path_scores = []
        
        for node in next_nodes:
            score = self._get_path_score_dfs(node, depth=0)
            path_scores.append((node, score))
            
            if score > best_score:
                best_score = score
                best_node = node
                
        # Logging
        # print(f"  [MapEval] Evaluated {len(next_nodes)} options.")
        # for n, s in path_scores:
        #     print(f"    -> Node {n['x']},{n['y']} ({n['symbol']}): {s:.1f}")
            
        return best_node['x'], best_score

    def _get_path_score_dfs(self, node, depth):
        # Base score for this node
        score = self._score_node(node)
        
        # Terminate if at boss or max depth
        # Boss is at floor 15 (index 15?), usually floor 14 connects to it.
        # self.nodes has 15 floors (0-14). Boss is implicit?
        # GameContext.h says `nodes[15][7]`.
        # Floor 14 is the last campfire usually?
        # Let's assume standard Act layout.
        # If node.y == 14, it's the end of path finding (connects to boss).
        if node['y'] >= 14:
            return score
            
        # Recursively find max score of children
        children_xs = node['children']
        if not children_xs:
            return score
            
        max_child_score = -9999.0
        next_floor = self.nodes[node['y'] + 1]
        
        # Optimization: Don't recurse too deep if map is huge, but 15 deep is fine for simple DFS with memoization?
        # Without memoization, branching factor 2-3 -> 2^15 is big.
        # We need MEMOIZATION.
        # But score depends on dynamic state? 
        # Actually, `_score_node` uses `self.gc` which is constant during this evaluation step.
        # So we can memoize based on (x, y).
        
        # Let's add memoization support in class or pass a dict.
        # For simplicity, I'll just do greedy lookahead or limited depth?
        # User asked for "traverse from current to Act Boss".
        # So I should use dynamic programming (work backwards from boss).
        
        # Better approach: DP from top to bottom.
        # But here I am calling DFS. Let's switch to DP if I can.
        # Actually, since I need it for the *current* step, I can just do a recursive search with cache.
        
        return score + self._get_max_future_score(node['y'], node['x'])

    def _get_max_future_score(self, y, x):
        # We want max path value starting from children of (x,y)
        # This function should be cached.
        if not hasattr(self, '_memo'):
            self._memo = {}
            
        key = (x, y)
        if key in self._memo:
            return self._memo[key]
            
        # Find node
        node = None
        for n in self.nodes[y]:
            if n['x'] == x:
                node = n
                break
        if not node: return 0
        
        if y >= 14:
            return 0
            
        children_xs = node['children']
        if not children_xs:
            return 0
            
        max_s = -9999.0
        next_floor = self.nodes[y+1]
        
        for nx in children_xs:
            # Find child node object
            child_node = None
            for n in next_floor:
                if n['x'] == nx:
                    child_node = n
                    break
            
            if child_node:
                # Score of visiting child + max future from child
                s = self._score_node(child_node) + self._get_max_future_score(y+1, nx)
                if s > max_s:
                    max_s = s
                    
        if max_s == -9999.0: max_s = 0
        
        self._memo[key] = max_s
        return max_s

    def _score_node(self, node):
        room = node['room_type']
        
        # Weights
        w_elite = 5.0
        w_rest = 2.0
        w_shop = 1.0
        w_monster = -0.5
        w_event = 0.5
        
        # Dynamic Adjustment based on GC
        hp_ratio = self.gc.cur_hp / self.gc.max_hp
        gold = self.gc.gold
        
        # Low HP Logic
        if hp_ratio < 0.3:
            w_elite = -5.0 # Avoid elites!
            w_rest = 8.0   # Desperately need rest
            w_monster = -2.0 # Avoid fights
            w_shop = 3.0   # Shop can heal/potion
            
        elif hp_ratio < 0.5:
            w_elite = 0.0  # Risky
            w_rest = 4.0
            
        # Rich Logic
        if gold > 250:
            w_shop += 3.0
            
        score = 0.0
        if room == self.ROOM_ELITE:
            score = w_elite
        elif room == self.ROOM_REST:
            score = w_rest
        elif room == self.ROOM_SHOP:
            score = w_shop
        elif room == self.ROOM_MONSTER:
            score = w_monster
        elif room == self.ROOM_EVENT:
            score = w_event
        elif room == self.ROOM_TREASURE:
            score = 2.0
            
        return score
