import re
import ast
import math
import networkx as nx
from typing import Any, List, Optional, Set, Dict, Tuple
from utils.graph_tools import build_nx_graph


class Evaluator:
    def __init__(self, llm_engine: Any):
        self.llm = llm_engine

    def _standardize(self, s: str) -> str:
        s = s.replace("<", "").replace(">", "")
        s = s.replace(",", " ").replace(".", "").replace("[", "").replace("]", "").replace("###", "").replace(
            "The solution is:", "").strip()
        if "```" in s:
            s = re.sub(r"```python|```", "", s).strip()
        return s


    def _as_str_graph(self, G: nx.Graph) -> nx.Graph:
        try:
            if G is None or G.number_of_nodes() == 0:
                return G
            mapping = {n: str(n) for n in list(G.nodes())}
            return nx.relabel_nodes(G, mapping, copy=True)
        except Exception:
            return G

    def _build_graph_fallback_from_question(self, question: str) -> nx.Graph:
        G = nx.Graph()
        if not question:
            return G

        weighted = re.findall(
            r'between node\s+(\d+)\s+and node\s+(\d+)\s+with weight\s+([-+]?\d+(?:\.\d+)?)',
            question, flags=re.I
        )
        for u, v, w in weighted:
            G.add_edge(str(int(u)), str(int(v)), weight=float(w))

        if G.number_of_edges() == 0:
            pairs = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', question)
            for u, v in pairs:
                G.add_edge(str(int(u)), str(int(v)), weight=1.0)

        m = re.search(r'numbered from\s+0\s+to\s+(\d+)', question, flags=re.I)
        if m:
            n = int(m.group(1))
            for i in range(n + 1):
                if str(i) not in G:
                    G.add_node(str(i))

        return G

    def _edge_weight(self, G: nx.Graph, u: str, v: str) -> Optional[float]:
        if G is None:
            return None
        data = G.get_edge_data(u, v)
        if data is None:
            return None

        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            ws = []
            for _, attr in data.items():
                try:
                    ws.append(float(attr.get("weight", 1.0)))
                except Exception:
                    ws.append(1.0)
            return min(ws) if ws else 1.0

        try:
            return float(data.get("weight", 1.0))
        except Exception:
            return 1.0

    def _parse_yes_no(self, s: str) -> Optional[bool]:
        if s is None:
            return None
        t = s.lower()
        if any(w in t for w in ["yes", "true"]):
            return True
        if any(w in t for w in ["no", "false", "impossible", "cannot", "not possible"]):
            return False
        return None

    def _extract_path_nodes(self, text: str, src: Optional[str] = None, dst: Optional[str] = None) -> List[str]:
        if not text:
            return []

        m = re.search(r'\[[^\]]+\]', text)
        if m:
            try:
                obj = ast.literal_eval(m.group(0))
                if isinstance(obj, (list, tuple)) and len(obj) >= 2:
                    nodes = []
                    for x in obj:
                        sx = str(x).strip().strip("'\"")
                        if re.fullmatch(r'\d+', sx):
                            nodes.append(str(int(sx)))
                    if len(nodes) >= 2:
                        return nodes
            except Exception:
                pass

        nums = [str(int(x)) for x in re.findall(r'\d+', text)]
        if not nums:
            return []

        if src is not None and dst is not None:
            best = None

            for i in range(len(nums)):
                if nums[i] == src:
                    for j in range(i + 1, len(nums)):
                        if nums[j] == dst:
                            cand = nums[i:j + 1]
                            if best is None or len(cand) > len(best):
                                best = cand

            for i in range(len(nums)):
                if nums[i] == dst:
                    for j in range(i + 1, len(nums)):
                        if nums[j] == src:
                            cand = nums[i:j + 1]
                            if best is None or len(cand) > len(best):
                                best = cand

            if best is not None and len(best) >= 2:
                return best

        return nums


    def evaluate(self, benchmark_name: str, task_type: str, prediction: str, ground_truth: str,
                 question: str = "") -> bool:
        p_clean = self._standardize(prediction)
        g_clean = self._standardize(ground_truth)
        t_type = task_type.lower()
        bench_name = benchmark_name.lower()

        if (not prediction or str(prediction).strip() == "") and 'connected' not in t_type:
            return False

        try:
            G = build_nx_graph({}, query=question)
        except:
            G = nx.Graph()

        G = self._as_str_graph(G)
        if G is None or G.number_of_nodes() == 0:
            G = self._build_graph_fallback_from_question(question)
            G = self._as_str_graph(G)

        try:
            if 'dfs' in t_type:
                return self._eval_dfs_graph_instruct(question, prediction)

            if 'bipartite' in t_type:
                return self._eval_bipartite_graph_instruct(question, ground_truth, prediction)

            if 'matching' in t_type:
                return self._eval_matching_rigorous(p_clean, g_clean, question)

            if 'hamilton' in t_type and ('nlgraph' in bench_name or 'graphinstruct' in bench_name):
                return self._eval_hamilton_oracle(prediction, ground_truth, G, question)

            if 'shortest_path' in t_type and 'nlgraph' in bench_name:
                return self._verify_shortest_path(prediction, ground_truth, G, question)

            if 'shortest_path' in t_type and 'talk_like_a_graph' in bench_name:
                if 'There is no path' in ground_truth:
                    if any(x in prediction for x in ['No', 'no', 'None', 'none', '-1', '0']):
                        return True
                    else:
                        return False
                return self._verify_numeric(p_clean, g_clean)

            if 'topology' in t_type and 'graphwiz' in bench_name:
                return self._verify_topological_sort(prediction, G)

            if any(x in t_type for x in ['flow', 'count', 'max_flow', 'jaccard', 'clustering', 'coefficient']):
                return self._verify_numeric(p_clean, g_clean)

            if 'common_neighbor' in t_type or 'degree' in t_type:
                return self._verify_numeric(p_clean, g_clean)

            if 'diameter' in t_type:
                return self._verify_numeric(p_clean, g_clean)

            if 'nodes' in t_type or 'connected' in t_type or 'neighbors' in t_type or 'page_rank' in t_type:
                return self._verify_sets(p_clean, g_clean)

            if 'predecessor' in t_type or 'neighbor' in t_type:
                return self._verify_sets(p_clean, g_clean)

            if 'gnn' in t_type:
                return self._eval_gnn_robust(p_clean, g_clean)

            if self._is_boolean(t_type, g_clean):
                return self._verify_boolean(p_clean, g_clean)

        except Exception as e:
            print(f"Evaluator Warning: {e}")

        return self._normalize(g_clean) in self._normalize(p_clean)

    def _extract_nums(self, s: str) -> List[str]:
        return re.findall(r'\d+', s)

    def _extract_graph_instruct_struct(self, question: str) -> Tuple[nx.Graph, Optional[int]]:
        is_directed = "directed" in question.lower()
        G = nx.DiGraph() if is_directed else nx.Graph()

        lines = question.strip().split('\n')
        for line in lines:
            numbers = re.findall(r'<(\d+)>', line)
            if not numbers:
                continue
            node = int(numbers[0])
            connections = [int(n) for n in numbers[1:]]
            for conn in connections:
                G.add_edge(node, conn)

        match = re.search(r'Start from node\s*<(\d+)>', question)
        start_node = int(match.group(1)) if match else None

        return G, start_node

    def _eval_dfs_graph_instruct(self, question: str, prediction: str) -> bool:
        try:
            graph, start = self._extract_graph_instruct_struct(question)

            path_nums = re.findall(r'\d+', prediction)
            path = [int(num) for num in path_nums]

            if not path:
                return False

            if start is None:
                start = path[0]

            visited = set()
            stack = [start]
            visited.add(start)
            path_index = 1

            if path[0] != start:
                return False

            while stack and path_index < len(path):
                current = stack[-1]
                neighbors = list(graph.neighbors(current))

                unvisited_neighbors = [n for n in neighbors if n not in visited]

                next_node_in_path = path[path_index]

                if next_node_in_path in unvisited_neighbors:
                    stack.append(next_node_in_path)
                    visited.add(next_node_in_path)
                    path_index += 1
                elif not unvisited_neighbors:
                    stack.pop()
                else:
                    return False

            return path_index == len(path) and len(visited) == len(graph)

        except Exception:
            return False


    def _eval_bipartite_graph_instruct(self, question: str, gt: str, prediction: str) -> bool:
        try:
            def extract_pairs(s):
                nums = re.findall(r'\d+', s)
                return [(int(nums[i]), int(nums[i + 1])) for i in range(0, len(nums), 2)]

            result_pairs = extract_pairs(prediction)
            gt_pairs = extract_pairs(gt)

            if len(result_pairs) != len(gt_pairs):
                return False

            def extract_set(q, key):
                start = q.find(key)
                if start == -1: return set()
                sub = q[start + len(key):]
                if 'Nodes set' in sub:
                    pass
                line = sub.split('\n')[0]
                return set([int(x) for x in re.findall(r'\d+', line)])

            node_set_1 = extract_set(question, 'Nodes set 1 contains: ')
            node_set_2 = extract_set(question, 'Nodes set 2 contains: ')

            set1_nodes_used = set()
            set2_nodes_used = set()

            for u, v in result_pairs:
                if u in node_set_1 and v in node_set_2:
                    pass
                elif u in node_set_2 and v in node_set_1:
                    u, v = v, u
                else:
                    return False

                if u in set1_nodes_used or v in set2_nodes_used:
                    return False

                set1_nodes_used.add(u)
                set2_nodes_used.add(v)

            return True
        except Exception:
            return False


    def _verify_shortest_path(self, pred: str, gt: str, G: nx.Graph, question: str = "") -> bool:
        if G is None or G.number_of_nodes() == 0:
            G = self._build_graph_fallback_from_question(question)
            G = self._as_str_graph(G)

        if G is None or G.number_of_nodes() == 0:
            p_nodes = self._extract_path_nodes(pred)
            g_nodes = self._extract_path_nodes(gt)
            return (p_nodes == g_nodes) or (p_nodes == list(reversed(g_nodes)))

        src = dst = None
        m = re.search(r'from node\s+(\d+)\s+to node\s+(\d+)', question, flags=re.I)
        if m:
            src, dst = str(int(m.group(1))), str(int(m.group(2)))

        p_nodes = self._extract_path_nodes(pred, src=src, dst=dst)
        if not p_nodes or len(p_nodes) < 2:
            return False

        if src is not None and dst is not None:
            ok_dir = (p_nodes[0] == src and p_nodes[-1] == dst) or (p_nodes[0] == dst and p_nodes[-1] == src)
            if not ok_dir:
                p_nodes = self._extract_path_nodes(self._standardize(pred), src=src, dst=dst)
                if not p_nodes:
                    return False
                ok_dir = (p_nodes[0] == src and p_nodes[-1] == dst) or (p_nodes[0] == dst and p_nodes[-1] == src)
                if not ok_dir:
                    return False

        pred_w = 0.0
        for i in range(len(p_nodes) - 1):
            u, v = p_nodes[i], p_nodes[i + 1]
            w = self._edge_weight(G, u, v)
            if w is None:
                return False
            pred_w += w

        if src is not None and dst is not None:
            try:
                true_w = nx.shortest_path_length(G, source=src, target=dst, weight="weight")
            except Exception:
                try:
                    true_w = nx.shortest_path_length(G, source=src, target=dst)
                except Exception:
                    return self._verify_numeric(pred, gt)

            return abs(float(pred_w) - float(true_w)) < 1e-6

        g_nodes = self._extract_path_nodes(gt)
        if g_nodes and (p_nodes == g_nodes or p_nodes == list(reversed(g_nodes))):
            return True

        return self._verify_numeric(pred, gt)


    def _verify_topological_sort(self, pred: str, G: nx.Graph) -> bool:
        p_nodes = self._extract_nums(pred)
        if not p_nodes or len(p_nodes) != G.number_of_nodes():
            return False

        try:
            pos = {str(node): i for i, node in enumerate(p_nodes)}
            for u, v in G.edges():
                if pos[str(u)] > pos[str(v)]:
                    return False
            return True
        except:
            return False

    def _verify_numeric(self, p: str, g: str) -> bool:

        def get_last_num(s):
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", s.replace(',', ''))
            return float(nums[-1]) if nums else None

        p_, g_ = p, g
        if len(p) >= 1 and len(g) >= 1:
            if p[0] == '0' and float(p[1:]) > 0:
                p_ = p[0] + '.' + p[1:]
            elif p[-1] == '0' and float(p[:-1]) >= 0:
                p_ = p[:-1] + '.' + p[-1]
            if g[0] == '0' and float(g[1:]) > 0:
                g_ = g[0] + '.' + g[1:]
            elif g[-1] == '0' and float(g[:-1]) >= 0:
                g_ = g[:-1] + '.' + g[-1]

        val_p = get_last_num(p_)
        val_g = get_last_num(g_)

        if val_p is None or val_g is None:
            return False

        return math.isclose(val_p, val_g, rel_tol=0.01, abs_tol=0.001)

    def _verify_sets(self, p: str, g: str) -> bool:
        if (p == "" or "not in the graph" in p) and g in ["", "no nodes", "none", "[]", "empty", "no such nodes", "none.", " No nodes.", "No nodes"]:
            return True
        p_set = set(self._extract_nums(p))
        g_set = set(self._extract_nums(g))
        return p_set == g_set

    def _is_boolean(self, t, g):
        return any(x in t for x in
                   ['cycle', 'reachability', 'connectivity', 'substructure', 'edge_existence']) or g.lower() in [
            'yes', 'no', 'true', 'false']

    def _verify_boolean(self, p, g):
        def parse(s):
            s = s.lower()
            if any(w in s for w in ['yes', 'true', '1']): return True
            if any(w in s for w in ['no', 'false', '0']): return False
            return None

        return parse(p) == parse(g)

    def _normalize(self, s):
        return re.sub(r'[^a-zA-Z0-9]', '', s).lower()


    def _eval_matching_rigorous(self, pred: str, gt: str, query: str) -> bool:
        interests = set()
        lines = query.split('\n')
        for line in lines:
            if 'interested' in line.lower() or 'job' in line.lower():
                nums = re.findall(r'\d+', line)
                if len(nums) >= 2:
                    app_id = int(nums[0])
                    for job_id_str in nums[1:]:
                        interests.add((app_id, int(job_id_str)))

        gt_target_match = re.search(r'(\d+)\s+(?:applicants|pairs|matches)', gt.lower())
        if gt_target_match:
            target_count = int(gt_target_match.group(1))
        else:
            gt_assignments = re.findall(r'(?:applicant|node)?\s*(\d+)[\s:\->=]+(?:job|node)?\s*(\d+)', gt.lower())
            target_count = len(set(gt_assignments))

        if target_count == 0:
            target_count = len(re.findall(r'\d+\s*[:\-]\s*\d+', gt))


        pred_clean = pred.replace('\n', ' , ')

        candidates = re.findall(r'(?:applicant|node)?\s*(\d+)[\s:\->=,]+(?:job|node)?\s*(\d+)', pred_clean.lower())

        assigned_apps = set()
        assigned_jobs = set()
        valid_matches = []

        for m in candidates:
            u, v = int(m[0]), int(m[1])

            final_app, final_job = None, None

            if (u, v) in interests:
                final_app, final_job = u, v
            elif (v, u) in interests:
                final_app, final_job = v, u
            else:
                continue

            if final_app not in assigned_apps and final_job not in assigned_jobs:
                assigned_apps.add(final_app)
                assigned_jobs.add(final_job)
                valid_matches.append((final_app, final_job))

        actual_count = len(valid_matches)

        return actual_count >= target_count and target_count > 0


    def _eval_hamilton_oracle(self, pred: str, gt: str, G: nx.Graph, question: str) -> bool:

        path = self._extract_path_nodes(pred)
        if not path:
            return self._parse_yes_no(pred) == self._parse_yes_no(gt)

        if G.number_of_nodes() == 0:
            G = self._build_graph_fallback_from_question(question)
            G = self._as_str_graph(G)

        num_nodes = G.number_of_nodes()

        if len(path) == num_nodes + 1 and len(path) > 1:
            if str(path[0]) == str(path[-1]):
                path = path[:-1]

        if len(path) != num_nodes:
            return False

        if len(set(path)) != len(path):
            return False

        str_to_node = {str(n): n for n in G.nodes()}
        graph_nodes_str = set(str_to_node.keys())

        if not set(str(n) for n in path).issubset(graph_nodes_str):
            return False

        is_directed = G.is_directed()

        real_path = [str_to_node[str(n)] for n in path]

        for i in range(len(real_path) - 1):
            u, v = real_path[i], real_path[i + 1]
            if is_directed:
                if not G.has_edge(u, v):
                    return False
            else:
                if not G.has_edge(u, v) and not G.has_edge(v, u):
                    return False

        return True


    def _eval_gnn_robust(self, pred: str, gt: str) -> bool:
        def get_embs(text):
            d = {}
            matches = re.findall(r'(?:node|Node)\s*(\d+)[\s:]*(-?\d+[\.\d]*)\s*[,\s]\s*(-?\d+[\.\d]*)', text)
            for nid, v1, v2 in matches:
                d[int(nid)] = [round(float(v1)), round(float(v2))]
            return d

        p_embs, g_embs = get_embs(pred), get_embs(gt)
        return p_embs == g_embs if g_embs else False


