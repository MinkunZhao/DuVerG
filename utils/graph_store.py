import os
import gzip
from collections import defaultdict, deque
from functools import lru_cache
from typing import Dict, Any, List, Tuple, Optional, Union
import networkx as nx


def _open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


class EdgeListGraphStore:
    def __init__(
            self,
            path: str,
            directed: bool = False,
            weighted: bool = False,
            delimiter: Optional[str] = None,
            comment_prefix: str = "#",
    ):
        self.path = path
        self.directed = directed
        self.weighted = weighted
        self.delimiter = delimiter
        self.comment_prefix = comment_prefix
        self._loaded = False
        self._adj: Dict[int, List[Union[int, Tuple[int, float]]]] = defaultdict(list)
        self._nodes = set()
        self._m = 0
        self._n = 0

    def load(self):
        if self._loaded:
            return
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"graph_file not found: {self.path}")

        with _open_maybe_gz(self.path) as f:
            for line in f:
                line = line.strip()
                if not line or (self.comment_prefix and line.startswith(self.comment_prefix)):
                    continue
                parts = line.split(self.delimiter) if self.delimiter else line.split()
                if len(parts) < 2:
                    continue
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    w = float(parts[2]) if (self.weighted and len(parts) >= 3) else 1.0
                except Exception:
                    continue

                self._nodes.add(u)
                self._nodes.add(v)
                if self.weighted:
                    self._adj[u].append((v, w))
                    if not self.directed:
                        self._adj[v].append((u, w))
                else:
                    self._adj[u].append(v)
                    if not self.directed:
                        self._adj[v].append(u)
                self._m += 1
        self._n = len(self._nodes)
        self._loaded = True

    def stats(self) -> Dict[str, Any]:
        if not self._loaded:
            self.load()
        n = self._n
        m = self._m if self.directed else self._m
        density = 0.0
        if n > 1:
            density = (m / (n * (n - 1))) if self.directed else (2 * m / (n * (n - 1)))
        return {
            "nodes": n,
            "edges": m,
            "density": round(density, 6),
            "directed": self.directed,
            "weighted": self.weighted,
            "capacitated": False,
        }

    def neighbors(self, u: int) -> List[int]:
        if not self._loaded:
            self.load()
        if self.weighted:
            return [v for (v, _) in self._adj.get(u, [])]
        return list(self._adj.get(u, []))

    def prune_khop(
            self,
            seed_nodes: List[int],
            hop: int,
            node_budget: int,
            edge_budget: int,
    ) -> nx.Graph:
        if not self._loaded:
            self.load()

        seeds = [s for s in seed_nodes if s in self._nodes]
        if not seeds:
            degs = []
            for u, neigh in self._adj.items():
                degs.append((u, len(neigh)))
            degs.sort(key=lambda x: x[1], reverse=True)
            seeds = [u for (u, _) in degs[:5]]

        visited = set(seeds)
        q = deque([(s, 0) for s in seeds])

        while q:
            if len(visited) >= node_budget and q[0][1] >= 1:
                break

            u, d = q.popleft()
            if d >= hop:
                continue

            neighs = self.neighbors(u)
            if d > 0 and len(neighs) > 500:
                pass

            for v in neighs:
                if v in visited:
                    continue
                visited.add(v)
                q.append((v, d + 1))

                if d > 0 and len(visited) >= node_budget:
                    break

        H = nx.DiGraph() if self.directed else nx.Graph()
        H.add_nodes_from(list(visited))

        added_edges = 0
        nodes_list = list(visited)
        seed_set = set(seeds)

        nodes_list.sort(key=lambda x: (0 if x in seed_set else 1, x))

        for u in nodes_list:
            if added_edges >= edge_budget:
                break

            raw_neigh = self._adj.get(u, [])


            if self.weighted:
                for v, w in raw_neigh:
                    if v in visited:
                        if added_edges < edge_budget or u in seed_set:
                            H.add_edge(u, v, weight=w, capacity=w)
                            added_edges += 1
            else:
                for v in raw_neigh:
                    if v in visited:
                        if added_edges < edge_budget or u in seed_set:
                            H.add_edge(u, v, capacity=1.0)
                            added_edges += 1

        return H


@lru_cache(maxsize=4)
def get_graph_store(path: str, directed: bool = False, weighted: bool = False, delimiter: Optional[str] = None,
                    comment_prefix: str = "#") -> EdgeListGraphStore:
    store = EdgeListGraphStore(path=path, directed=directed, weighted=weighted, delimiter=delimiter,
                               comment_prefix=comment_prefix)
    store.load()
    return store