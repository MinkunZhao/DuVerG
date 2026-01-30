import re
import json
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx


def _to_int_if_possible(x: Any):
    try:
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return int(x) if float(x).is_integer() else float(x)
        s = str(x).strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d+\.\d+", s):
            return float(s)
    except:
        pass
    return x


def _infer_directed(graph_data: Any, query: str) -> bool:
    q = (query or "").lower()
    if "undirected graph" in q:
        return False
    if "directed graph" in q:
        return True

    if isinstance(graph_data, dict):
        if "directed" in graph_data:
            return bool(graph_data.get("directed"))
        if "undirected" in graph_data:
            return not bool(graph_data.get("undirected"))
        if "is_directed" in graph_data:
            return bool(graph_data.get("is_directed"))

    return False


def _extract_nodes_from_query(query: str) -> List[Any]:
    if not query:
        return []

    m_range = re.search(r"numbered\s+from\s+(\d+)\s+to\s+(\d+)", query, flags=re.I)
    if m_range:
        start_n = int(m_range.group(1))
        end_n = int(m_range.group(2))
        return list(range(start_n, end_n + 1))

    m = re.search(r"among nodes\s+(.+?)(?:\.\s|\.\n|\n|$)", query, flags=re.I | re.S)
    if not m:
        m = re.search(r"among nodes\s+(.+)$", query, flags=re.I | re.S)

    if m:
        return [_to_int_if_possible(x) for x in re.findall(r"-?\d+", m.group(1))]

    return []


def _extract_edges_from_query(query: str) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any, float]]]:
    plain_edges = []
    weighted_edges = []

    if not query:
        return plain_edges, weighted_edges

    for u, v in re.findall(r"\((\-?\d+)\s*,\s*(\-?\d+)\)", query):
        plain_edges.append((_to_int_if_possible(u), _to_int_if_possible(v)))

    for u, v in re.findall(r"\((\-?\d+)\s*->\s*(\-?\d+)\)", query):
        plain_edges.append((_to_int_if_possible(u), _to_int_if_possible(v)))

    for u, v, w in re.findall(r"node\s+(\-?\d+)\s+and\s+node\s+(\-?\d+)\s+with\s+weight\s+(\-?\d+(?:\.\d+)?)", query,
                              flags=re.I):
        weighted_edges.append((_to_int_if_possible(u), _to_int_if_possible(v), float(w)))

    lines = query.split('\n')
    for line in lines:
        if "connected to" not in line:
            continue

        src_match = re.search(r"Node\s+<(\d+)>", line, re.I)
        if not src_match:
            continue
        src = _to_int_if_possible(src_match.group(1))

        target_part = line.split("connected to")[-1]

        raw_targets = re.split(r',\s*(?=<)', target_part)

        for chunk in raw_targets:
            t_match = re.search(r"<(\d+)>", chunk)
            if t_match:
                dst = _to_int_if_possible(t_match.group(1))

                w_match = re.search(r"\(weight:\s*(\d+(?:\.\d+)?)\)", chunk, re.I)
                if w_match:
                    weight = float(w_match.group(1))
                    weighted_edges.append((src, dst, weight))
                else:
                    plain_edges.append((src, dst))

    return plain_edges, weighted_edges


def build_nx_graph(graph_data: Any, query: str = "") -> nx.Graph:
    graph_data = graph_data or {}
    directed = _infer_directed(graph_data, query)
    G: nx.Graph = nx.DiGraph() if directed else nx.Graph()

    nodes: List[Any] = []
    if isinstance(graph_data, dict):
        for k in ("nodes", "node_list", "vertices"):
            if k in graph_data and isinstance(graph_data[k], list):
                nodes = [_to_int_if_possible(x) for x in graph_data[k]]
                break
    if not nodes:
        nodes = _extract_nodes_from_query(query)
    if nodes:
        G.add_nodes_from(nodes)

    edges_added = 0
    if isinstance(graph_data, dict):
        raw_edges = None
        for k in ("edges", "edge_list", "links"):
            if k in graph_data and isinstance(graph_data[k], list):
                raw_edges = graph_data[k]
                break

        if raw_edges:
            for e in raw_edges:
                try:
                    u = v = None
                    weight = None
                    cap = None

                    if isinstance(e, dict):
                        u = _to_int_if_possible(e.get("u", e.get("source")))
                        v = _to_int_if_possible(e.get("v", e.get("target")))
                        if "capacity" in e:
                            cap = float(e["capacity"])
                        if "weight" in e:
                            weight = float(e["weight"])
                        if cap is None and "c" in e:
                            cap = float(e["c"])
                        if weight is None and "w" in e:
                            weight = float(e["w"])

                    elif isinstance(e, (list, tuple)) and len(e) >= 2:
                        u = _to_int_if_possible(e[0])
                        v = _to_int_if_possible(e[1])
                        if len(e) >= 3:
                            w3 = _to_int_if_possible(e[2])
                            try:
                                cap = float(w3)
                                weight = float(w3)
                            except:
                                pass
                        if len(e) >= 4:
                            w4 = _to_int_if_possible(e[3])
                            try:
                                cap = float(w4)
                            except:
                                pass

                    if u is None or v is None:
                        continue

                    attrs = {}
                    if cap is not None:
                        attrs["capacity"] = float(cap)
                    if weight is not None:
                        attrs["weight"] = float(weight)

                    if not attrs:
                        attrs["capacity"] = 1.0

                    G.add_edge(u, v, **attrs)
                    edges_added += 1
                except:
                    continue

    if edges_added == 0:
        plain_edges, weighted_edges = _extract_edges_from_query(query)

        for u, v, w in weighted_edges:
            G.add_edge(u, v, capacity=float(w), weight=float(w))

        for u, v in plain_edges:
            if G.has_edge(u, v):
                continue
            G.add_edge(u, v, capacity=1.0)

        if G.number_of_nodes() == 0:
            nodes2 = _extract_nodes_from_query(query)
            if nodes2:
                G.add_nodes_from(nodes2)

    return G


def graph_stats(G: nx.Graph) -> Dict[str, Any]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 0.0
    if n > 1:
        if G.is_directed():
            density = m / (n * (n - 1))
        else:
            density = 2 * m / (n * (n - 1))
    weighted = any(("weight" in d) for _, _, d in G.edges(data=True))
    capacitated = any(("capacity" in d) for _, _, d in G.edges(data=True))
    return {
        "nodes": n,
        "edges": m,
        "density": round(density, 6),
        "directed": G.is_directed(),
        "weighted": weighted,
        "capacitated": capacitated,
    }


def extract_seed_nodes(query: str) -> List[Union[int, str]]:
    if not query:
        return []

    qpart = query
    if "Q:" in query:
        qpart = query.split("Q:")[-1]
    q = qpart.lower()

    seeds: List[Union[int, str]] = []

    for m in re.findall(r"node\s*([0-9]+)", q):
        try:
            seeds.append(int(m))
        except:
            pass

    m2 = re.search(r"from\s+node\s*([0-9]+)\s+to\s+node\s*([0-9]+)", q)
    if m2:
        try:
            seeds.append(int(m2.group(1)))
            seeds.append(int(m2.group(2)))
        except:
            pass

    seen = set()
    out = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:10]


def locality_preserving_prune(
    G: nx.Graph,
    seed_nodes: List[Union[int, str]],
    hop: int,
    node_budget: int,
    edge_budget: int,
) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G

    seeds_in = [s for s in seed_nodes if s in G]
    if not seeds_in:
        nodes_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
        picked = [n for n, _ in nodes_sorted[: max(1, min(node_budget, len(nodes_sorted)))]]
        H = G.subgraph(picked).copy()
        return _cap_edges(H, edge_budget)

    visited = set(seeds_in)
    q = deque([(s, 0) for s in seeds_in])

    while q and len(visited) < node_budget:
        u, d = q.popleft()
        if d >= hop:
            continue
        for v in G.neighbors(u):
            if v in visited:
                continue
            visited.add(v)
            q.append((v, d + 1))
            if len(visited) >= node_budget:
                break

    H = G.subgraph(list(visited)).copy()
    return _cap_edges(H, edge_budget)


def _cap_edges(H: nx.Graph, edge_budget: int) -> nx.Graph:
    if H.number_of_edges() <= edge_budget:
        return H

    edges = list(H.edges(data=True))
    def score(e):
        d = e[2] or {}
        if "capacity" in d:
            return float(d.get("capacity", 1.0))
        if "weight" in d:
            return float(d.get("weight", 1.0))
        return 1.0

    edges_sorted = sorted(edges, key=score, reverse=True) if edges else []
    keep = edges_sorted[:edge_budget]

    G2 = H.__class__()
    G2.add_nodes_from(H.nodes(data=True))
    for u, v, d in keep:
        G2.add_edge(u, v, **(d or {}))
    return G2


def graph_to_payload(G: nx.Graph) -> Dict[str, Any]:
    links = []
    for u, v, d in G.edges(data=True):
        link = {"source": u, "target": v}
        if d:
            if "weight" in d:
                link["weight"] = d.get("weight")
            if "capacity" in d:
                link["capacity"] = d.get("capacity")
        links.append(link)

    return {
        "directed": G.is_directed(),
        "multigraph": G.is_multigraph(),
        "graph": {},
        "nodes": [{"id": n} for n in G.nodes()],
        "links": links
    }


def solve_connected_nodes(query: str, graph_data: Any) -> Optional[str]:
    m = re.search(r"connected\s+to\s+(\d+)", query.lower())
    if not m:
        return None
    x = int(m.group(1))
    G = build_nx_graph(graph_data, query=query)
    if x not in G:
        return ""
    neigh = sorted(list(G.neighbors(x)))
    return ",".join(str(n) for n in neigh)


def solve_disconnected_nodes(query: str, graph_data: Any) -> Optional[str]:
    m = re.search(r"not\s+connected\s+to\s+(\d+)", query.lower())
    if not m:
        return None
    x = int(m.group(1))
    G = build_nx_graph(graph_data, query=query)
    all_nodes = sorted(list(G.nodes()))
    if x not in G:
        return ",".join(str(n) for n in all_nodes)
    neigh = set(G.neighbors(x))
    ans = [n for n in all_nodes if (n != x and n not in neigh)]
    return ",".join(str(n) for n in ans)


def solve_maximum_flow(query: str, graph_data: Any) -> Optional[str]:
    m = re.search(r"flow\s+from\s+node\s*(\d+)\s+to\s+node\s*(\d+)", query.lower())
    if not m:
        return None
    s = int(m.group(1))
    t = int(m.group(2))

    G = build_nx_graph(graph_data, query=query)
    if s not in G or t not in G:
        return "0"

    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())

    def get_cap(d: Dict[str, Any]) -> float:
        if not d:
            return 1.0
        if "capacity" in d and d["capacity"] is not None:
            try:
                return float(d["capacity"])
            except:
                return 1.0
        if "weight" in d and d["weight"] is not None:
            try:
                return float(d["weight"])
            except:
                return 1.0
        return 1.0

    if G.is_directed():
        for u, v, d in G.edges(data=True):
            cap = get_cap(d)
            if DG.has_edge(u, v):
                DG[u][v]["capacity"] += cap
            else:
                DG.add_edge(u, v, capacity=cap)
    else:
        for u, v, d in G.edges(data=True):
            cap = get_cap(d)
            if DG.has_edge(u, v):
                DG[u][v]["capacity"] += cap
            else:
                DG.add_edge(u, v, capacity=cap)
            if DG.has_edge(v, u):
                DG[v][u]["capacity"] += cap
            else:
                DG.add_edge(v, u, capacity=cap)

    try:
        val = nx.maximum_flow_value(DG, s, t, capacity="capacity")
        if float(val).is_integer():
            return str(int(val))
        return str(val)
    except:
        return None
