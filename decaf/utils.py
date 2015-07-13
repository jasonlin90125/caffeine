# -*- coding: utf-8 -*-
"""
functions for DeCAF

Created on Wed Apr  8 09:41:59 2015
by Marta Stepniewska
"""

from decaf import PHARS, COLORS, Pharmacophore
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.font_manager import FontManager


def compare_nodes(n1, n2):
    """Compare types of two nodes. Return unnormalised similarity score and new
    dictionary of pharmacophoric properties for nodes combination.

    Args:
       n1, n2 (dict): nodes to compare

    Returns:
      float: unnormalised similarity score
      dict: pharmacophoric properties for nodes combination
    """
    if not isinstance(n1, dict):
        raise TypeError("Invalid n1! Expected dict, got %s instead" %
                        type(n1).__name__)
    if not isinstance(n2, dict):
        raise TypeError("Invalid n2! Expected dict, got %s instead" %
                        type(n2).__name__)

    if not Pharmacophore.check_node(n1):
        raise ValueError("Invalid n1!")

    if not Pharmacophore.check_node(n2):
        raise ValueError("Invalid n2!")

    c = n1["freq"] + n2["freq"]
    d1 = sum(n1["type"].values())
    d2 = sum(n2["type"].values())
    d = d1 + d2
    sim = 0.0
    t = {}

    for phar in PHARS:
        if phar in n1["type"] and phar in n2["type"]:
            sim += (n1["type"][phar] + n2["type"][phar]) / d
            t[phar] = n1["type"][phar] + n2["type"][phar]
        elif phar in n1["type"]:
            t[phar] = n1["type"][phar]
        elif phar in n2["type"]:
            t[phar] = n2["type"][phar]
    return sim * c, t


def __POS(p, start):
    """Compute partial ordering set (POS).

    Args:
       p (Pharmacophore): 
       start (int): node index to start from

    Returns:
       list: POS representation, list of dictionaries with:
          * idx (int): position in ordering
          * prev (int): previous node in graph (not in ordering!)
          * i (int): node id
          * out_dg (float): node out-degree


    see: Xu, Jun. "GMA: a generic match algorithm for structural homomorphism,
         isomorphism, and maximal common substructure match and its
         applications." J. Chem. Inf. Comput. Sci., 1996, 36 (1), 25–34
    """
    stack = deque([start])
    seen = []
    order = [{"prev": None, "i": i, "idx": None,
             "out_dg": np.sum(p.edges[i] > 0)} for i in xrange(p.numnodes)]

    idx = 0
    while stack:
        i = stack.pop()
        if i not in seen:
            order[i]["idx"] = idx
            seen.append(i)
            idx += 1
        for j in np.where(p.edges[i] > 0)[0]:
            if j not in seen:
                stack.append(j)
                order[j]["out_dg"] -= 1
                order[j]["prev"] = idx-1

    return sorted(order, key=lambda x: x["idx"])


def __CBA(p1, p2, start, order, mapping, dist1, dist2, dist_tol, score=None,
          cost=None, map_order=None, idx=None, seen=None):
    """Run constrained backtracking algorithm (CBA).

    Args:
       p1, p2 (Pharmacophore): models to align
       start (int): first node to map
       order (list): partial ordering set for p1
       mapping (numpy array): array describing nodes compatibility
       dist1, dist2 (numpy array): distances between nodes
       dist_tol (float): distance tolerance

       args needed for recursive calls:
          score (float): current similarity score
          cost (float): current edge differences cost
          map_order (list): partial ordering set for p2
          idx (int)- current index
          seen (list): visited nodes

    Returns:
        res (list): all common substructure matches (lists of dictionaries
          complementary to given order)


    see: Xu, Jun. "GMA: a generic match algorithm for structural homomorphism,
         isomorphism, and maximal common substructure match and its
         applications." J. Chem. Inf. Comput. Sci., 1996, 36 (1), 25–34
    """

    def compatible(idx, candidate):
        cand_map = order[idx]["i"]
        if mapping[cand_map][candidate] <= 0:
            return False   # incompatible types

        if order[idx]["out_dg"] > map_order[candidate]["out_dg"]:
            return False    # too few neighbors

        for node in seen:
            node_map = order[map_order[node]["idx"]]["i"]
            d = 0
            if p2.edges[candidate][node] != 0 and \
               p1.edges[cand_map][node_map] != 0:
                d = math.fabs(p2.edges[candidate][node] - p1.edges[cand_map][node_map])
            elif p2.edges[candidate][node] != 0:
                d = math.fabs(p2.edges[candidate][node] - dist1[cand_map][node_map])
            elif p1.edges[cand_map][node_map] != 0:
                d = math.fabs(dist2[candidate][node] - p1.edges[cand_map][node_map])
            if d > dist_tol:
                return False    # difference between lengths above cutoff

        return True     # node is compatible

    if seen is None:
        seen = []
    seen.append(start)

    if idx is None:
        idx = 0

    if score is None:
        score = 0.0
    if cost is None:
        cost = 0.0

    if map_order is None:
        map_order = [{"i": i, "idx": None, "out_dg": np.sum(p2.edges[i] > 0)}
                     for i in xrange(p2.numnodes)]
    map_order[start]["idx"] = idx
    res = []

    #add similarity score for last mapped pair of nodes
    score += mapping[order[idx]["i"]][start]

    #add edge length cost for last mapped pair of nodes
    for node in seen:
        node_map = order[map_order[node]["idx"]]["i"]
        if p2.edges[start][node] != 0 and \
           p1.edges[order[idx]["i"]][node_map] != 0:
            cost += math.fabs(p2.edges[start][node] - p1.edges[order[idx]["i"]][node_map])
        elif p2.edges[start][node] != 0:
            cost += math.fabs(p2.edges[start][node] - dist1[order[idx]["i"]][node_map])
        elif p1.edges[order[idx]["i"]][node_map] != 0:
            cost += math.fabs(dist2[start][node] - p1.edges[order[idx]["i"]][node_map])

    idx += 1

    if idx == len(order):   # all nodes mapped
        res.append((score, cost, map_order))

    else:
        prev_mapped = order[idx]["prev"]
        prev = None
        for j in map_order:
            if j["idx"] == prev_mapped:
                prev = j["i"]
                break

        to_check = []
        for j in np.where(p2.edges[prev] > 0)[0]:
            if j not in seen:
                if compatible(idx, j):
                    to_check.append(j)

        if len(to_check) == 0:
            res.append((score, cost, map_order))

        elif len(to_check) >= 1:
            for j in to_check:
                res += __CBA(p1, p2, j, order[:], mapping, dist1, dist2,
                             dist_tol, score, cost,
                             [m.copy() for m in map_order], idx, seen[:])

    return res


def __extend(p1, p2, matched, to_check, mapping, dist_tol):
    """Extend common substructure for given Pharmacophores.

    Args:
       p1, p2 (Pharmacophore): models to align
       matched (list): tuples representing already matched nodes
       to_check (list): list of lists with nodes indicies to check
       mapping (numpy array): array describing nodes compatibility
       dist_tol (float): distance tolerance

    Returns:
       res (list): all common substructure matches (lists of tuples
         representing matched nodes)

    see: Cao, Yiqun, Tao Jiang, and Thomas Girke. "A maximum common
         substructure-based algorithm for searching and predicting drug-like
         compounds." Bioinformatics. 2008 Jul 1;24(13):i366-74
    """

    def next_node(p, m, to_check):
        connected = np.sum(p.edges[np.array([i[0] for i in m])], axis=0)
        max_possible = np.max(connected[np.array(to_check[0])])

        best_score = 0.0
        best = None
        for i in to_check[0]:
            if connected[i] == max_possible:
                s = np.max(mapping[i, np.array(to_check[1])])
                if s > best_score:
                    best_score = s
                    best = i

        return best

    def compatible(u, v, matched):
        if mapping[u][v] <= 0.0:
            return False
        else:
            neigh1 = set(np.where(p1.edges[u] > 0.0)[0])
            neigh2 = set(np.where(p2.edges[v] > 0.0)[0])
            connected = False
            for i in xrange(len(matched)):
                if matched[i][0] in neigh1:
                    if matched[i][1] in neigh2:
                        if math.fabs(p1.edges[u][matched[i][0]] -
                                     p2.edges[v][matched[i][1]]) <= dist_tol:
                            connected = True
                        else:
                            return False
                    else:
                        return False
                elif matched[i][1] in neigh2:
                    return False
            return connected

    res = []
    while len(to_check[0]) > 0 and len(to_check[1]) > 0:
        u = next_node(p1, matched, to_check)
        if u is not None:
            to_check[0].remove(u)
            for v in to_check[1]:
                if compatible(u, v, matched):
                    tmp = [i for i in to_check[1] if i != v]

                    res += __extend(p1, p2, matched[:]+[(u, v)],
                                    [to_check[0][:], tmp], mapping, dist_tol)
        else:
            res = [matched]
            return res
    if len(res) == 0:
        res = [matched]
    return res


def distances(p):
    """Compute lengths of shortest paths between all nodes in Pharmacophore.

    Args:
       p (Pharmacophore): model to analyse

    Returns:
       dist (numpy array): array with distances between all nodes
    """

    if not isinstance(p, Pharmacophore):
        raise TypeError("Expected Pharmacophore, got %s instead" %
                        type(p).__name__)

    dist = np.array(p.edges)

    for i in xrange(p.numnodes):
        for j in xrange(i):
            if dist[i][j] == 0:
                dist[i][j] = dist[j][i] = float("inf")

    for i in xrange(len(dist)):
        compute = False
        for j in xrange(i):
            if dist[i][j] == float("inf"):
                compute = True
                break
        if compute:
            queue = [k for k in xrange(p.numnodes)]
            while queue:
                queue.sort(key=lambda x: dist[i, x])
                u = queue[0]
                del queue[0]
                for v in np.where(p.edges[u] > 0)[0]:
                    if v in queue:
                        alt = dist[i, u] + p.edges[u, v]
                        if alt < dist[i, v]:
                            dist[i, v] = dist[v, i] = alt
    return dist


def dfs(p, n, to_check=None, visited=None):
    """Perform depth-first search.

    Args:
       p (Pharmacophore): model to search
       n (int): id of first node
       to_check (set, optional): indices of nodes do check
       visited (list, optional): list of indicies of already visited nodes; if
       given it will be updated

    Returns:
       visited (list): all nodes reachable from n
    """
    if not isinstance(p, Pharmacophore):
        raise TypeError("Expected Pharmacophore, got %s instead" %
                        type(p).__name__)

    if visited is None:
        visited = []
    elif not isinstance(visited, list):
        raise TypeError("Expected list, got %s instead" %
                        type(visited).__name__)
    else:
        for i in visited:
            if not isinstance(i, int):
                raise TypeError("Invalid visited list! Node %s is not int!" % i)
            elif i < 0 or i >= p.numnodes:
                raise ValueError("Invalid visited list! Node index out of "
                                 "range: %s" % i)

    if to_check is None:
        to_check = set(range(p.numnodes))
    elif not isinstance(to_check, set):
        raise TypeError("Expected set, got %s instead" % type(to_check).__name__)
    else:
        for i in to_check:
            if not isinstance(i, int):
                raise TypeError("Invalid to_check list! Node %s is not int!" % i)
            elif i < 0 or i >= p.numnodes:
                raise ValueError("Invalid to_check list! Node index out of"
                                 "range: %s" % i)
    if not isinstance(n, int):
        raise TypeError("Starting node is not int!")
    elif n < 0 or n >= p.numnodes:
        raise ValueError("Starting node index out of range!")

    tmp = list(to_check)
    for v in tmp:
        if v not in visited:
            if p.edges[n, v] > 0.0:
                visited.append(v)
                to_check.remove(v)
                dfs(p, v, to_check, visited)
    return visited


def split_components(p, nodes=None):
    """Find all connected components in given Pharmacophore.
    
    Args:
       p (Pharmacophore): model to analyse
       nodes (list, optional): list of nodes indices; if given, find
         components in subgraph induced by those nodes.

    Returns:
       list: nodes indices grouped into connected components, sorted by
         component size
    """
    if not isinstance(p, Pharmacophore):
        raise TypeError("Expected Pharmacophore, got %s instead" %
                        type(p).__name__)

    if nodes is None:
        nodes = range(p.numnodes)
    else:
        if not isinstance(nodes, list):
            raise TypeError("Expected nodes list, got %s instead" %
                            type(nodes).__name__)
        else:
            for n in nodes:
                if not isinstance(n, int):
                    raise TypeError("Node %s is not int!" % n)
                elif n < 0 or n >= p.numnodes:
                    raise ValueError("Node index out of range: %s" % n)

    to_check = set(nodes)
    visited = []
    components = []

    for n in nodes:
        if n in to_check:
            visited = [n]
            to_check.remove(n)
            dfs(p, n, to_check, visited)
            components.append(visited[:])

    return sorted(components, key=len, reverse=True)


def __components(phars, pairs):
    """Find all connected components in common substructure of two Pharmacophores.

    Arguments:
       phars (tuple): two Pharmacophores
       pairs (list): list of pairs of corresponding nodes

    Returns:
       list: nodes indicies grouped into connected components, sorted by
         component size
    """
    p1, p2 = phars
    nodes1 = [i[0] for i in pairs]
    nodes2 = [i[1] for i in pairs]
    components1 = split_components(p1, nodes1)
    components2 = split_components(p2, nodes2)

    def nested_idx(item, l):
        for idx, sublist in enumerate(l):
            if item in sublist:
                return idx

    if len(components1) == 1 and len(components2) == 1:
        return [pairs]
    else:
        partition = {}
        for pair in pairs:
            nr1 = nested_idx(pair[0], components1)
            nr2 = nested_idx(pair[1], components2)
            if (nr1, nr2) not in partition:
                partition[(nr1, nr2)] = [pair]
            else:
                partition[(nr1, nr2)].append(pair)
        return sorted(partition.values(), key=len)


def map_pharmacophores(p1, p2, dist_tol=1.0):
    """Find best common substructure match for two Pharmacophores.

    Args:
       p1, p2 (pharmacophore): models to align
       dist_tol (float, optional): accept distance differences below this threshold

    Returns:
        score (float): unnormalised similarity score
        cost (float): edge length differences cost
        best_subgraph (lists): tuples representing matched nodes
    """
    if not isinstance(p1, Pharmacophore):
        raise TypeError("Expected Pharmacophore, got %s instead" %
                        type(p1).__name__)

    if not isinstance(p2, Pharmacophore):
        raise TypeError("Expected Pharmacophore, got %s instead" %
                        type(p2).__name__)

    if not isinstance(dist_tol, int) and not isinstance(dist_tol, float):
        raise TypeError("dist_tol must be float or int!")

    if dist_tol < 0:
        raise ValueError("dist_tol must be greater than or equal 0")

    #always map smaller pharmacophore to bigger
    if p1.numnodes > p2.numnodes:
        changed = True
        p1, p2 = p2, p1
    else:
        changed = False

    mapping = np.zeros((p1.numnodes, p2.numnodes))

    for i in xrange(p1.numnodes):
        for j in xrange(p2.numnodes):
            weighted_freq, types = compare_nodes(p1.nodes[i], p2.nodes[j])
            if weighted_freq > 0.0:
                mapping[i][j] = weighted_freq

    dist1 = distances(p1)
    dist2 = distances(p2)

    best_subgraphs = [[]]
    score = [0.0]
    cost = [0.0]
    scorecost = [float("-inf")]

    def compute_cost(matched):
        dist_diff = np.zeros((len(matched), len(matched)))

        for i in xrange(len(matched)):
            (u1, v1) = matched[i]
            for j in xrange(i):
                (u2, v2) = matched[j]
                diff = 0.0

                #NOT redundant! edge length can differ from shortest distance!
                if p1.edges[u1, u2] != 0 and p2.edges[v1, v2] != 0:
                    diff = math.fabs(p1.edges[u1, u2] - p2.edges[v1, v2])
                elif p1.edges[u1, u2] != 0:
                    diff = math.fabs(p1.edges[u1, u2] - dist2[v1, v2])
                elif p2.edges[v1, v2] != 0:
                    diff = math.fabs(dist1[u1, u2] - p2.edges[v1, v2])
                dist_diff[i][j] = dist_diff[j][i] = diff

        to_remove = list(set(np.where(dist_diff > dist_tol)[0]))
        cost = np.sum(dist_diff) / 2.0
        return cost, to_remove

    def compute_score(matched):
        idx1 = [i[0] for i in matched]
        idx2 = [i[1] for i in matched]

        if len(np.where(mapping[idx1, idx2] == 0.0)[0]) > 0:
            return 0.0
        else:
            score = np.sum(mapping[idx1, idx2])
            return score

    def update_score(matched, s=None, c=None):
        """remember all subgraphs with highest difference between score and cost"""
        #TODO penalty for breaking rings?
        matched.sort()

        if matched in best_subgraphs:
            return

        if s is None:
            s = compute_score(matched)

        if s < scorecost[0]:
            return    # impossible to get higher score

        if c is None:
            c, to_remove = compute_cost(matched)
            if len(to_remove) > 0:
                for i in to_remove:
                    for comp in __components((p1, p2), matched[:i]+matched[i+1:]):
                        to_check = [range(p1.numnodes), range(p2.numnodes)]
                        for (n1, n2) in comp:
                            to_check[0].remove(n1)
                            to_check[1].remove(n2)
                        tmp = __extend(p1, p2, comp, to_check, mapping, dist_tol)
                        for extended in tmp:
                            update_score(extended)
                return

        sc = s - c
        if sc > scorecost[0]:
            score[0] = s
            cost[0] = c
            best_subgraphs[0] = matched
            scorecost[0] = sc
            del score[1:]
            del cost[1:]
            del best_subgraphs[1:]

        elif sc == scorecost[0]:
            best_subgraphs.append(matched)
            score.append(s)
            cost.append(c)

        return

    completed = False
    for i in xrange(p1.numnodes):
        if completed:
            break
        r = __POS(p1, i)
        for j in np.where(mapping[i] > 0)[0]:
            tmp = __CBA(p1, p2, j, r, mapping, dist1, dist2, dist_tol)

            #highest score and lowest cost first
            tmp.sort(key=lambda x: (-x[0], x[1]))

            best_score = tmp[0][0] - tmp[0][1]
            if best_score < scorecost[0]:
                continue    # impossible to get higher score, check next
            for t in tmp:
                map_score = t[0]
                map_cost = t[1]
                if map_score - map_cost < best_score:
                    break
                map_r = t[2]
                map_r.sort(key=lambda x: x["idx"])
                idx2 = []
                for n in map_r:
                    if n["idx"] is not None:
                        idx2.append(n["i"])
                idx1 = [k["i"] for k in r[:len(idx2)]]
                matched = [(idx1[k], idx2[k]) for k in xrange(len(idx1))]
                update_score(matched, map_score, map_cost)

            map_length = max([len(i) for i in best_subgraphs])
            if map_length == p1.numnodes or map_length == p2.numnodes \
               and score[0] > 0 and cost == 0:
                completed = True
                break

    if len(best_subgraphs[0]) == 0:
        return 0.0, 0.0, [[]]

    for i in best_subgraphs:
        to_check = [range(p1.numnodes), range(p2.numnodes)]
        for pair in i:
            to_check[0].remove(pair[0])
            to_check[1].remove(pair[1])
        tmp = __extend(p1, p2, i, to_check, mapping, dist_tol)
        for t in tmp:
            update_score(t)

    #return subgraph with best similarity score
    best_s = max(score)
    best_c = 0.0
    idx = -1

    for i in xrange(len(score)):
        if score[i] == best_s:
            idx = i
            best_c = cost[i]
            break

    if changed:
        best_inverted = [(j[1], j[0]) for j in best_subgraphs[idx]]
        return best_s, best_c, best_inverted
    else:
        return best_s, best_c, best_subgraphs[idx]


def similarity(p1, p2, dist_tol=1):
    """Find common part of two Pharmacophores, calculate normalized similarity
    score and edge length differences cost.
    
    Args:
       p1, p2 (pharmacophore): models to align
       dist_tol (float, optional): accept distance differences below this threshold

    Returns:
        score (float): normalized similarity score (value between 0 and 1)
        cost (float): edge length differences cost     
    """
    if not isinstance(p1, Pharmacophore):
        raise TypeError("Expected Pharmacophore, got %s instead" %
                        type(p1).__name__)

    if not isinstance(p2, Pharmacophore):
        raise TypeError("Expected Pharmacophore, got %s instead" %
                        type(p2).__name__)

    if not isinstance(dist_tol, int) and not isinstance(dist_tol, float):
        raise TypeError("dist_tol must be float or int!")

    if dist_tol < 0:
        raise ValueError("dist_tol must be greater than or equal 0")

    score, cost, _ = map_pharmacophores(p1, p2, dist_tol)
    a1 = 0.0
    a2 = 0.0
    for n in p1.nodes:
        a1 += n["freq"]
    for n in p2.nodes:
        a2 += n["freq"]

    return (score / (a1 + a2)), cost


def combine_pharmacophores(p1, p2, dist_tol=1.0, freq_cutoff=0.0):
    """Create new model from Pharmacophores p1 and p2

    Find common part of two Pharmacophores, add unique elements and calculate
      new frequencies and distances.

    Args:
      p1, p2 (Pharmacophore): models to combine
      dist_tol (float, optional): accept distance differences below this threshold
      freq_cutoff (float, optional): skip unique nodes with frequencies below
        this threshold

    Returns:
       Pharmacophore: combination of p1 and p2
    """
    if not isinstance(p1, Pharmacophore):
        raise TypeError("Expected Pharmacophore, got %s instead" %
                        type(p1).__name__)

    if not isinstance(p2, Pharmacophore):
        raise TypeError("Expected Pharmacophore, got %s instead" %
                        type(p2).__name__)

    if not isinstance(dist_tol, int) and not isinstance(dist_tol, float):
        raise TypeError("dist_tol must be float or int!")

    if dist_tol < 0:
        raise ValueError("dist_tol must be greater than or equal 0")

    if not isinstance(freq_cutoff, int) and not isinstance(freq_cutoff, float):
        raise TypeError("freq_cutoff must be float or int!")

    if freq_cutoff < 0 or freq_cutoff > 1:
        raise ValueError("Invalid freq_cutoff! Use value in the range [0,1]")

    #find common pharmacophore
    _, _, mapping = map_pharmacophores(p1, p2, dist_tol)

    #we'll need it later
    mapped_nodes = [[], []]
    added = {0: {}, 1: {}}

    #create new graph from common part
    molecules = p1.molecules + p2.molecules
    title = "("+p1.title+")+("+p2.title+")"
    nodes = []

    idx = 0
    for n in mapping:
        _, types = compare_nodes(p1.nodes[n[0]], p2.nodes[n[1]])
        nodes.append({"label": idx, "type": types,
                      "freq": p1.nodes[n[0]]["freq"] + p2.nodes[n[1]]["freq"]})
        for i in [0, 1]:
            added[i][idx] = n[i]
            mapped_nodes[i].append(n[i])
        idx += 1

    #add edges
    edges = np.zeros((idx, idx))
    for i in xrange(idx):
        no1 = (added[0][i], added[1][i])
        for j in xrange(i):
            dist = 0.0
            no2 = (added[0][j], added[1][j])
            freq1 = p1.nodes[no1[0]]["freq"] + p1.nodes[no2[0]]["freq"]
            freq2 = p2.nodes[no1[1]]["freq"] + p2.nodes[no2[1]]["freq"]
            d1 = p1.edges[no1[0], no2[0]]
            d2 = p2.edges[no1[1], no2[1]]
            if not d1:
                dist = d2
            elif not d2:
                dist = d1
            else:
                dist = (d1 * freq1 + d2 * freq2) / (freq1 + freq2)
            edges[i, j] = edges[j, i] = dist

    new_p = Pharmacophore(nodes=nodes, edges=edges, molecules=molecules,
                          title=title)

    #add unique elements
    freq_cutoff = molecules * freq_cutoff

    to_check = [[i for i in xrange(p1.numnodes) if i not in mapped_nodes[0]],
                [i for i in xrange(p2.numnodes) if i not in mapped_nodes[1]]]

    for (nr, phar) in {0: p1, 1: p2}.iteritems():
        for node in mapped_nodes[nr]:
            for n in dfs(phar, node, set(to_check[nr])):
                to_check[nr].remove(n)
                if phar.nodes[n]["freq"] >= freq_cutoff:
                    added[nr][idx] = n
                    new_p.add_node(phar.nodes[n].copy())
                    new_p.nodes[idx]["label"] = idx
                    for (k, v) in added[nr].iteritems():
                        if phar.edges[n, v] > 0:
                            new_p.add_edge(k, idx, phar.edges[n, v])
                    idx += 1
                else:
                    break
    return new_p


def filter_nodes(p, freq_range=(0.0, 1.0), rm_outside=True):
    """Create new model without nodes that does not fulfill given frequency
    criteria.

    Args:
       p (Pharmacophore): model to filter
       freq_range (tuple, optional) - two floats, frequence range for filtering
       rm_outside (bool, optional): if True remove nodes with frequencies
         outside given range; remove nodes with frequencies inside range otherwise

    Returns:
       Pharmacohpre: new model
    """

    if not isinstance(p, Pharmacophore):
        raise TypeError("Expected Pharmacophore, got %s instead" %
                        type(p).__name__)

    if not isinstance(freq_range, tuple):
        raise TypeError("Invalid freq_range!")

    elif not (len(freq_range) == 2 and
       (isinstance(freq_range[0], float) or isinstance(freq_range[0], int)) and
       (isinstance(freq_range[1], float) or isinstance(freq_range[1], int)) and
       freq_range[0] <= freq_range[1]):
        raise ValueError("Invalid freq_range!")
    elif not (freq_range[0] >= 0 and freq_range[1] <= 1):
        raise ValueError("Invalid freq_range! Use values in the range [0,1]")

    if not isinstance(rm_outside, bool):
        raise TypeError("rm_outside should be bool!")

    new_p = p.copy()

    freq_range = [i * p.molecules for i in freq_range]
    if freq_range == [0.0, p.molecules] and rm_outside:
        return new_p

    elif rm_outside:
        for n in xrange(p.numnodes - 1, -1, -1):
            if new_p.nodes[n]["freq"] < freq_range[0] or \
               new_p.nodes[n]["freq"] > freq_range[1]:
                new_p.remove_node(n)
        return new_p
    else:
        for n in xrange(p.numnodes - 1, -1, -1):
            if new_p.nodes[n]["freq"] >= freq_range[0] or \
               new_p.nodes[n]["freq"] <= freq_range[1]:
                new_p.remove_node(n)
        return new_p


def spring_layout(p, c0=0.2, c1=1.0):
    """Calculate points positions for Pharmacophore depiction using spring layout.

    Args:
       p (Pharmacophore): model to depict
       c0, c1 (float, optional): coefficients for spring and repulsive forces

    Returns:
       numpy array: 2D array with nodes positions
    """
    if not isinstance(p, Pharmacophore):
        raise TypeError("Expected Pharmacophore, got %s instead" %
                        type(p).__name__)

    if not ((isinstance(c0, float) or isinstance(c0, int)) and
            (isinstance(c1, float) or isinstance(c1, int))):
        raise TypeError("Invalid constants!")
    if not (c0 > 0 and c1 > 0):
        raise ValueError("Invalid constants! Use values greater than 0.")

    from scipy.optimize import minimize

    def f(x):
        eng = 0.0
        for i in xrange(p.numnodes):
            nx = x[i] - x[:p.numnodes]
            ny = x[p.numnodes + i] - x[p.numnodes:]
            norms = np.sqrt(nx**2 + ny**2)
            norms[norms == 0] = 0.000001
            norms[i] = 0.0
            spring = (norms - p.edges[i])[np.where(p.edges[i] > 0)[0]]**2
            eng += np.sum(spring) * c1

            repulsive = norms[np.where(p.edges[i] == 0)[0]]
            eng += np.sum(c0 / repulsive[np.nonzero(repulsive)])
        return eng

    x0 = np.random.random(p.numnodes*2)

    res = minimize(f, x0)
    newpositions = res.x.reshape((2, p.numnodes)).T
    return newpositions


def draw(p, layout="rd"):
    """Draw Pharmacophore using RDKit ("rd"), OpenBabel ("ob") or spring
    layout ("spring") to calculate nodes positions.

    We recommend to use RDKit ("rd"), as it generates the clearest layouts.
    
    Args:
       p (Pharmacophore): model to depict
       layout (str, optional): layout name

    Returns:
       tuple:
         * matplotlib Figure
         * matplotlib axis
    """
    if not isinstance(p, Pharmacophore):
        raise TypeError("Expected Pharmacophore, got %s instead" %
                        type(p).__name__)

    if not isinstance(layout, str):
        raise TypeError("Invalid layout! Expected str, got %s instead." %
                        type(layout).__name__)
    if layout == "rd":
        try:
            from decaf.toolkits.rd import layout
            pos = layout(p)
        except Exception as e:
            raise ImportError('Cannot use "rd" layout! Use "ob" or "spring" instead', e)

    elif layout == "ob":
        try:
            from decaf.toolkits.ob import layout
            pos = layout(p)
        except Exception as e:
            raise ImportError('Cannot use "ob" layout! Use "rd" or "spring" instead', e)

    elif layout == "spring":
        try:
            pos = spring_layout(p)
        except Exception as e:
            raise ImportError("Cannot use spring layout!", e)
    else:
        raise ValueError('Wrong layout specified! Use "rd", "ob" or "spring"'
                         'instead.')

    ax_coeff = 1.

    def fontsize(idx, default=FontManager.get_default_size()):
        coeff = p.nodes[idx]["freq"] / p.molecules
        size = default * coeff * ax_coeff
        return size

    fig, ax = plt.subplots()
    plt.axis('equal')
    plt.axis('off')

    axis = (np.min(pos[:, 0])-1,
            np.max(pos[:, 0])+1,
            np.min(pos[:, 1])-1,
            np.max(pos[:, 1])+1)
    plt.axis(axis)

    #calculate scaling ratio for font
    ax_coeff = 12. / max((axis[1]-axis[0]), (axis[3]-axis[2]))

    for i in xrange(p.numnodes):
        for j in xrange(i):
            if p.edges[i, j] > 0:
                tmp = np.array([pos[i], pos[j]])
                ax.plot(tmp[:, 0], tmp[:, 1], color='#000000', zorder=1)

        r = p.nodes[i]["freq"] / p.molecules * 0.3
        fsize = fontsize(i)
        nfreq = sum(p.nodes[i]["type"].values())
        theta1 = 0.0
        for t in p.nodes[i]["type"]:
            delta = 360 * p.nodes[i]["type"][t] / nfreq
            theta2 = theta1+delta
            w = Wedge(pos[i], r, theta1, theta2, ec="none", fc=COLORS[t])
            ax.add_artist(w)
            ax.text(pos[i][0], pos[i][1], str(p.nodes[i]["label"]),
                    color='#000000', ha="center", va="center", size=fsize)
            theta1 = theta2

    plt.show()
    return fig, ax
