# -*- coding: utf-8 -*-
"""
Unit tests for DeCAF module

Created on Thu Apr  2 15:57:49 2015
@author: Marta Stepniewska
"""
import unittest
import numpy as np


class PharmacophoreTests(unittest.TestCase):

    def setUp(self):
        from decaf import Pharmacophore
        nodes = [{"label": 0, "freq": 2.0, "type": {"HH": 2.0, "AR": 2.0, "R": 2.0}},
                 {"label": 1, "freq": 2.0, "type": {"HH": 2.0, "AR": 2.0, "R": 2.0}},
                 {"label": 2, "freq": 2.0, "type": {"HH": 2.0, "AR": 2.0, "R": 2.0}},
                 {"label": 3, "freq": 2.0, "type": {"HH": 2.0, "AR": 2.0, "R": 2.0}},
                 {"label": 4, "freq": 2.0, "type": {"HH": 2.0, "AR": 2.0, "R": 2.0}},
                 {"label": 5, "freq": 2.0, "type": {"AR": 2.0, "R": 2.0}},
                 {"label": 6, "freq": 2.0, "type": {"HA": 2.0}},
                 {"label": 7, "freq": 2.0, "type": {"HH": 2.0}},
                 {"label": 8, "freq": 1.0, "type": {"HA": 1.0, "HD": 1.0}},
                 {"label": 9, "freq": 1.0, "type": {"HA": 1.0, "HD": 1.0}}]

        edges = np.array([[0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
                          [1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
                          [1., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 1., 0., 2., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 2., 0., 1., 1.],
                          [0., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
                          [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.]])

        self.phar = Pharmacophore(nodes, edges, molecules=2,
                                  title="test")

    def tearDown(self):
        self.phar = None

    def testCreate(self):
        self.assertEqual(self.phar.numnodes, len(self.phar.nodes))
        for i in xrange(self.phar.numnodes):
            for j in xrange(i):
                self.assertEqual(self.phar.edges[i, j], self.phar.edges[j, i],
                                 msg=("Array is asymetric! %s!=%s for i=%s, j=%s" %
                                      (self.phar.edges[i, j],
                                       self.phar.edges[j, i], i, j)))

    def testIter(self):
        i = 0
        for node in self.phar:
            self.assertEqual(node, self.phar.nodes[i])
            i += 1

    def testAddNode(self):
        node = {"label": "CH3", "freq": 1.0, "type": {"HH": 2.0}}
        num = self.phar.numnodes + 1.0
        nodes = self.phar.nodes+[node]

        self.phar.add_node(node)

        self.assertEqual(num, self.phar.numnodes)
        self.assertEqual(num, len(self.phar.edges))
        self.assertEqual(num, len(self.phar.edges[0]))
        self.assertEqual(nodes, self.phar.nodes)

    def testDelNode(self):
        from random import randint
        idx = randint(0, self.phar.numnodes - 1)

        num = self.phar.numnodes - 1.0
        nodes = self.phar.nodes[:idx]+self.phar.nodes[idx+1:]

        self.phar.remove_node(idx)

        self.assertEqual(num, self.phar.numnodes)
        self.assertEqual(nodes, self.phar.nodes)

    def testAddEdge(self):
        l = 1.0
        num = np.sum(self.phar.edges > 0) / 2.0 + 1
        for idx1 in xrange(self.phar.numnodes):
            for idx2 in xrange(idx1):
                if self.phar.edges[idx1, idx2] == 0:
                    self.phar.add_edge(idx1, idx2, l)

                    self.assertEqual(num, np.sum(self.phar.edges > 0) / 2.0)
                    self.assertEqual(self.phar.edges[idx1, idx2],
                                     self.phar.edges[idx2, idx1])
                    self.assertEqual(self.phar.edges[idx1, idx2], l)
                    self.setUp()

    def testRemoveEdge(self):
        num = np.sum(self.phar.edges > 0) / 2.0 - 1.0
        for idx1 in xrange(self.phar.numnodes):
            for idx2 in xrange(idx1):
                if self.phar.edges[idx1, idx2] > 0:

                    self.phar.remove_edge(idx1, idx2)

                    self.assertEqual(self.phar.edges[idx1, idx2],
                                     self.phar.edges[idx2, idx1])
                    self.assertEqual(self.phar.edges[idx1, idx2], 0.0)
                    self.assertEqual(num, np.sum(self.phar.edges > 0) / 2.0)
                    self.setUp()

    def testSaveRead(self):
        from decaf import Pharmacophore
        from os import remove
        filename = "test.p"
        self.phar.save(filename)
        p_copy = Pharmacophore.read(filename)

        self.assertEqual(self.phar.numnodes, p_copy.numnodes)
        self.assertEqual(self.phar.nodes, p_copy.nodes)
        for i in xrange(p_copy.numnodes):
            for j in xrange(p_copy.numnodes):
                self.assertEqual(self.phar.edges[i, j], p_copy.edges[i, j])
        self.assertEqual(self.phar.title, p_copy.title)
        self.assertEqual(self.phar.molecules, p_copy.molecules)
        remove(filename)

    def testValidation(self):
        from decaf import Pharmacophore

        self.assertRaises(TypeError, Pharmacophore, "a", self.phar.edges)
        self.assertRaises(TypeError, Pharmacophore, self.phar.nodes, "a")
        self.assertRaises(TypeError, Pharmacophore, self.phar.nodes,
                          self.phar.edges, molecules="a")
        self.assertRaises(ValueError, Pharmacophore, self.phar.nodes,
                          self.phar.edges, molecules=-1)
        self.assertRaises(TypeError, Pharmacophore, self.phar.nodes,
                          self.phar.edges, title=1)

        invalid = [([{"freq": 2.0, "type": {"HH": 2.0, "AR": 2.0}}] +
                    self.phar.nodes[1:], self.phar.edges),
                   ([{"label": 0, "type": {"HH": 2.0, "AR": 2.0}}] +
                    self.phar.nodes[1:], self.phar.edges),
                   ([{"label": 0, "freq": 2.0}]+self.phar.nodes[1:],
                    self.phar.edges),
                   ([{"label": 0, "freq": 2.0, "type": {"H": 2.0, "AR": 2.0}}] +
                    self.phar.nodes[1:], self.phar.edges),
                   (self.phar.nodes, self.phar.edges[:3][:, :3])]

        for args in invalid:
            self.assertRaises(ValueError, Pharmacophore, *args)

        self.assertRaises(TypeError, self.phar.add_node, "1")
        self.assertRaises(ValueError, self.phar.add_node, {})
        self.assertRaises(TypeError, self.phar.remove_node, "1")
        self.assertRaises(ValueError, self.phar.remove_node, -1)
        self.assertRaises(ValueError, self.phar.remove_node, self.phar.numnodes)
        self.assertRaises(TypeError, self.phar.add_edge, "0", 1, 2)
        self.assertRaises(TypeError, self.phar.add_edge, 0, "1", 2)
        self.assertRaises(TypeError, self.phar.add_edge, 0, 1, "2")
        self.assertRaises(ValueError, self.phar.add_edge, 0, 0, 2)
        self.assertRaises(ValueError, self.phar.add_edge, -1, 0, 2)
        self.assertRaises(ValueError, self.phar.add_edge, 0, self.phar.numnodes, 2)
        self.assertRaises(ValueError, self.phar.remove_edge, -1, 0)
        self.assertRaises(TypeError, self.phar.remove_edge, "0", 1)
        self.assertRaises(TypeError, self.phar.remove_edge, 0, "1")
        self.assertRaises(ValueError, self.phar.remove_edge, 0, self.phar.numnodes)


class ToolkitsTests(unittest.TestCase):

    def setUp(self):
        self.string = "Nc1ccc(C(O)O)cc1	mol1"
        self.numnodes = 9
        self.numedges = 10
        self.types = {"AR": 6, "HH": 5, "HA": 3, "HD": 3, "R": 6}

    def testCreateOb(self):
        from pybel import readstring
        import decaf.toolkits.ob as ob
        mol = readstring("smi", self.string)
        phar = ob.phar_from_mol(mol)
        self.assertEqual(phar.numnodes, self.numnodes)
        self.assertEqual(np.sum(phar.edges > 0) / 2.0, self.numedges)

        types = {t: 0 for t in self.types}
        for i in xrange(phar.numnodes):
            for t in phar.nodes[i]["type"].keys():
                types[t] += 1
        self.assertEqual(types, self.types)

    def testValidationOb(self):
        import decaf.toolkits.ob as ob
        self.assertRaises(TypeError, ob.phar_from_mol, "c1ccccc1")
        self.assertRaises(TypeError, ob.phar_from_mol, 2)
        self.assertRaises(TypeError, ob.layout, "c1ccccc1")
        self.assertRaises(TypeError, ob.layout, 2)

    def testCreateRd(self):
        from rdkit.Chem import MolFromSmiles
        import decaf.toolkits.rd as rd
        molstring, name = self.string.split()
        mol = MolFromSmiles(molstring)
        mol.SetProp("_Name", name)
        phar = rd.phar_from_mol(mol)
        self.assertEqual(phar.numnodes, self.numnodes)
        self.assertEqual(np.sum(phar.edges > 0) / 2.0, self.numedges)

        types = {t: 0 for t in self.types}
        for i in xrange(phar.numnodes):
            for t in phar.nodes[i]["type"].keys():
                types[t] += 1
        self.assertEqual(types, self.types)

    def testValidationRd(self):
        import decaf.toolkits.rd as rd
        self.assertRaises(TypeError, rd.phar_from_mol, "c1ccccc1")
        self.assertRaises(TypeError, rd.phar_from_mol, 2)
        self.assertRaises(TypeError, rd.layout, "c1ccccc1")
        self.assertRaises(TypeError, rd.layout, 2)


class UtilsTests(unittest.TestCase):

    def setUp(self):
        from pybel import readstring
        import decaf.toolkits.ob as ob

        self.smiles = ["Cc1c(N)cccc1C(=O)N2CCCC2",
                       "Cc1c(NCCCCC(=O)O)cccc1C(=O)N2CCCC2",
                       "Cc1c(N)ccc(F)c1C(=O)N2CCCC2",
                       "NCCCCC(=O)O"]
        self.phars = [ob.phar_from_mol(readstring("smi", s)) for s in
                      self.smiles]

    def tearDown(self):
        self.phars = None

    def testCompareNodes(self):
        from decaf.utils import compare_nodes

        max_sim = self.phars[0].molecules * 2.0
        for n1 in xrange(self.phars[0].numnodes):
            for n2 in xrange(self.phars[0].numnodes):
                s, t = compare_nodes(self.phars[0].nodes[n1],
                                     self.phars[0].nodes[n2])
                if n1 == n2:
                    self.assertEqual(s, max_sim)
                else:
                    min_length = max(len(self.phars[0].nodes[n1]["type"]),
                                    (len(self.phars[0].nodes[n2]["type"])))

                    self.assertGreaterEqual(len(t), min_length)
                    if len(t) == len(self.phars[0].nodes[n1]["type"]) == \
                       len(self.phars[0].nodes[n2]["type"]):
                        self.assertEqual(s, max_sim)

    def testDistances(self):
        from decaf.utils import distances

        for p in self.phars:
            dist = distances(p)
            edges_id = np.where(p.edges > 0)
            self.assertTrue(((dist - p.edges)[edges_id] <= 0).all())
            self.assertTrue((dist.diagonal() == 0).all())
            dist[range(p.numnodes), range(p.numnodes)] = 1
            self.assertFalse((dist <= 0).any())

    def testDfs(self):
        from decaf.utils import dfs

        for p in self.phars:
            for i in xrange(p.numnodes):
                visited = dfs(p, i)
                self.assertEqual(len(visited), p.numnodes)

    def testSplitComponents(self):
        from decaf.utils import split_components

        for p in self.phars:
            for i in xrange(p.numnodes):
                comps = split_components(p, nodes=range(i)+range(i+1, p.numnodes))
                self.assertLessEqual(len(comps), 2)

    def testMap(self):
        from decaf.utils import map_pharmacophores

        scores = [[0]*len(self.phars) for i in xrange(len(self.phars))]
        costs = [[0]*len(self.phars) for i in xrange(len(self.phars))]
        best_mapped = [[0]*len(self.phars) for i in xrange(len(self.phars))]
        for i in xrange(len(self.phars)):
            for j in xrange(len(self.phars)):
                s, c, m = map_pharmacophores(self.phars[i], self.phars[j],
                                             coarse_grained=False)
                scores[i][j] = s
                costs[i][j] = c
                self.assertEqual(len(m[0]), len(m[1]))
                best_mapped[i][j] = len(m[0])

        for i in xrange(len(self.phars)):
            self.assertEqual(best_mapped[i][i], self.phars[i].numnodes)
            self.assertEqual(scores[i][i], 2.*best_mapped[i][i])
            self.assertEqual(costs[i][i], 0)
            for j in xrange(i):
                self.assertEqual(scores[i][j], scores[j][i])
                self.assertEqual(costs[i][j], costs[j][i])
                self.assertGreaterEqual(scores[i][j], 0)
                self.assertLessEqual(scores[i][j],
                                     2.*min(self.phars[i].numnodes,
                                            self.phars[j].numnodes))

    def testSame(self):
        from rdkit.Chem import MolFromSmiles
        import decaf.toolkits.rd as rd
        from decaf.utils import similarity

        phars2 = [rd.phar_from_mol(MolFromSmiles(s)) for s in
                  self.smiles]
        for i in xrange(len(phars2)):
            score, cost = similarity(self.phars[i], phars2[i])
            self.assertEqual(score, 1.0)
            self.assertEqual(cost, 0.0)

    def testCombine(self):
        from decaf.utils import map_pharmacophores, combine_pharmacophores

        expected = [[0]*len(self.phars) for i in xrange(len(self.phars))]
        real = [[0]*len(self.phars) for i in xrange(len(self.phars))]

        for i in xrange(len(self.phars)):
            for j in xrange(len(self.phars)):
                _, _, m = map_pharmacophores(self.phars[i], self.phars[j],
                                             coarse_grained=False)
                expected[i][j] = self.phars[i].numnodes+self.phars[j].numnodes-len(m[0])
                tmp = combine_pharmacophores(self.phars[i], self.phars[j])
                real[i][j] = tmp.numnodes

        for i in xrange(len(self.phars)):
            self.assertEqual(real[i][i], self.phars[i].numnodes)
            for j in xrange(i):
                self.assertEqual(real[i][j], expected[j][i])
                self.assertEqual(real[i][j], real[j][i])

    def testInclusiveSimilarity(self):
        from decaf.utils import inclusive_similarity

        for i in xrange(2):
            s1, s2, _ = inclusive_similarity(self.phars[0], self.phars[i])
            self.assertEqual(s1, 1.0)

    def testModel(self):
        from decaf.utils import combine_pharmacophores as cp

        cutoff = 0.5
        model0 = cp(self.phars[0], self.phars[0])
        model1 = cp(model0, self.phars[1], freq_cutoff=cutoff)
        self.assertEqual(self.phars[0].numnodes, model0.numnodes)
        freq = self.phars[0].molecules * 2.0
        for node in model0.nodes:
            self.assertEqual(node["freq"], freq)

        freq += self.phars[1].molecules
        for node in model1.nodes:
            self.assertGreaterEqual(node["freq"], freq*cutoff)

    def testFilter(self):
        from decaf.utils import combine_pharmacophores as cp, filter_nodes

        cutoff = 0.5
        model0 = cp(self.phars[0], self.phars[0])
        model1 = cp(model0, self.phars[1])
        model2 = filter_nodes(model1, freq_range=(cutoff, 1.))

        freq = model1.molecules
        for node in model2.nodes:
            self.assertGreaterEqual(node["freq"], freq*cutoff)

    def testValidation(self):
        from decaf.utils import compare_nodes, distances, dfs, filter_nodes, \
            map_pharmacophores as mp, similarity, split_components, \
            combine_pharmacophores as cp

        node = self.phars[0].nodes[0]
        self.assertRaises(TypeError, compare_nodes, node, 0)
        self.assertRaises(TypeError, compare_nodes, 0, node)
        self.assertRaises(ValueError, compare_nodes, node, {})
        self.assertRaises(ValueError, compare_nodes, {}, node)

        self.assertRaises(TypeError, distances, 0)

        p = self.phars[0]
        self.assertRaises(TypeError, dfs, 0, 0)
        self.assertRaises(TypeError, dfs, p, 0, visited=0)
        self.assertRaises(TypeError, dfs, p, 0, to_check=0)
        self.assertRaises(TypeError, dfs, p, 0, to_check=[1, 2, 3])
        self.assertRaises(ValueError, dfs, p, 0, visited=[p.numnodes])
        self.assertRaises(ValueError, dfs, p, 0, visited=[-1])
        self.assertRaises(ValueError, dfs, p, 0, to_check=set([p.numnodes]))
        self.assertRaises(ValueError, dfs, p, 0, to_check=set([-1]))
        self.assertRaises(TypeError, dfs, p, "1")
        self.assertRaises(ValueError, dfs, p, -1)
        self.assertRaises(ValueError, dfs, p, p.numnodes)

        self.assertRaises(TypeError, split_components, 0)
        self.assertRaises(TypeError, split_components, p, nodes=0)
        self.assertRaises(ValueError, split_components, p, nodes=[-1])
        self.assertRaises(ValueError, split_components, p, nodes=[p.numnodes])

        for f in [mp, similarity, cp]:
            self.assertRaises(TypeError, f, p, 0)
            self.assertRaises(TypeError, f, 0, p)
            self.assertRaises(TypeError, f, p, p, dist_tol="1")
            self.assertRaises(ValueError, f, p, p, dist_tol=-1)
        self.assertRaises(TypeError, cp, p, p, freq_cutoff="1")
        self.assertRaises(ValueError, cp, p, p, freq_cutoff=-1)
        self.assertRaises(ValueError, cp, p, p, freq_cutoff=2)

        self.assertRaises(TypeError, filter_nodes, 0)
        self.assertRaises(TypeError, filter_nodes, p, freq_range="0, 1")
        self.assertRaises(ValueError, filter_nodes, p, freq_range=(0.5, 0.25))
        self.assertRaises(ValueError, filter_nodes, p, freq_range=(-1, 0))
        self.assertRaises(ValueError, filter_nodes, p, freq_range=(0, 2))

if __name__ == '__main__':
    unittest.main(verbosity=2)
