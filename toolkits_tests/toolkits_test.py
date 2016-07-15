# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 09:07:48 2016

@author: martas
"""
import unittest
import numpy as np


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
        for i in range(phar.numnodes):
            for t in list(phar.nodes[i]["type"].keys()):
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
        for i in range(phar.numnodes):
            for t in list(phar.nodes[i]["type"].keys()):
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
        from rdkit.Chem import MolFromSmiles
        import decaf.toolkits.rd as rd

        self.smiles = ["Cc1c(N)cccc1C(=O)N2CCCC2",
                       "Cc1c(NCCCCC(=O)O)cccc1C(=O)N2CCCC2",
                       "Cc1c(N)ccc(F)c1C(=O)N2CCCC2",
                       "NCCCCC(=O)O",
                       "O=C(O)CC2CCCC(CC1CCCCC1)C2C(=O)O",
                       "O=C(O)CC2CCCC(CCC1CCCCC1)C2C(=O)O"]
        self.phars = [rd.phar_from_mol(MolFromSmiles(s)) for s in
                      self.smiles]

    def tearDown(self):
        self.phars = None

    def testCompareNodes(self):
        from decaf.utils import compare_nodes

        max_sim = self.phars[0].molecules * 2.0
        for n1 in range(self.phars[0].numnodes):
            for n2 in range(self.phars[0].numnodes):
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
            dist[list(range(p.numnodes)), list(range(p.numnodes))] = 1
            self.assertFalse((dist <= 0).any())

    def testDfs(self):
        from decaf.utils import dfs

        for p in self.phars:
            for i in range(p.numnodes):
                visited = dfs(p, i)
                self.assertEqual(len(visited), p.numnodes)

    def testSplitComponents(self):
        from decaf.utils import split_components

        for p in self.phars:
            for i in range(p.numnodes):
                comps = split_components(p, nodes=list(range(i))+list(range(i+1, p.numnodes)))
                self.assertLessEqual(len(comps), 2)

    def testMap(self):
        from decaf.utils import map_pharmacophores

        scores = [[0]*len(self.phars) for i in range(len(self.phars))]
        costs = [[0]*len(self.phars) for i in range(len(self.phars))]
        best_mapped = [[0]*len(self.phars) for i in range(len(self.phars))]

        for d in [0.0, 1.0]:
            for i in range(len(self.phars)):
                for j in range(len(self.phars)):

                    s, c, m = map_pharmacophores(self.phars[i], self.phars[j],
                                                 dist_tol=d,
                                                 coarse_grained=False)
                    scores[i][j] = s
                    costs[i][j] = c
                    self.assertEqual(len(m[0]), len(m[1]))
                    best_mapped[i][j] = len(m[0])

                    s2, c2, m2 = map_pharmacophores(self.phars[i],
                                                    self.phars[j],
                                                    dist_tol=d,
                                                    coarse_grained=False,
                                                    add_neighbours=True)

                    self.assertGreaterEqual(s2, s)
                    self.assertGreaterEqual(c2, c)
                    self.assertEqual(len(m2[0]), len(m2[1]))
                    self.assertGreaterEqual(len(m2[0]), len(m[0]))

            for i in range(len(self.phars)):
                self.assertEqual(best_mapped[i][i], self.phars[i].numnodes)
                self.assertEqual(scores[i][i], 2.*best_mapped[i][i])
                self.assertEqual(costs[i][i], 0)
                for j in range(i):
                    self.assertAlmostEqual(scores[i][j], scores[j][i])
                    self.assertAlmostEqual(costs[i][j], costs[j][i])
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
        for i in range(len(phars2)):
            score, cost = similarity(self.phars[i], phars2[i])
            self.assertEqual(score, 1.0)
            self.assertEqual(cost, 0.0)

    def testCombine(self):
        from decaf.utils import map_pharmacophores, combine_pharmacophores

        expected = [[0]*len(self.phars) for i in range(len(self.phars))]
        real = [[0]*len(self.phars) for i in range(len(self.phars))]

        for i in range(len(self.phars)):
            for j in range(len(self.phars)):
                _, _, m = map_pharmacophores(self.phars[i], self.phars[j],
                                             coarse_grained=False)
                expected[i][j] = self.phars[i].numnodes+self.phars[j].numnodes-len(m[0])
                tmp = combine_pharmacophores(self.phars[i], self.phars[j])
                real[i][j] = tmp.numnodes

        for i in range(len(self.phars)):
            self.assertEqual(real[i][i], self.phars[i].numnodes)
            for j in range(i):
                self.assertAlmostEqual(real[i][j], expected[j][i])
                self.assertAlmostEqual(real[i][j], real[j][i])

    def testInclusiveSimilarity(self):
        from decaf.utils import inclusive_similarity

        for i in range(2):
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

        for f in [mp, similarity]:
            self.assertRaises(TypeError, f, p, p, coarse_grained="1")
            self.assertRaises(TypeError, f, p, p, coarse_grained=False,
                              add_neighbours="1")

        self.assertRaises(TypeError, cp, p, p, freq_cutoff="1")
        self.assertRaises(ValueError, cp, p, p, freq_cutoff=-1)
        self.assertRaises(ValueError, cp, p, p, freq_cutoff=2)
        self.assertRaises(TypeError, cp, p, p, add_neighbours="1")

        self.assertRaises(TypeError, filter_nodes, 0)
        self.assertRaises(TypeError, filter_nodes, p, freq_range="0, 1")
        self.assertRaises(ValueError, filter_nodes, p, freq_range=(0.5, 0.25))
        self.assertRaises(ValueError, filter_nodes, p, freq_range=(-1, 0))
        self.assertRaises(ValueError, filter_nodes, p, freq_range=(0, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
