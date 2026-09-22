"""Unit tests for the PDB parser/writer.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbenergy.pdbio import (  # noqa: E402
    AA3_TO_1,
    Atom,
    Structure,
    chain_summary,
    drop_hydrogens,
    format_atom_line,
    infer_element,
    parse_atom_line,
    parse_pdb_string,
    protein_only,
    read_pdb,
    resolve_altlocs,
    write_pdb,
)

CRN = os.path.join("data", "raw", "1CRN.pdb")
L2Y = os.path.join("data", "raw", "1L2Y.pdb")

ATOM_LINE = (
    "ATOM      1  N   THR A   1      17.047  14.099   3.625  1.00 13.79           N  "
)


class TestElementInference(unittest.TestCase):
    def test_alpha_carbon_is_carbon_not_calcium(self):
        # " CA " in an amino acid is the alpha carbon...
        self.assertEqual(infer_element(" CA ", "ALA"), "C")
        # ...but "CA  " in a calcium ion is calcium.
        self.assertEqual(infer_element("CA  ", "CA"), "CA")

    def test_leading_digits_stripped(self):
        self.assertEqual(infer_element("1HB ", "ALA"), "H")
        self.assertEqual(infer_element("HD21", "ASN"), "H")
        self.assertEqual(infer_element(" OXT", "ALA"), "O")

    def test_selenium_in_mse(self):
        self.assertEqual(infer_element("SE  ", "MSE"), "SE")

    def test_unknown_name(self):
        self.assertEqual(infer_element("", "ALA"), "")


class TestAtomLineParsing(unittest.TestCase):
    def test_fields(self):
        atom = parse_atom_line(ATOM_LINE)
        self.assertIsNotNone(atom)
        assert atom is not None
        self.assertEqual(atom.serial, 1)
        self.assertEqual(atom.name, " N  ")
        self.assertEqual(atom.resname, "THR")
        self.assertEqual(atom.chain, "A")
        self.assertEqual(atom.resseq, 1)
        self.assertEqual(atom.element, "N")
        self.assertAlmostEqual(atom.x, 17.047, places=3)
        self.assertAlmostEqual(atom.z, 3.625, places=3)
        self.assertAlmostEqual(atom.occupancy, 1.00, places=2)
        self.assertAlmostEqual(atom.bfactor, 13.79, places=2)

    def test_blank_element_column_is_inferred(self):
        line = ATOM_LINE[:76] + "  " + ATOM_LINE[78:]
        atom = parse_atom_line(line)
        assert atom is not None
        self.assertEqual(atom.element, "N")

    def test_non_atom_line_returns_none(self):
        self.assertIsNone(parse_atom_line("HEADER    PLANT PROTEIN"))
        self.assertIsNone(parse_atom_line("ATOM  short"))


class TestRoundTrip(unittest.TestCase):
    def test_columns_are_stable(self):
        atom = parse_atom_line(ATOM_LINE)
        assert atom is not None
        line = format_atom_line(atom, 1)
        self.assertEqual(len(line) >= 78, True)
        again = parse_atom_line(line)
        assert again is not None
        self.assertEqual(again.name.strip(), atom.name.strip())
        self.assertEqual(again.resname, atom.resname)
        self.assertEqual(again.chain, atom.chain)
        self.assertAlmostEqual(again.x, atom.x, places=3)
        self.assertEqual(again.element, atom.element)

    @unittest.skipUnless(os.path.exists(CRN), "1CRN.pdb not downloaded")
    def test_write_read_round_trip(self):
        structure = read_pdb(CRN)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rt.pdb")
            write_pdb(structure, path)
            back = read_pdb(path)[0]
        self.assertEqual(len(back), len(structure))
        np.testing.assert_allclose(back.coords, structure.coords, atol=1e-3)
        self.assertEqual(back.sequence(), structure.sequence())

    @unittest.skipUnless(os.path.exists(CRN), "1CRN.pdb not downloaded")
    def test_ter_written_once_per_chain_not_per_residue(self):
        """Regression test: a TER between every residue splits each residue into
        its own chain and breaks OpenMM template matching."""
        structure = read_pdb(CRN)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ter.pdb")
            write_pdb(structure, path)
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        n_ter = sum(1 for line in text.splitlines() if line.startswith("TER"))
        n_chains = len({a.chain for a in structure.atoms})
        self.assertEqual(n_ter, n_chains)


class TestStructureViews(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(CRN), "1CRN.pdb not downloaded")
    def test_crambin_facts(self):
        structure = read_pdb(CRN)[0]
        self.assertEqual(len(structure), 327)
        self.assertEqual(structure.n_residues(), 46)
        self.assertEqual(structure.sequence()["A"], "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN")
        self.assertEqual(set(structure.elements), {"C", "N", "O", "S"})

    @unittest.skipUnless(os.path.exists(L2Y), "1L2Y.pdb not downloaded")
    def test_nmr_models(self):
        models = read_pdb(L2Y)
        self.assertEqual(len(models), 38)
        self.assertEqual([m.model for m in models[:3]], [1, 2, 3])
        # Every model of the same entry must have the same atom count.
        self.assertEqual(len({len(m) for m in models}), 1)

    def test_drop_hydrogens(self):
        structure = Structure([
            Atom(1, " CA ", "", "ALA", "A", 1, "", 0.0, 0.0, 0.0, 1.0, 0.0, "C"),
            Atom(2, " HA ", "", "ALA", "A", 1, "", 1.0, 0.0, 0.0, 1.0, 0.0, "H"),
        ])
        heavy = drop_hydrogens(structure)
        self.assertEqual(len(heavy), 1)
        self.assertEqual(heavy.elements, ["C"])

    def test_resolve_altlocs_prefers_blank_then_occupancy(self):
        atoms = [
            Atom(1, " CB ", "A", "SER", "A", 1, "", 0.0, 0.0, 0.0, 0.5, 0.0, "C"),
            Atom(2, " CB ", "B", "SER", "A", 1, "", 1.0, 0.0, 0.0, 0.9, 0.0, "C"),
        ]
        resolved = resolve_altlocs(Structure(atoms))
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved.atoms[0].altloc, "B")   # higher occupancy wins

        atoms[1] = atoms[1].copy(altloc="", occupancy=0.4)
        resolved = resolve_altlocs(Structure(atoms))
        self.assertEqual(resolved.atoms[0].altloc, "")    # blank always wins

    def test_protein_only_strips_water_and_ligands(self):
        atoms = [
            Atom(1, " CA ", "", "ALA", "A", 1, "", 0.0, 0.0, 0.0, 1.0, 0.0, "C"),
            Atom(2, " O  ", "", "HOH", "A", 2, "", 1.0, 0.0, 0.0, 1.0, 0.0, "O",
                 record="HETATM"),
            Atom(3, " ZN ", "", " ZN", "A", 3, "", 2.0, 0.0, 0.0, 1.0, 0.0, "ZN",
                 record="HETATM"),
        ]
        clean = protein_only(Structure(atoms))
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean.atoms[0].resname, "ALA")

    def test_set_coords_validates_shape(self):
        structure = Structure([
            Atom(1, " CA ", "", "ALA", "A", 1, "", 0.0, 0.0, 0.0, 1.0, 0.0, "C"),
        ])
        moved = structure.set_coords(np.array([[1.0, 2.0, 3.0]]))
        np.testing.assert_allclose(moved.coords, [[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            structure.set_coords(np.zeros((3, 3)))

    def test_chain_summary_smoke(self):
        structure = Structure([
            Atom(1, " CA ", "", "ALA", "A", 1, "", 0.0, 0.0, 0.0, 1.0, 0.0, "C"),
        ], pdb_id="TEST")
        self.assertIn("TEST", chain_summary(structure))


class TestSequenceMapping(unittest.TestCase):
    def test_all_20_standard_residues_mapped(self):
        self.assertEqual(len(AA3_TO_1), 20)
        self.assertEqual(sorted(AA3_TO_1.values()), sorted("ACDEFGHIKLMNPQRSTVWY"))


class TestEmptyAndMalformedInput(unittest.TestCase):
    def test_empty_file_gives_empty_structure(self):
        models = parse_pdb_string("HEADER nothing here\nEND\n")
        self.assertEqual(len(models), 1)
        self.assertEqual(len(models[0]), 0)
        self.assertEqual(models[0].coords.shape, (0, 3))

    def test_brute_force_no_crash_on_garbage(self):
        text = "\n".join(["garbage", "ATOM", "MODEL", "ENDMDL", "ATOM  x"])
        models = parse_pdb_string(text)
        self.assertIsInstance(models, list)


if __name__ == "__main__":
    unittest.main()
