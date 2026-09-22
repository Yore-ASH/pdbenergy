"""A small, dependency-free reader/writer for the PDB file format.

Why hand-roll a parser instead of using Biopython/ProDy?
-------------------------------------------------------
The PDB format is a *fixed column* format.  Writing our own parser makes every
column explicit, which is exactly the kind of detail that matters when you later
discover that a structure has altlocs, insertion codes, multi-model NMR frames,
or is missing its element column.  It also keeps the project installable with
nothing but NumPy + PyTorch + OpenMM.

References
----------
* PDB format guide (v3.3):
  https://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html
* wwPDB PDBx/mmCIF (the modern replacement format):
  https://www.wwpdb.org/documentation/file-format
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Chemical knowledge tables
# --------------------------------------------------------------------------- #

#: The 20 standard amino acids, three-letter code -> one-letter code.
AA3_TO_1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

#: Protonation / chemical variants and non-standard residues that still map
#: onto a standard residue template.  OpenMM force fields know HID/HIE/HIP
#: (histidine tautomers), CYX (disulfide cysteine), ASH/GLH (protonated acids)
#: and LYN (neutral lysine).
AA3_VARIANTS: dict[str, str] = {
    "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K",
    "MSE": "M",  # selenomethionine
    "SEC": "C",  # selenocysteine
    "PYL": "K",
}

STANDARD_AA: frozenset[str] = frozenset(AA3_TO_1) | frozenset(AA3_VARIANTS)

#: Solvent / cryoprotectant / buffer residue names to discard.
WATER_NAMES: frozenset[str] = frozenset(
    {"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "TIP4", "SOL"}
)

#: Two-letter element symbols that can show up in a PDB.  Used only when the
#: element column is blank, and never for standard amino-acid atom names (there
#: ``CA`` is the alpha carbon, not calcium).
TWO_LETTER_ELEMENTS: frozenset[str] = frozenset(
    {"SE", "FE", "ZN", "MG", "MN", "NA", "CA", "CL", "CU", "NI", "CO", "CD",
     "HG", "BR", "SI", "LI", "AL", "AG", "AU", "PT", "PB", "SR", "BA", "RB",
     "CS", "MO", "SN", "SB", "TE", "TI", "CR", "V", "W", "ZR", "NB", "GD", "YB"}
)

#: Backbone atom names shared by every amino acid.
BACKBONE_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Atom:
    """One ATOM/HETATM record."""

    serial: int
    name: str          # columns 13-16, e.g. " CA "
    altloc: str        # column 17, alternate-location indicator
    resname: str       # columns 18-20
    chain: str         # column 22
    resseq: int        # columns 23-26
    icode: str         # column 27, insertion code
    x: float
    y: float
    z: float
    occupancy: float   # columns 55-60
    bfactor: float     # columns 61-66
    element: str       # columns 77-78
    record: str = "ATOM"   # "ATOM" or "HETATM"

    # -- convenience ------------------------------------------------------- #
    @property
    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def is_hydrogen(self) -> bool:
        return self.element.upper() in ("H", "D")

    @property
    def residue_id(self) -> tuple[str, int, str]:
        """(chain, resseq, icode) - the unique key of a residue within a model."""
        return (self.chain, self.resseq, self.icode)

    def copy(self, **changes) -> "Atom":
        import dataclasses

        return dataclasses.replace(self, **changes)


@dataclass
class Residue:
    """A residue: a group of atoms sharing (chain, resseq, icode)."""

    name: str
    chain: str
    resseq: int
    icode: str
    atoms: list[Atom] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode)

    def atom(self, name: str) -> Atom | None:
        for a in self.atoms:
            if a.name.strip() == name:
                return a
        return None


@dataclass
class Structure:
    """A single structural model (one MODEL block, or the whole file)."""

    atoms: list[Atom] = field(default_factory=list)
    pdb_id: str = ""
    model: int = 0
    title: str = ""

    # -- container protocol ------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.atoms)

    def __iter__(self) -> Iterator[Atom]:
        return iter(self.atoms)

    def __getitem__(self, idx):
        return self.atoms[idx]

    # -- geometry ---------------------------------------------------------- #
    @property
    def coords(self) -> np.ndarray:
        """(N, 3) float64 array of Cartesian coordinates in Angstrom."""
        if not self.atoms:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array([[a.x, a.y, a.z] for a in self.atoms], dtype=np.float64)

    @property
    def elements(self) -> list[str]:
        return [a.element.upper() for a in self.atoms]

    @property
    def names(self) -> list[str]:
        return [a.name.strip() for a in self.atoms]

    def set_coords(self, coords: np.ndarray) -> "Structure":
        """Return a copy with new coordinates (atom order unchanged)."""
        coords = np.asarray(coords, dtype=np.float64)
        if coords.shape != (len(self.atoms), 3):
            raise ValueError(
                f"expected coords shape {(len(self.atoms), 3)}, got {coords.shape}"
            )
        out = Structure([a.copy() for a in self.atoms], self.pdb_id, self.model, self.title)
        for atom, xyz in zip(out.atoms, coords):
            atom.x, atom.y, atom.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        return out

    # -- residue level views ----------------------------------------------- #
    def residues(self) -> list[Residue]:
        """Group atoms into residues, preserving file order."""
        out: list[Residue] = []
        index: dict[tuple[str, int, str], Residue] = {}
        for a in self.atoms:
            key = a.residue_id
            res = index.get(key)
            if res is None:
                res = Residue(a.resname, a.chain, a.resseq, a.icode)
                index[key] = res
                out.append(res)
            res.atoms.append(a)
        return out

    def sequence(self) -> dict[str, str]:
        """One-letter sequence per chain (non-standard residues become ``X``)."""
        seqs: dict[str, list[str]] = {}
        for res in self.residues():
            code = AA3_TO_1.get(res.name, AA3_VARIANTS.get(res.name, "X"))
            seqs.setdefault(res.chain, []).append(code)
        return {chain: "".join(codes) for chain, codes in seqs.items()}

    def n_residues(self) -> int:
        return len({a.residue_id for a in self.atoms})

    # -- filtering --------------------------------------------------------- #
    def subset(self, keep: Sequence[int]) -> "Structure":
        keep = list(keep)
        return Structure([self.atoms[i] for i in keep], self.pdb_id, self.model, self.title)

    def copy(self) -> "Structure":
        return Structure([a.copy() for a in self.atoms], self.pdb_id, self.model, self.title)


# --------------------------------------------------------------------------- #
# Element inference
# --------------------------------------------------------------------------- #


def infer_element(atom_name: str, resname: str = "") -> str:
    """Best-effort element assignment for files with a blank element column.

    PDB atom-name columns are right-justified for one-letter elements and
    left-justified for two-letter ones, which is why ``" CA "`` in an amino acid
    means carbon while ``"CA  "`` in a calcium ion means calcium.
    """
    name = atom_name.strip()
    if not name:
        return ""
    res = resname.strip().upper()
    if res == "MSE" and name.upper().startswith("SE"):
        return "SE"
    # Strip leading digits: "1HB " -> "HB", "HD21" -> "HD".
    stripped = name.lstrip("0123456789")
    if not stripped:
        return ""
    if res not in STANDARD_AA:
        two = stripped[:2].upper()
        if two in TWO_LETTER_ELEMENTS:
            return two
    return stripped[0].upper()


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _f(text: str, start: int, end: int) -> str:
    """Slice columns [start, end) using 1-based column numbers (PDB style)."""
    return text[start - 1:end]


def parse_atom_line(line: str) -> Atom | None:
    """Parse a single ATOM/HETATM record; return ``None`` on malformed input."""
    if len(line) < 54:
        return None
    record = line[:6].strip()
    if record not in ("ATOM", "HETATM"):
        return None
    try:
        serial = int(_f(line, 7, 11))
    except ValueError:
        serial = 0
    name = _f(line, 13, 16)
    altloc = _f(line, 17, 17).strip()
    resname = _f(line, 18, 21).strip()
    chain = _f(line, 22, 22).strip() or "A"
    try:
        resseq = int(_f(line, 23, 27))
    except ValueError:
        return None
    icode = _f(line, 27, 27).strip()
    try:
        x = float(_f(line, 31, 39))
        y = float(_f(line, 39, 47))
        z = float(_f(line, 47, 55))
    except ValueError:
        return None
    try:
        occupancy = float(_f(line, 55, 61))
    except ValueError:
        occupancy = 1.0
    try:
        bfactor = float(_f(line, 61, 67))
    except ValueError:
        bfactor = 0.0
    element = _f(line, 77, 79).strip()
    if not element:
        element = infer_element(name, resname)
    return Atom(
        serial=serial, name=name, altloc=altloc, resname=resname, chain=chain,
        resseq=resseq, icode=icode, x=x, y=y, z=z, occupancy=occupancy,
        bfactor=bfactor, element=element, record=record,
    )


def parse_pdb_string(text: str, keep_models: Sequence[int] | None = None) -> list[Structure]:
    """Parse PDB text into a list of :class:`Structure` (one per MODEL block).

    Parameters
    ----------
    text:
        Full file contents.
    keep_models:
        Optional 1-based model indices to keep.  ``None`` keeps every model.
    """
    models: list[Structure] = []
    current = Structure()
    in_model = False
    model_index = 0
    title_parts: list[str] = []
    header_id = ""

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        record = line[:6].strip()

        if record == "HEADER":
            header_id = _f(line, 63, 66).strip()
        elif record == "TITLE":
            title_parts.append(_f(line, 11, 80).strip())
        elif record == "MODEL":
            in_model = True
            model_index += 1
            try:
                declared = int(line[10:14])
            except ValueError:
                declared = model_index
            current = Structure([], pdb_id=header_id, model=declared)
        elif record == "ENDMDL":
            if len(current):
                models.append(current)
            current = Structure([], pdb_id=header_id, model=model_index)
            in_model = False
        elif record in ("ATOM", "HETATM"):
            atom = parse_atom_line(line)
            if atom is not None:
                current.atoms.append(atom)

    if len(current):
        models.append(current)

    title = " ".join(title_parts)
    if not models:
        models = [Structure([], pdb_id=header_id, model=0)]
    for i, m in enumerate(models):
        m.title = title
        if not m.pdb_id:
            m.pdb_id = header_id
        if m.model == 0:
            m.model = i + 1

    if keep_models is not None:
        wanted = set(keep_models)
        models = [m for m in models if m.model in wanted]
    return models


def read_pdb(path: str | os.PathLike, keep_models: Sequence[int] | None = None) -> list[Structure]:
    """Read a (optionally gzipped) PDB file and return all requested models."""
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt"
    try:
        with opener(path, mode, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except UnicodeDecodeError:  # pragma: no cover - defensive
        with opener(path, "rb") as fh:
            text = fh.read().decode("latin-1")
    return parse_pdb_string(text, keep_models=keep_models)


def read_first_model(path: str | os.PathLike) -> Structure:
    """Convenience wrapper returning just the first model."""
    models = read_pdb(path)
    if not models:
        raise ValueError(f"no atoms found in {path}")
    return models[0]


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def format_atom_line(atom: Atom, serial: int | None = None) -> str:
    """Format one atom as a valid 80-column PDB ATOM/HETATM record."""
    serial = atom.serial if serial is None else serial
    element = (atom.element or infer_element(atom.name, atom.resname)).upper()
    # Right-justify single-character elements starting at column 14, left-justify
    # two-character elements starting at column 13.
    name = atom.name
    if len(element) == 1:
        name = f" {name.strip():<3s}"
    else:
        name = f"{name.strip():<4s}"
    return (
        f"{atom.record:<6s}{serial:>5d} {name}{atom.altloc or ' ':<1s}"
        f"{atom.resname:>3s} {(atom.chain or 'A'):<1s}{atom.resseq:>4d}{(atom.icode or ' '):<1s}"
        f"   {atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}"
        f"{atom.occupancy:>6.2f}{atom.bfactor:>6.2f}          {element:>2s}  "
    )


def _ter_line(serial: int, atom: Atom) -> str:
    """Format the TER record that closes the chain ending at ``atom``."""
    return (
        f"TER   {serial:>5d}      {atom.resname:>3s} "
        f"{(atom.chain or 'A'):<1s}{atom.resseq:>4d}"
    )


def write_pdb(
    structures: Structure | Sequence[Structure],
    path: str | os.PathLike,
    *,
    model_records: bool = True,
    remark: str | None = None,
) -> str:
    """Write one or more models to a PDB file (re-serialising atom numbers)."""
    if isinstance(structures, Structure):
        structures = [structures]
    structures = list(structures)
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)

    lines: list[str] = []
    if remark:
        lines.append(f"REMARK  99 {remark}"[:80])
    multi = len(structures) > 1
    for model_idx, struct in enumerate(structures, start=1):
        if multi and model_records:
            lines.append(f"MODEL     {model_idx:>4d}")
        serial = 1
        last_chain: str | None = None
        last_atom: Atom | None = None
        for atom in struct.atoms:
            # A TER record marks the end of a *chain*, not of a residue.  Writing
            # one per residue would make every residue its own chain and break
            # downstream template matching (N/C-terminal residue variants).
            if last_chain is not None and atom.chain != last_chain:
                lines.append(_ter_line(serial, last_atom))
                serial += 1
            lines.append(format_atom_line(atom, serial))
            serial += 1
            last_chain = atom.chain
            last_atom = atom
        # Every chain, including the last one, is terminated by a TER record.
        if last_atom is not None:
            lines.append(_ter_line(serial, last_atom))
        if multi and model_records:
            lines.append("ENDMDL")
    lines.append("END")
    text = "\n".join(lines) + "\n"
    with open(str(path), "w", encoding="utf-8") as fh:
        fh.write(text)
    return str(path)


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #


def resolve_altlocs(structure: Structure) -> Structure:
    """Keep a single conformer per atom: altloc ``' '`` or the highest occupancy.

    Crystallographers use altlocs to describe disordered side chains.  A force
    field cannot represent two positions for one atom, so one must be chosen.
    """
    best: dict[tuple, Atom] = {}
    order: list[tuple] = []
    for a in structure.atoms:
        key = (a.chain, a.resseq, a.icode, a.name)
        prev = best.get(key)
        if prev is None:
            best[key] = a
            order.append(key)
        else:
            # Blank altloc always wins; otherwise highest occupancy, then 'A'.
            def rank(atom: Atom) -> tuple:
                return (atom.altloc == "", atom.occupancy, atom.altloc in ("", "A"))

            if rank(a) > rank(prev):
                best[key] = a
    return Structure([best[k] for k in order], structure.pdb_id, structure.model, structure.title)


def protein_only(
    structure: Structure,
    *,
    keep_hydrogens: bool = True,
    keep_variants: bool = True,
) -> Structure:
    """Strip waters, ions, ligands and (optionally) hydrogens.

    Only standard amino-acid residues survive, so the result is guaranteed to be
    something an all-atom protein force field has templates for.
    """
    allowed = set(STANDARD_AA) if keep_variants else set(AA3_TO_1)
    kept = [
        a for a in structure.atoms
        if a.resname in allowed and (keep_hydrogens or not a.is_hydrogen)
    ]
    out = Structure(kept, structure.pdb_id, structure.model, structure.title)
    return resolve_altlocs(out)


def drop_hydrogens(structure: Structure) -> Structure:
    """Heavy-atom-only view - the representation the neural network consumes."""
    return Structure(
        [a for a in structure.atoms if not a.is_hydrogen],
        structure.pdb_id, structure.model, structure.title,
    )


def chain_summary(structure: Structure) -> str:
    """Human-readable one-line description, handy for logging."""
    seqs = structure.sequence()
    parts = [f"{c}:{len(s)}res" for c, s in seqs.items()]
    het = {a.resname for a in structure.atoms if a.record == "HETATM"}
    return (
        f"{structure.pdb_id or '?'} model {structure.model} | "
        f"{len(structure)} atoms | {structure.n_residues()} residues | "
        f"chains [{' '.join(parts)}]"
        + (f" | HETATM {sorted(het)}" if het else "")
    )
