import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial import HalfspaceIntersection
from functools import cmp_to_key

PREFAC = 0.0591
check = True

class PourbaixEntry():
    def __init__(self, npH=0, nH2O=0, nPhi=0, energy=0.0, concentration=1.0, name=None):
        self.concentration = concentration
        self.phase_type = "Solid"
        self.charge = 0.0
        self.npH  = npH
        self.nH2O = nH2O
        self.nPhi = nPhi
        self.energy = energy
        self.name = name

    @property
    def conc_term(self):
        """
        Return the concentration contribution to the Gibbs free energy,
        and should only be present for ions.
        """
        return PREFAC*np.log10(self.concentration)

    @property
    def corrected_energy(self):
        return self.energy + self.conc_term

class AITDEntry():
    def __init__(self, nA=0, muA=0.0, nB=0, muB=0.0, energy=0.0, name=None):
        self.phase_type = "Solid"
        self.nA  = nA
        self.muA = muA
        self.nB  = nB
        self.muB = muB
        self.energy = energy
        self.name = name

class PhaseDiagram():
    def __init__(self, entries, limits=None, diagram_type="Pourbaix"):
        entries = deepcopy(entries)
        self.limits = limits
        self.diagram_type = diagram_type

        # Process single entry inputs
        solid_entries = [entry for entry in entries if entry.phase_type == "Solid"]
        ion_entries   = [entry for entry in entries if entry.phase_type == "Ion"]

        self._processed_entries = solid_entries + ion_entries

        self._stable_domains, self._stable_domain_vertices = self.get_domains(self._processed_entries, 
                                                                              limits=self.limits, diagram_type=self.diagram_type)

    @staticmethod
    def get_domains(entries, limits=None, diagram_type=None):
        if limits is None:
            limits = [[0, 14], [-4, 4]]

        if diagram_type == "Pourbaix":
            # 0 < energy + pH + nPhi --> -energy - pH - nPhi < 0
            hyperplanes = [np.array([-PREFAC*entry.npH, -entry.nPhi, 0, -entry.corrected_energy]) for entry in entries]
        elif diagram_type == "AITD":
            # surface energy : 0 < G(slab+ads) - sum{nX*muX} --> -energy + sum{nX*muX}
            hyperplanes = [np.array([entry.nA*entry.muA, entry.nB*entry.muB, 0, -entry.energy]) for entry in entries]

        hyperplanes = np.array(hyperplanes)
        hyperplanes[:, 2] = 1

        max_contribs = np.max(np.abs(hyperplanes), axis=0)
        g_max = np.dot(-max_contribs, [limits[0][1], limits[1][1], 0, 1])

        border_hyperplanes = [
            [-1,  0,  0,  limits[0][0]],
            [ 1,  0,  0, -limits[0][1]],
            [ 0, -1,  0,  limits[1][0]],
            [ 0,  1,  0, -limits[1][1]],
            [ 0,  0, -1, 2*g_max]
        ]
        hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
        interior_point = np.average(limits, axis=1).tolist() + [g_max]   # midpoint of xmin and xmax
        hs_int = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))

        # organize the boundary points by entry
        phase_domains = {entry: [] for entry in entries}
        for intersection, facet in zip(hs_int.intersections, hs_int.dual_facets):
            for v in facet:
                if v < len(entries):
                    this_entry = entries[v]
                    phase_domains[this_entry].append(intersection)

        # remove entries with no region
        phase_domains = {k: v for k, v in phase_domains.items() if v}
        phase_domain_vertices = {}
        for entry, points in phase_domains.items():
            points = np.array(points)[:, :2]

            # sort points ... necessary
            points = points[np.lexsort(np.transpose(points))]
            center = np.average(points, axis=0)
            points_centered = points - center
            points_centered = sorted(points_centered, key=cmp_to_key(lambda x, y: x[0]*y[1] - x[1]*y[0]))
            points = points_centered + center

            phase_domain_vertices[entry] = points

        return phase_domains, phase_domain_vertices

class PhaseDiagramPlotter():
    def __init__(self, phase_diagram, diagram_type="Pourbaix"):
        self._pbx = phase_diagram
        self._diagram_type = diagram_type

    def show(self, *args, **kwards):
        plt = self.get_diagram(*args, **kwards)
        plt.show()

    def get_diagram(self, title=None, limits=None):
        if limits is None:
            limits = [[0, 14], [-3, 3]]

        xlim = limits[0]
        ylim = limits[1]

        ax = plt.gca()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        lw = 1

        for entry, vertices in self._pbx._stable_domain_vertices.items():
            center = np.average(vertices, axis=0)
            x, y = np.transpose(np.vstack([vertices, vertices[0]]))
            plt.plot(x, y, "k-", linewidth=lw)

            plt.annotate(generate_entry_label(entry), center, ha="center", va="center", fontsize=10, color="b").draggable()

        if self._diagram_type == "Pourbaix":
            plt.xlabel("pH")
            plt.ylabel("E (V)")
        elif self._diagram_type == "AITD":
            plt.xlabel("delta mu_A")
            plt.ylabel("delta mu_B")

        plt.title(title, fontsize=10, fontweight="bold")
        return plt

def generate_entry_label(entry):
    string = entry.name
    return string


from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.build import fcc111, add_adsorbate
from ase.visualize import view
from ase.constraints import FixAtoms

def opt_and_get_Gibbs_energy(Atoms):
    opt = BFGS(Atoms)
    opt.run(fmax=0.05, steps=100)
    energy = Atoms.get_potential_energy()
    _ZPE = 0.0
    _TS  = 0.0
    Gibbs = energy + _ZPE - _TS
    return Gibbs

def adsorb_and_fix(Atoms, adsorbate=Atoms("O"), offset=[0.66, 0.66], num_fix=2):
    newAtoms = deepcopy(Atoms)
    add_adsorbate(newAtoms, adsorbate, offset=offset, height=1.6)
    c = FixAtoms(indices=[atom.index for atom in newAtoms if atom.tag > num_fix])
    newAtoms.set_constraint(c)
    return newAtoms

# parameters
ZPE = 0.0
T = 300.0  # temperature [K]
entropy = {"O2": 3.0e-3, "HO": 5.0e-3, "CO": 1.0e-3}

ag = fcc111(symbol="Ag", size=[2, 2, 4], a=4.0, vacuum=10.0)
c = FixAtoms(indices=[atom.index for atom in ag if atom.tag > 2])
ag.set_constraint(c)
ag.calc = EMT()

G = {}

# adsorbates
o2 = Atoms("O2", [[0,0,0],[0,0,1.2]])
oh = Atoms("OH", [[0,0,0],[0,0,1.2]])
co = Atoms("CO", [[0,0,0],[0,0,1.2]])
for ads in [o2, oh, co]:
    ads.calc = EMT()
    energy = opt_and_get_Gibbs_energy(ads)
    name = ads.get_chemical_formula()

    if name == "O2":
        nu = 0.5
    else:
        nu = 1.0

    TS = nu*T*entropy[name]
    Gibbs = energy + ZPE - TS
    G.update({name: Gibbs})

# Ag
energy = opt_and_get_Gibbs_energy(ag)
G.update({"Ag": energy})
if check:
    view(ag)

# O
ag_o1 = adsorb_and_fix(ag, adsorbate="O")
ag_o1.calc = EMT()
energy = opt_and_get_Gibbs_energy(ag_o1)
G.update({"AgO": energy})
if check:
    view(ag_o1)

# HO
ag_oh1 = adsorb_and_fix(ag, adsorbate=oh)
ag_oh1.calc = EMT()
energy = opt_and_get_Gibbs_energy(ag_oh1)
G.update({"AgOH": energy})
if check:
    view(ag_oh1)

# CO
ag_co1 = adsorb_and_fix(ag, adsorbate=co)
ag_co1.calc = EMT()
energy = opt_and_get_Gibbs_energy(ag_co1)
G.update({"AgCO": energy})
if check:
    view(ag_co1)

deltaG = {}
deltaG.update({"Ag": 0.0})
deltaG.update({"AgO": G["AgO"]-(G["Ag"]+0.5*G["O2"])})
deltaG.update({"AgOH": G["AgOH"]-(G["Ag"]+G["HO"])})
deltaG.update({"AgCO": G["AgCO"]-(G["Ag"]+G["CO"])})
#
# Pourbaix diagram
#
#ag  = PourbaixEntry(npH= 0, nH2O=0, nPhi= 0, energy=deltaG["Ag"], name="Ag")
#oh1 = PourbaixEntry(npH=-1, nH2O=1, nPhi=-1, energy=deltaG["AgOH"]/6+1.0, name="1/6 ML OH*")
#o1  = PourbaixEntry(npH=-2, nH2O=2, nPhi=-2, energy=deltaG["AgO"]/6 +2.0, name="1/4 ML O*")
#agp = PourbaixEntry(npH=0, nH2O=0, nPhi=-1, energy=1.0, concentration=1.0e-6, name="Ag+")

ag  = PourbaixEntry(npH= 0, nH2O=0, nPhi= 0, energy=deltaG["Ag"], name="Ag")
oh1 = PourbaixEntry(npH=-1, nH2O=1, nPhi=-1, energy=deltaG["AgOH"], name="1/6 ML OH*")
o1  = PourbaixEntry(npH=-2, nH2O=2, nPhi=-2, energy=deltaG["AgO"], name="1/4 ML O*")
agp = PourbaixEntry(npH=0, nH2O=0, nPhi=-1, energy=0.8, concentration=1.0e-6, name="Ag+")

limits = [[0, 14], [-1, 1.5]]
#pbx = PhaseDiagram([ag, oh1, oh2, o1, o2], limits=limits)
pbx = PhaseDiagram([ag, oh1, o1, agp], limits=limits)
plotter = PhaseDiagramPlotter(pbx, diagram_type="Pourbaix")
plotter.get_diagram(limits=limits).show()

#
# AITD phase diagram
#
# A...O, B..CO
ag  = AITDEntry(nA=0, muA=0.0, nB=0, muB=0.0, energy=deltaG["Ag"],  name="Ag")
o1  = AITDEntry(nA=1, muA=0.5*G["O2"], nB=0, muB=0.0, energy=deltaG["AgO"],  name="AgO")
co1 = AITDEntry(nA=0, muA=0.0, nB=1, muB=G["CO"], energy=deltaG["AgCO"],  name="AgCO")
limits = [[-2, 2], [-2, 2]]
pbx = PhaseDiagram([ag, o1, co1], limits=limits, diagram_type="AITD")
plotter = PhaseDiagramPlotter(pbx, diagram_type="AITD")
plotter.get_diagram(limits=limits).show()

