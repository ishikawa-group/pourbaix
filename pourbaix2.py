# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.
"""
This module is intended to be used to compute Pourbaix diagrams
of arbitrary compositions and formation energies. If you use
this module in your work, please consider citing the following:

General formalism for solid-aqueous equilibria from DFT:
    Persson et al., DOI: 10.1103/PhysRevB.85.235438
Decomposition maps, or Pourbaix hull diagrams
    Singh et al., DOI: 10.1021/acs.chemmater.7b03980
Fast computation of many-element Pourbaix diagrams:
    Patel et al., https://arxiv.org/abs/1909.00035 (submitted)
"""

from __future__ import annotations

import itertools
import logging
import re
import warnings
from copy import deepcopy
from functools import cmp_to_key, lru_cache, partial
from multiprocessing import Pool

import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.spatial import ConvexHull, HalfspaceIntersection

try:
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core.composition import Composition
from pymatgen.core.ion import Ion
from pymatgen.core.periodic_table import Element
from pymatgen.entries.compatibility import MU_H2O
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.util.coord import Simplex
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import Stringify

__author__ = "Sai Jayaraman"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "0.4"
__maintainer__ = "Joseph Montoya"
__credits__ = "Arunima Singh, Joseph Montoya, Anjli Patel"
__email__ = "joseph.montoya@tri.global"
__status__ = "Production"
__date__ = "Nov 1, 2012"

logger = logging.getLogger(__name__)

PREFAC = 0.0591


# TODO: Revise to more closely reflect PDEntry, invoke from energy/composition
# TODO: PourbaixEntries depend implicitly on having entry energies be
#       formation energies, should be a better way to get from raw energies
# TODO: uncorrected_energy is a bit of a misnomer, but not sure what to rename

class PourbaixEntry(MSONable, Stringify):
    """
    An object encompassing all data relevant to a solid or ion in a Pourbaix diagram. Each bulk solid/ion has an energy
    g of the form: e = e0 + 0.0591 log10(conc) - nO mu_H2O + (nH - 2nO) pH + phi (-nH + 2nO + q)
    
    Note that the energies corresponding to the input entries should be formation energies with respect to hydrogen and
    oxygen gas in order for the Pourbaix diagram formalism to work. This may be changed to be more flexible in the future.
    """
    def __init__(self, entry, entry_id=None, concentration=1e-6):
        """
        Args:
            entry (ComputedEntry/ComputedStructureEntry/PDEntry/IonEntry): An entry object
            entry_id ():
            concentration ():
        """
        self.entry = entry
        if isinstance(entry, IonEntry):
            self.concentration = concentration
            self.phase_type = "Ion"
            self.charge = entry.ion.charge
        else:
            self.concentration = 1.0
            self.phase_type = "Solid"
            self.charge = 0.0
        self.uncorrected_energy = entry.energy
        if entry_id is not None:
            self.entry_id = entry_id
        elif hasattr(entry, "entry_id") and entry.entry_id:
            self.entry_id = entry.entry_id
        else:
            self.entry_id = None

    @property
    def npH(self):
        """
        Returns:
        """
        return self.entry.composition.get("H", 0.0) - 2*self.entry.composition.get("O", 0.0)

    @property
    def nH2O(self):
        """
        Returns: Number of H2O.
        """
        return self.entry.composition.get("O", 0.0)

    @property
    def nPhi(self):
        """
        Returns: Number of H2O.
        """
        return self.npH - self.charge

    @property
    def name(self):
        """
        Returns: Name for entry
        """
        if self.phase_type == "Solid":
            return self.entry.composition.reduced_formula + "(s)"

        return self.entry.name

    @property
    def energy(self):
        """
        returns energy

        Returns (float): total energy of the Pourbaix
            entry (at pH, V = 0 vs. SHE)
        """
        # Note: this implicitly depends on formation energies as input
        return self.uncorrected_energy + self.conc_term - (MU_H2O*self.nH2O)

    @property
    def energy_per_atom(self):
        """
        energy per atom of the Pourbaix entry

        Returns (float): energy per atom
        """
        return self.energy / self.composition.num_atoms

    def energy_at_conditions(self, pH, V):
        """
        Get free energy for a given pH and V

        Args:
            pH (float): pH at which to evaluate free energy
            V (float): voltage at which to evaluate free energy

        Returns:
            free energy at conditions
        """
        return self.energy + self.npH*PREFAC*pH + self.nPhi*V

    @property
    def normalized_energy(self):
        """
        Returns:
             energy normalized by number of non H or O atoms, e. g. for Zn2O6, energy / 2 or for AgTe3(OH)3, energy / 4
        """
        return self.energy*self.normalization_factor

    def normalized_energy_at_conditions(self, pH, V):
        """
        Energy at an electrochemical condition, compatible with
        numpy arrays for pH/V input
        Args:
            pH (float): pH at condition
            V (float): applied potential at condition
        Returns:
            energy normalized by number of non-O/H atoms at condition
        """
        return self.energy_at_conditions(pH, V) * self.normalization_factor

    @property
    def conc_term(self):
        """
        Returns the concentration contribution to the free energy,
        and should only be present when there are ions in the entry
        """
        return PREFAC * np.log10(self.concentration)

    @property
    def normalization_factor(self):
        """
        Sum of number of atoms minus the number of H and O in composition
        """
        return 1.0 / (self.num_atoms - self.composition.get("H", 0) - self.composition.get("O", 0))

    @property
    def composition(self):
        """
        Returns composition
        """
        return self.entry.composition

    @property
    def num_atoms(self):
        """
        Return number of atoms in current formula. Useful for normalization
        """
        return self.composition.num_atoms

    def to_pretty_string(self) -> str:
        """
        :return: A pretty string representation.
        """
        if self.phase_type == "Solid":
            return self.entry.composition.reduced_formula + "(s)"

        return self.entry.name

    def __repr__(self):
        return (
            f"Pourbaix Entry : {self.entry.composition} with energy = {self.energy:.4f}, npH = {self.npH}, "
            f"nPhi = {self.nPhi}, nH2O = {self.nH2O}, entry_id = {self.entry_id} "
        )

    def __str__(self):
        return self.__repr__()


class MultiEntry(PourbaixEntry):
    """
    PourbaixEntry-like object for constructing multi-elemental Pourbaix diagrams.
    """
    def __init__(self, entry_list, weights=None):
        """
        Initializes a MultiEntry.

        Args:
            entry_list ([PourbaixEntry]): List of component PourbaixEntries
            weights ([float]): Weights associated with each entry. Default is None
        """
        if weights is None:
            self.weights = [1.0]*len(entry_list)
        else:
            self.weights = weights
        self.entry_list = entry_list

    @lru_cache
    def __getattr__(self, item):
        """
        Because most of the attributes here are just weighted averages of the entry_list, we save some space by
        having a set of conditionals to define the attributes
        """
        # Attributes that are weighted averages of entry attributes
        if item in ["energy", "npH", "nH2O", "nPhi", "conc_term", "composition", "uncorrected_energy"]:
            # TODO: Composition could be changed for compat with sum
            if item == "composition":
                start = Composition({})
            else:
                start = 0
            return sum((getattr(e, item)*w for e, w in zip(self.entry_list, self.weights)), start)

        # Attributes that are just lists of entry attributes
        if item in ["entry_id", "phase_type"]:
            return [getattr(e, item) for e in self.entry_list]

        # normalization_factor, num_atoms should work from superclass
        return self.__getattribute__(item)

    @property
    def name(self):
        """
        MultiEntry name, i. e. the name of each entry joined by ' + '
        """
        return " + ".join([e.name for e in self.entry_list])

    def __repr__(self):
        return (
            f"Multiple Pourbaix Entry: energy = {self.energy:.4f}, npH = {self.npH}, nPhi = {self.nPhi}, "
            f"nH2O = {self.nH2O}, entry_id = {self.entry_id}, species: {self.name}"
        )

    def __str__(self):
        return self.__repr__()

    def as_dict(self):
        """
        Returns: MSONable dict
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "entry_list": [e.as_dict() for e in self.entry_list],
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Args:
            d (): Dict representation

        Returns:
            MultiEntry
        """
        entry_list = [PourbaixEntry.from_dict(e) for e in d.get("entry_list")]
        return cls(entry_list, d.get("weights"))


# TODO: this class isn't particularly useful in its current form, could be
#       refactored to include information about the reference solid

class IonEntry(PDEntry):
    """
    Object similar to PDEntry, but contains an Ion object instead of a Composition object.

    .. attribute:: name

        A name for the entry. This is the string shown in the phase diagrams.
        By default, this is the reduced formula for the composition, but can be
        set to some other string for display purposes.
    """

    def __init__(self, ion, energy, name=None, attribute=None):
        """
        Args:
            ion: Ion object
            energy: Energy for composition.
            name: Optional parameter to name the entry. Defaults to the chemical formula.
        """
        self.ion = ion
        # Auto-assign name
        name = name if name else self.ion.reduced_formula
        super().__init__(composition=ion.composition, energy=energy, name=name, attribute=attribute)

    @classmethod
    def from_dict(cls, d):
        """
        Returns an IonEntry object from a dict.
        """
        return IonEntry(Ion.from_dict(d["ion"]), d["energy"], d.get("name"), d.get("attribute"))

    def as_dict(self):
        """
        Creates a dict of composition, energy, and ion name
        """
        d = {"ion": self.ion.as_dict(), "energy": self.energy, "name": self.name}
        return d

    def __repr__(self):
        return f"IonEntry : {self.composition} with energy = {self.energy:.4f}"

    def __str__(self):
        return self.__repr__()


ELEMENTS_HO = {Element("H"), Element("O")}


# TODO: the solids filter breaks some of the functionality of the
#       heatmap plotter, because the reference states for decomposition
#       don't include oxygen/hydrogen in the OER/HER regions

# TODO: create a from_phase_diagram class method for non-formation energy
#       invocation
# TODO: invocation from a MultiEntry entry list could be a bit more robust
# TODO: serialization is still a bit rough around the edges

class PourbaixDiagram(MSONable):
    """
    Class to create a Pourbaix diagram from entries
    """
    def __init__(self, entries: list[PourbaixEntry] | list[MultiEntry], comp_dict: dict[str, float] | None = None,
                 conc_dict: dict[str, float] | None = None, filter_solids: bool = True, nproc: int | None = None):
        """
        Args:
            entries ([PourbaixEntry] or [MultiEntry]): Entries list containing Solids and Ions or a list of MultiEntries
            comp_dict (dict[str, float]): Dictionary of compositions, defaults to equal parts of each elements
            conc_dict (dict[str, float]): Dictionary of ion concentrations, defaults to 1e-6 for each element
            filter_solids (bool): applying this filter to a Pourbaix
                diagram ensures all included solid phases are filtered by
                stability on the compositional phase diagram. Defaults to True.
                The practical consequence of this is that highly oxidized or reduced
                phases that might show up in experiments due to kinetic limitations
                on oxygen/hydrogen evolution won't appear in the diagram, but they are
                not actually "stable" (and are frequently overstabilized from DFT errors).
                Hence, including only the stable solid phases generally leads to the
                most accurate Pourbaix diagrams.
            nproc (int): number of processes to generate multientries with in parallel. Defaults to None (serial processing)
        """
        entries = deepcopy(entries)
        self.filter_solids = filter_solids

        # Get non-OH elements
        self.pbx_elts = list(set(itertools.chain.from_iterable([entry.composition.elements for entry in entries])) - ELEMENTS_HO)
        self.dim = len(self.pbx_elts) - 1

        # Process multientry inputs
        if isinstance(entries[0], MultiEntry):
            self._processed_entries = entries

            # Extract individual entries
            single_entries = list(set(itertools.chain.from_iterable([e.entry_list for e in entries])))
            self._unprocessed_entries = single_entries
            self._filtered_entries = single_entries
            self._conc_dict = None
            self._elt_comp = {k: v for k, v in entries[0].composition.items() if k not in ELEMENTS_HO}
            self._multielement = True

        # Process single entry inputs
        else:
            # --- comes here
            # Set default conc/comp dicts
            if not comp_dict:
                # --- comes here
                comp_dict = {elt.symbol: 1.0/len(self.pbx_elts) for elt in self.pbx_elts}
            if not conc_dict:
                # --- comes here
                conc_dict = {elt.symbol: 1e-6 for elt in self.pbx_elts}

            self._conc_dict = conc_dict
            self._elt_comp  = comp_dict
            self.pourbaix_elements = self.pbx_elts

            solid_entries = [entry for entry in entries if entry.phase_type == "Solid"]
            ion_entries   = [entry for entry in entries if entry.phase_type == "Ion"]

            # If a conc_dict is specified, override individual entry concentrations
            for entry in ion_entries:
                ion_elts = list(set(entry.composition.elements) - ELEMENTS_HO)
                # TODO: the logic here for ion concentration setting is in two
                #       places, in PourbaixEntry and here, should be consolidated
                if len(ion_elts) == 1:
                    entry.concentration = conc_dict[ion_elts[0].symbol]*entry.normalization_factor
                elif len(ion_elts) > 1 and not entry.concentration:
                    raise ValueError("Elemental concentration not compatible with multi-element ions")

            self._unprocessed_entries = solid_entries + ion_entries

            if not len(solid_entries + ion_entries) == len(entries):
                raise ValueError("All supplied entries must have a phase type of " 'either "Solid" or "Ion"')

            if self.filter_solids:
                # --- comes here
                # O is 2.46 b/c pbx entry finds energies referenced to H2O
                entries_HO = [ComputedEntry("H", 0), ComputedEntry("O", 2.46)]
                solid_pd = PhaseDiagram(solid_entries + entries_HO)
                solid_entries = list(set(solid_pd.stable_entries) - set(entries_HO))

            self._filtered_entries = solid_entries + ion_entries

            if len(comp_dict) > 1:
                self._multielement = True
                self._processed_entries = self._preprocess_pourbaix_entries(self._filtered_entries, nproc=nproc)
            else:
                # --- comes here
                self._processed_entries = self._filtered_entries
                self._multielement = False

        self._stable_domains, self._stable_domain_vertices = self.get_pourbaix_domains(self._processed_entries)

    @staticmethod
    def get_pourbaix_domains(pourbaix_entries, limits=None):
        """
        Returns a set of Pourbaix stable domains (i. e. polygons) in pH-V space from a list of pourbaix_entries

        This function works by using scipy's HalfspaceIntersection function to construct all of the 2-D polygons that form the
        boundaries of the planes corresponding to individual entry gibbs free energies as a function of pH and V. Hyperplanes
        of the form a*pH + b*V + 1 - g(0, 0) are constructed and supplied to HalfspaceIntersection, which then finds the
        boundaries of each Pourbaix region using the intersection points.

        Args:
            pourbaix_entries ([PourbaixEntry]): Pourbaix entries with which to construct stable Pourbaix domains
            limits ([[float]]): limits in which to do the pourbaix analysis

        Returns:
            Returns a dict of the form {entry: [boundary_points]}.
            The list of boundary points are the sides of the N-1 dim polytope bounding the allowable ph-V range of each entry.
        """
        if limits is None:
            limits = [[-2, 16], [-4, 4]]

        # Get hyperplanes
        hyperplanes = [np.array([-PREFAC*entry.npH, -entry.nPhi, 0, -entry.energy])*entry.normalization_factor for entry in pourbaix_entries]
        hyperplanes = np.array(hyperplanes)
        hyperplanes[:, 2] = 1

        max_contribs = np.max(np.abs(hyperplanes), axis=0)
        g_max = np.dot(-max_contribs, [limits[0][1], limits[1][1], 0, 1])

        # Add border hyperplanes and generate HalfspaceIntersection
        border_hyperplanes = [
            [-1,  0,  0,  limits[0][0]],
            [ 1,  0,  0, -limits[0][1]],
            [ 0, -1,  0,  limits[1][0]],
            [ 0,  1,  0, -limits[1][1]],
            [ 0,  0, -1, 2*g_max]
        ]
        hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
        interior_point = np.average(limits, axis=1).tolist() + [g_max]
        hs_int = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))

        # organize the boundary points by entry
        pourbaix_domains = {entry: [] for entry in pourbaix_entries}
        for intersection, facet in zip(hs_int.intersections, hs_int.dual_facets):
            for v in facet:
                if v < len(pourbaix_entries):
                    this_entry = pourbaix_entries[v]
                    pourbaix_domains[this_entry].append(intersection)

        # Remove entries with no Pourbaix region
        pourbaix_domains = {k: v for k, v in pourbaix_domains.items() if v}
        pourbaix_domain_vertices = {}

        for entry, points in pourbaix_domains.items():
            points = np.array(points)[:, :2]

            # Initial sort to ensure consistency
            points = points[np.lexsort(np.transpose(points))]
            center = np.average(points, axis=0)
            points_centered = points - center

            # Sort points by cross product of centered points, isn't strictly necessary but useful for plotting tools
            points_centered = sorted(points_centered, key=cmp_to_key(lambda x, y: x[0]*y[1] - x[1]*y[0]))
            points = points_centered + center

            # Create simplices corresponding to Pourbaix boundary
            simplices = [Simplex(points[indices]) for indices in ConvexHull(points).simplices]
            pourbaix_domains[entry] = simplices
            pourbaix_domain_vertices[entry] = points

        return pourbaix_domains, pourbaix_domain_vertices

    @classmethod
    def from_dict(cls, d):
        """
        Args:
            d (): Dict representation.

        Returns:
            PourbaixDiagram
        """
        decoded_entries = MontyDecoder().process_decoded(d["entries"])
        return cls(decoded_entries, d.get("comp_dict"), d.get("conc_dict"), d.get("filter_solids"))


class PourbaixPlotter:
    """
    A plotter class for phase diagrams.
    """

    def __init__(self, pourbaix_diagram):
        """
        Args:
            pourbaix_diagram (PourbaixDiagram): A PourbaixDiagram object.
        """
        self._pbx = pourbaix_diagram

    def show(self, *args, **kwargs):
        """
        Shows the Pourbaix plot

        Args:
            *args: args to get_pourbaix_plot
            **kwargs: kwargs to get_pourbaix_plot

        Returns:
            None
        """
        plt = self.get_pourbaix_plot(*args, **kwargs)
        plt.show()

    def get_pourbaix_plot(self, limits=None, title="", label_domains=True,
                          label_fontsize=20, show_water_lines=True, show_neutral_axes=True, plt=None):
        """
        Plot Pourbaix diagram.

        Args:
            limits: 2D list containing limits of the Pourbaix diagram of the form [[xlo, xhi], [ylo, yhi]]
            title (str): Title to display on plot
            label_domains (bool): whether to label Pourbaix domains
            label_fontsize: font size for domain labels
            show_water_lines: whether to show dashed lines indicating the region of water stability.
            show_neutral_axes; whether to show dashed horizontal and vertical lines at 0 V and pH 7, respectively.
            plt (pyplot): Pyplot instance for plotting

        Returns:
            plt (pyplot) - matplotlib plot object with Pourbaix diagram
        """
        if limits is None:
            limits = [[-2, 16], [-3, 3]]

        plt = plt or pretty_plot(16)

        xlim = limits[0]
        ylim = limits[1]

        ax = plt.gca()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        lw = 3

        if show_water_lines:
            h_line = np.transpose([[xlim[0], -xlim[0]*PREFAC], [xlim[1], -xlim[1]*PREFAC]])
            o_line = np.transpose([[xlim[0], -xlim[0]*PREFAC + 1.23], [xlim[1], -xlim[1]*PREFAC + 1.23]])
            plt.plot(h_line[0], h_line[1], "r--", linewidth=lw)
            plt.plot(o_line[0], o_line[1], "r--", linewidth=lw)

        if show_neutral_axes:
            neutral_line = np.transpose([[7, ylim[0]], [7, ylim[1]]])
            V0_line = np.transpose([[xlim[0], 0], [xlim[1], 0]])
            plt.plot(neutral_line[0], neutral_line[1], "k-.", linewidth=lw)
            plt.plot(V0_line[0], V0_line[1], "k-.", linewidth=lw)

        for entry, vertices in self._pbx._stable_domain_vertices.items():
            print(entry)
            center = np.average(vertices, axis=0)
            x, y = np.transpose(np.vstack([vertices, vertices[0]]))
            plt.plot(x, y, "k-", linewidth=lw)

            if label_domains:
                plt.annotate(generate_entry_label(entry), center, ha="center", va="center", fontsize=label_fontsize, color="b").draggable()

        plt.xlabel("pH")
        plt.ylabel("E (V)")
        plt.title(title, fontsize=16, fontweight="bold")
        return plt

def generate_entry_label(entry):
    """
    Generates a label for the Pourbaix plotter

    Args:
        entry (PourbaixEntry or MultiEntry): entry to get a label for
    """
    if isinstance(entry, MultiEntry):
        return " + ".join([e.name for e in entry.entry_list])

    # TODO - a more elegant solution could be added later to Stringify
    # for example, the pattern re.sub(r"([-+][\d\.]*)", r"$^{\1}$", )
    # will convert B(OH)4- to B(OH)$_4^-$.
    # for this to work, the ion's charge always must be written AFTER
    # the sign (e.g., Fe+2 not Fe2+)
    string = entry.to_latex_string()
    return re.sub(r"()\[([^)]*)\]", r"\1$^{\2}$", string)


from pymatgen.ext.matproj import MPRester

API = "cmvjQza1i5mk7Gt5uP"

mpr = MPRester(API)
entries = mpr.get_pourbaix_entries(["Cu"])
pbx = PourbaixDiagram(entries)
plotter = PourbaixPlotter(pbx)
plotter.get_pourbaix_plot().show()

