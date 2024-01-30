import fractions
import functools
import re
from collections import OrderedDict
from typing import List, Tuple, Dict

import numpy as np
from scipy.spatial import ConvexHull

import ase.units as units
from ase.formula import Formula

_solvated: List[Tuple[str, Dict[str, int], float, bool, float]] = []


def parse_formula(formula):
    aq = formula.endswith('(aq)')
    if aq:
        formula = formula[:-4]
    charge = formula.count('+') - formula.count('-')
    if charge:
        formula = formula.rstrip('+-')
    count = Formula(formula).count()
    return count, charge, aq


def float2str(x):
    f = fractions.Fraction(x).limit_denominator(100)
    n = f.numerator
    d = f.denominator
    if abs(n / d - f) > 1e-6:
        return '{:.3f}'.format(f)
    if d == 0:
        return '0'
    if f.denominator == 1:
        return str(n)
    return '{}/{}'.format(f.numerator, f.denominator)


def solvated(symbols):
    """Extract solvation energies from database.

    symbols: str
        Extract only those molecules that contain the chemical elements given by the symbols string (plus water and H+).

    Data from:
        Johnson JW, Oelkers EH, Helgeson HC (1992)
        Comput Geosci 18(7):899.
        doi:10.1016/0098-3004(92)90029-Q
    and:
        Pourbaix M (1966)
        Atlas of electrochemical equilibria in aqueous solutions.
        No. v. 1 in Atlas of Electrochemical Equilibria in Aqueous Solutions.
        Pergamon Press, New York.

    Returns list of (name, energy) tuples.
    """

    if isinstance(symbols, str):
        symbols = Formula(symbols).count().keys()
    if len(_solvated) == 0:
        for line in _aqueous.splitlines():
            energy, formula = line.split(',')
            name = formula + '(aq)'
            count, charge, aq = parse_formula(name)
            energy = float(energy) * 0.001 * units.kcal / units.mol
            _solvated.append((name, count, charge, aq, energy))

    references = []
    for name, count, charge, aq, energy in _solvated:
        for symbol in count:
            if symbol not in 'HO' and symbol not in symbols:
                break
        else:
            references.append((name, energy))
    return references


def bisect(A, X, Y, f):
    a = []
    for i in [0, -1]:
        for j in [0, -1]:
            if A[i, j] == -1:
                A[i, j] = f(X[i], Y[j])
            a.append(A[i, j])

    if np.ptp(a) == 0:
        A[:] = a[0]
        return
    if a[0] == a[1]:
        A[0] = a[0]
    if a[1] == a[3]:
        A[:, -1] = a[1]
    if a[3] == a[2]:
        A[-1] = a[3]
    if a[2] == a[0]:
        A[:, 0] = a[2]
    if not (A == -1).any():
        return
    i = len(X) // 2
    j = len(Y) // 2
    bisect(A[:i + 1, :j + 1], X[:i + 1], Y[:j + 1], f)
    bisect(A[:i + 1, j:], X[:i + 1], Y[j:], f)
    bisect(A[i:, :j + 1], X[i:], Y[:j + 1], f)
    bisect(A[i:, j:], X[i:], Y[j:], f)


def print_results(results):
    total_energy = 0.0
    print('reference    coefficient      energy')
    print('------------------------------------')
    for name, coef, energy in results:
        total_energy += coef * energy
        if abs(coef) < 1e-7:
            continue
        print('{:14}{:>10}{:12.3f}'.format(name, float2str(coef), energy))
    print('------------------------------------')
    print('Total energy: {:22.3f}'.format(total_energy))
    print('------------------------------------')


class Pourbaix:
    def __init__(self, references, formula=None, T=300.0, **kwargs):
        """Pourbaix object.

        references: list of (name, energy) tuples
            Examples of names: ZnO2, H+(aq), H2O(aq), Zn++(aq), ...
        formula: str
            Stoichiometry.  Example: ``'ZnO'``.  Can also be given as
            keyword arguments: ``Pourbaix(refs, Zn=1, O=1)``.
        T: float
            Temperature in Kelvin.
        """

        if formula:
            assert not kwargs
            kwargs = parse_formula(formula)[0]

        if 'O' not in kwargs:
            kwargs['O'] = 0
        if 'H' not in kwargs:
            kwargs['H'] = 0

        self.kT = units.kB * T
        self.references = []
        for name, energy in references:
            if name == 'O':
                continue
            count, charge, aq = parse_formula(name)
            if all(symbol in kwargs for symbol in count):
                self.references.append((count, charge, aq, energy, name))

        self.references.append(({}, -1, False, 0.0, 'e-'))  # an electron

        self.count = kwargs

        self.N = {'e-': 0}
        for symbol in kwargs:
            if symbol not in self.N:
                self.N[symbol] = len(self.N)

    def decompose(self, U, pH, verbose=True, concentration=1e-6):
        """Decompose material.

        U: float
            Potential in V.
        pH: float
            pH value.
        verbose: bool
            Default is True.
        concentration: float
            Concentration of solvated references.

        Returns optimal coefficients and energy:

        >>> from ase.phasediagram import Pourbaix, solvated
        >>> refs = solvated('CoO') + [('Co', 0.0), ('CoO', -2.509), ('Co3O4', -9.402)]
        >>> pb = Pourbaix(refs, Co=3, O=4)
        >>> coefs, energy = pb.decompose(U=1.5, pH=0, concentration=1e-6, verbose=True)
        0    HCoO2-(aq)    -3.974
        1    CoO2--(aq)    -3.098
        2    H2O(aq)       -2.458
        3    CoOH+(aq)     -2.787
        4    CoO(aq)       -2.265
        5    CoOH++(aq)    -1.355
        6    Co++(aq)      -0.921
        7    H+(aq)         0.000
        8    Co+++(aq)      1.030
        9    Co             0.000
        10   CoO           -2.509
        11   Co3O4         -9.402
        12   e-            -1.500
        reference    coefficient      energy
        ------------------------------------
        H2O(aq)                4      -2.458
        Co++(aq)               3      -0.921
        H+(aq)                -8       0.000
        e-                    -2      -1.500
        ------------------------------------
        Total energy:                 -9.596
        ------------------------------------
        """
        alpha = np.log(10) * self.kT
        entropy = -np.log(concentration) * self.kT

        # We want to minimize np.dot(energies, x) under the constraints:
        #
        #     np.dot(x, eq2) == eq1
        #
        # with bounds[i,0] <= x[i] <= bounds[i, 1].
        #
        # First two equations are charge and number of hydrogens, and the rest are the remaining species.

        eq1 = [0] + list(self.count.values())
        eq2 = []
        energies = []
        bounds = []
        names = []
        for count, charge, aq, energy, name in self.references:
            eq = np.zeros(len(self.N))
            eq[0] = charge
            for symbol, n in count.items():
                eq[self.N[symbol]] = n
            eq2.append(eq)
            if name in ['H2O(aq)', 'H+(aq)', 'e-']:
                bounds.append((-np.inf, np.inf))
                if name == 'e-':
                    energy = -U
                elif name == 'H+(aq)':
                    energy = -pH * alpha
            else:
                bounds.append((0, np.inf))
                if aq:
                    energy -= entropy
            if verbose:
                print('{:<5}{:10}{:10.3f}'.format(len(energies), name, energy))
            energies.append(energy)
            names.append(name)

        from scipy.optimize import linprog

        result = linprog(c=energies, A_eq=np.transpose(eq2), b_eq=eq1, bounds=bounds, options={'lstsq': True, 'presolve': True})

        if verbose:
            print_results(zip(names, result.x, energies))

        return result.x, result.fun

    def diagram(self, U, pH, plot=True, show=False, ax=None):
        """Calculate Pourbaix diagram.

        U: list of float
            Potentials in V.
        pH: list of float
            pH values.
        plot: bool
            Create plot.
        show: bool
            Open graphical window and show plot.
        ax: matplotlib axes object
            When creating plot, plot onto the given axes object. If none given, plot onto the current one.
        """
        a = np.empty((len(U), len(pH)), int)
        a[:] = -1
        colors = {}
        f = functools.partial(self.colorfunction, colors=colors)
        bisect(a, U, pH, f)
        compositions = [None]*len(colors)
        names = [ref[-1] for ref in self.references]
        for indices, color in colors.items():
            compositions[color] = ' + '.join(names[i] for i in indices if names[i] not in ['H2O(aq)', 'H+(aq)', 'e-'])
                                            
        text = []
        for i, name in enumerate(compositions):
            b = (a == i)
            x = np.dot(b.sum(1), U) / b.sum()
            y = np.dot(b.sum(0), pH) / b.sum()
            name = re.sub(r'(\S)([+-]+)', r'\1$^{\2}$', name)
            name = re.sub(r'(\d+)', r'$_{\1}$', name)
            text.append((x, y, name))

        if plot:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            if ax is None:
                ax = plt.gca()

            # rasterized pcolormesh has a bug which leaves a tiny white border. Unrasterized pcolormesh produces
            # unreasonably large files. Avoid this by using the more general imshow.
            ax.imshow(a, cmap=cm.Accent, extent=[min(pH), max(pH), min(U), max(U)], origin='lower', aspect='auto')

            for x, y, name in text:
                ax.text(y, x, name, horizontalalignment='center')
            ax.set_xlabel('pH')
            ax.set_ylabel('potential [V]')
            ax.set_xlim(min(pH), max(pH))
            ax.set_ylim(min(U), max(U))
            if show:
                plt.show()

        return a, compositions, text

    def colorfunction(self, U, pH, colors):
        coefs, _ = self.decompose(U, pH, verbose=False)
        indices = tuple(sorted(np.where(abs(coefs) > 1e-3)[0]))
        color = colors.get(indices)
        if color is None:
            color = len(colors)
            colors[indices] = color
        return color

_aqueous = """\
-93290,ZnO2--
-81190,ZnOH+
-67420,ZnO
-56690,H2O
-35200,Zn++
0,H+
"""

import numpy as np
refs = solvated("Zn")
pb = Pourbaix(refs, Zn=1, O=1)
U  = np.linspace(-2, 2, 200)
pH = np.linspace(-2, 16, 300)
pb.diagram(U, pH, plot=True, show=True)

