from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter

API = "cmvjQza1i5mk7Gt5uP"

mpr = MPRester(API)
entries = mpr.get_pourbaix_entries(["Cu"])
pbx = PourbaixDiagram(entries)
plotter = PourbaixPlotter(pbx)
plotter.get_pourbaix_plot().show()
