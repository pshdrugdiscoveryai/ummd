import matplotlib.pyplot as plt
import seaborn as sns

# Figure styling
sns.set_style("darkgrid")
plt.rcParams.update({
    'font.family':          'sans-serif',
    'font.sans-serif':      ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':            7,
    'axes.labelsize':       9,
    'axes.titlesize':       9,
    'xtick.labelsize':      7,
    'ytick.labelsize':      7,
    'legend.fontsize':      7,
    'figure.titlesize':     11,
    'axes.linewidth':       0.75,
    'grid.linewidth':       0.5,
    'lines.linewidth':      1.5,
    'patch.linewidth':      0.75,
    'xtick.major.width':    0.75,
    'ytick.major.width':    0.75,
    'xtick.major.size':     3,
    'ytick.major.size':     3,
    'axes.edgecolor':       'lightgrey',
    'axes.grid':            True,
    'axes.spines.top':      True,
    'axes.spines.right':    True,
    'grid.color':           'lightgrey',
    'legend.frameon':       False,
})