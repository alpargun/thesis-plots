
#%%

import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


# x axis
ks = range(1, 10)
 
# y axis
results = range(1,10)
 
plt.plot(ks, results)
plt.show()
plt.savefig('histogram.png', dpi=400)

# %%
