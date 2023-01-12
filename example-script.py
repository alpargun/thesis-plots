#%%

import matplotlib
import matplotlib.pyplot as plt


# Configure matplotlib for LaTeX

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex", # make sure you have pdflatex installed or change the tex distribution
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

ks = range(1, 10) # x axis
results = range(1,10) # y axis
 
plt.plot(ks, results)
plt.show()
plt.savefig('histogram.png', dpi=400)

# %% Seaborn example

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5, rc={'text.usetex' : True})

x = np.linspace(-50,50,100)
y = np.sin(x)**2/x

fig = plt.figure(1)
sns.set_style('white')
sns.kdeplot(np.array(y), label='hey')
fig.gca().set(xlabel=r'$e(t_0)$ [s]', ylabel='PDF')
fig.savefig("seaborntest.png")
