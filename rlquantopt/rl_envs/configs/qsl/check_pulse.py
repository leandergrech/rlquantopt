import pandas as pd
import numpy as np

data = pd.read_csv('optimization_at_qsl.csv')
tlist = data['time']
ghz = data['ghz']
raw = data['pulse']

import matplotlib.pyplot as plt

fig, axs = plt.subplots(3)
ax = axs[0]
ax.plot(tlist, ghz)
ax.set_ylabel('GHz')

ax = axs[1]
ax.plot(tlist, raw)
ax.set_ylabel('Raw')
print(np.divide(raw, ghz)/np.pi)

ax = axs[2]
raw_diff = np.diff(raw)
ax.plot(tlist[:-1], raw_diff)
print(f'Max diff = {max(raw_diff)} & Min diff = {min(raw_diff)}')
ax.set_ylabel('Raw diffs')


plt.show()


