import numpy as np
import matplotlib.pyplot as plt
from rl_envs.qu_pulse_env import QuPulseEnv

env = QuPulseEnv()
o = env.reset()
acts = []
obses = [o.copy()]
rews = []
dones = []
for step in range(10):
    a = env.action_space.sample()
    otp1, r, d, _ = env.step(a)

    acts.append(a)
    obses.append(otp1)
    rews.append(r)
    dones.append(d)

    o = otp1.copy()

fig, axs = plt.subplots(4)
ax = axs[0]
ax.plot(np.array(obses))
ax.set_ylabel('State')
ax = axs[1]
for i, (act, ls) in enumerate(zip(np.array(acts).T, ('solid', 'dashed', 'dotted'))):
    ax.step(range(len(act)), np.array(act).astype(int)-1, ls=ls, lw=7, alpha=0.3, label=f'act #{i}')
ax.set_ylabel('Action')
ax.legend(loc='best')
ax = axs[2]
ax.plot(np.array(rews))
ax.set_ylabel('Reward')
ax = axs[3]
ax.plot(np.array(dones))
ax.set_ylabel('Done')

plt.show()