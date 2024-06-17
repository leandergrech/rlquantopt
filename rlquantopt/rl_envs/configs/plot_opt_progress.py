import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('iswap_nelder-mead_optimisation.csv', sep=' ')
    its = data['it'].to_numpy()
    fids = data['fidelity'].to_numpy()

    plt.plot(its, fids)
    plt.title('Fidelity when optimising iSWAP gate with Nelder-Mead')
    plt.xlabel('Iterations')
    plt.ylabel('Fidelity')
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.show()

