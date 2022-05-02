import matplotlib.pyplot as plt
import numpy as np

def rmsd_hist(x, path, num_bins = 50, xlim =None, title=''):
    """
    """

    print(f'std: {np.std(x)}')
    
    n, bins, patches = plt.hist(x, num_bins, 
                                color ='green',
                                alpha = 0.7,
                                density =True)
      
    # plt.plot(bins)
    if xlim is not None:
         plt.xlim([0, xlim])
         plt.ylim([0, 0.6])
    plt.xlabel('RMSD', fontsize='xx-large')
    plt.ylabel('Frequency', fontsize='xx-large')
      
    plt.title(f'{title}\n\n', fontweight ="bold", fontsize='xx-large')
      
    plt.savefig(path)
    plt.close()