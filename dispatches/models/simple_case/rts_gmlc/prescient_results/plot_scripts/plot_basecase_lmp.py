import matplotlib.pyplot as plt

run_index=1

#Load up file again
with open(run_dir + '/rts_results_basecase_run_{}.npy'.format(run_index), 'rb') as f:
    dispatch = np.load(f)
    prices = np.load(f)
    
#Optionally plot histograms of prices and dispatch
#LMP
plt.figure()
(n, bins, patches) = plt.hist(lmp, bins=100, label='hst')

#Dispatch
plt.figure()
(n, bins, patches) = plt.hist(dispatch, bins=100, label='hst')
plt.show()