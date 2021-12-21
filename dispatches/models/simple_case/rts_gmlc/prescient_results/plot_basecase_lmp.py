import matplotlib.pyplot as plt

#Load up file again
with open('rts_results_all_prices_base_case.npy', 'rb') as f:
    dispatch = np.load(f)
    prices = np.load(f)
    
#Optionally plot histograms of prices and dispatch
(n, bins, patches) = plt.hist(lmp, bins=100, label='hst')
plt.show()


#Plot the dispatch profile
dispatch_np = dispatch.to_numpy()
(n, bins, patches) = plt.hist(dispatch_np, bins=100, label='hst')
plt.show()