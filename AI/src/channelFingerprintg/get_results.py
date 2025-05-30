import h5py
import numpy as np


directory_results = "/home/Research/models/ChannelFingerprinting/results/"
#file_results = "QExtractor_results_Dataset_Channels_sinusoid_dev_871_400_freq_2.4e9_sr_1e6_gain_0_60_512_batch256_val0.1_RMS_DSsin2.4dev1287-4800.h5"
#file_results = "QExtractor_results_Dataset_Channels_sinusoid_dev_128_400_freq_2.4e9_sr_1e6_gain_0_60_512_batch256_val0.1_RMS_DSsin2.4dev1287-4800.h5"
file_results = "QExtractor_results_Dataset_Channels_sinusoid_dev_5109_400_freq_2.4e9_sr_1e6_gain_0_60_512_batch256_val0.1_RMS_DSsin2.4dev1287-4800.h5"

print(file_results)

print("Opening File")
with h5py.File(directory_results+file_results, "r") as f:
    print("Getting info")
    lr = f['lr'][:]
    alpha = f['alpha'][:]
    beta = f['beta'][:]
    KDR_AB = f['KDR_AB'][:]
    KDR_AC = f['KDR_AC'][:]
    KDR_BC = f['KDR_BC'][:]

KDR_AB_avg = np.average(KDR_AB, axis=1)
KDR_AC_avg = np.average(KDR_AC, axis=1)
KDR_BC_avg = np.average(KDR_BC, axis=1)
print(KDR_AB_avg.shape)
#print(KDR_AB_avg)
min_KDR_AB = KDR_AB_avg.min()
max_KDR_AB = KDR_AB_avg.max()
print("max:",max_KDR_AB)
print("min:",min_KDR_AB)
index_min_AB = np.where(KDR_AB<=min_KDR_AB)
print("index_min:",index_min_AB)
print("index_min length:",len(index_min_AB[0]))
min_KDR_AC_avg_arr = []
min_KDR_BC_avg_arr = []
for i in index_min_AB[0]:
    min_KDR_AC_avg_arr.append(KDR_AC_avg[i])
    min_KDR_BC_avg_arr.append(KDR_BC_avg[i])
min_KDR_AC_avg_arr = np.array(min_KDR_AC_avg_arr)
min_KDR_BC_avg_arr = np.array(min_KDR_BC_avg_arr)

max_KDR_AC_avg = min_KDR_AC_avg_arr.max()
print(max_KDR_AC_avg)
index_max_AC = np.where(min_KDR_AC_avg_arr>=max_KDR_AC_avg)
print(index_max_AC[0])
max_KDR_BC_avg = min_KDR_BC_avg_arr.max()
print(max_KDR_BC_avg)
index_max_BC = np.where(min_KDR_BC_avg_arr>=max_KDR_BC_avg)
print(index_max_BC[0])

# for i in index_max_AC[0]:
#     print(i)
#     indexBest = index_min_AB[0][i]
#     print("KDR_AB",KDR_AB_avg[indexBest])
#     print("KDR_AC",KDR_AC_avg[indexBest])
#     print("KDR_BC",KDR_BC_avg[indexBest])
#     print("lr",lr[indexBest])
#     print("alpha",alpha[indexBest])
#     print("beta",beta[indexBest])

indexBest = index_min_AB[0][index_max_AC[0][0]]
print(indexBest)
print("KDR_AB",KDR_AB_avg[indexBest])
print("KDR_AC",KDR_AC_avg[indexBest])
print("KDR_BC",KDR_BC_avg[indexBest])
print("lr",lr[indexBest])
print("alpha",alpha[indexBest])
print("beta",beta[indexBest])