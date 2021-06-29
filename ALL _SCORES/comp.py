import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plt.figure(figsize=(6, 3))
plt.style.use('ggplot')

scores_dqn = np.loadtxt("DQN_scores.txt")
scores_d2qn = np.loadtxt("scores_d2qn.txt")
scores_d3qn = np.loadtxt("scores_d3qn.txt")

scores_dqn_dual_mem = np.loadtxt("scores_DQN_DUAL_MEM.txt")
scores_d2qn_dual_mem = np.loadtxt("scores_DDQN_DUAL_MEM.txt")
scores_d3qn_dual_mem = np.loadtxt("scores_D3QN_DUAL_MEM.txt")

#DQN
scores_ma_100 = moving_average(scores_dqn, n = 100)
scores_ma_1000 = moving_average(scores_dqn, n = 1000)
plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'r')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'r', label = "DQN")
#DQN_dual
scores_ma_100 = moving_average(scores_dqn_dual_mem, n = 100)
scores_ma_1000 = moving_average(scores_dqn_dual_mem, n = 1000)
plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'b')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'b', label = "DQN with dual memory")


#D2QN
scores_ma_100 = moving_average(scores_d2qn, n = 100)
scores_ma_1000 = moving_average(scores_d2qn, n = 1000)
plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'g')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'g', label = "DDQN")
#D2QN_dual
scores_ma_100 = moving_average(scores_d2qn_dual_mem, n = 100)
scores_ma_1000 = moving_average(scores_d2qn_dual_mem, n = 1000)
plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'm')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'm', label = "DDQN with dual memory")

#D3QN
scores_ma_100 = moving_average(scores_d3qn, n = 100)
scores_ma_1000 = moving_average(scores_d3qn, n = 1000)
plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'c')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'c', label = "D3QN")
#D3QN_dual
scores_ma_100 = moving_average(scores_d3qn_dual_mem, n = 100)
scores_ma_1000 = moving_average(scores_d3qn_dual_mem, n = 1000)
plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'k')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'k', label = "D3QN with dual memory")



plt.xlim(xmin=0 , xmax = 100_501) 
plt.ylim(ymin=-335)  

plt.ylabel('Reward', fontsize=5)
plt.xlabel('Episode',fontsize=5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
#plt.title('All DQN comparison', fontsize=15)
plt.legend(facecolor='white',fontsize=5, loc=4)
plt.savefig('comparison_six_graph.png',bbox_inches = 'tight', dpi=350)
#plt.show()


# #DQN_target_mem
# scores_dqn_target_mem = np.loadtxt("C:/Users/Towsif/Desktop/Algo_DQNS/Ume code/Dual Memory SAR/DQN_DUAL_MEM/scores.txt")
# scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, alpha = 0.1, color = 'r')
# plt.plot(np.arange(len(scores_ma_dqn_target_1000_mem)), scores_ma_dqn_target_1000_mem, alpha = 1, color = 'r', label = "DQN_target_mem")