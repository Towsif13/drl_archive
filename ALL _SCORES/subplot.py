import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plt.figure(figsize=(20, 11))
plt.style.use('ggplot')

scores_dqn = np.loadtxt("DQN_scores.txt")
scores_d2qn = np.loadtxt("scores_d2qn.txt")
scores_d3qn = np.loadtxt("scores_d3qn.txt")

scores_dqn_dual_mem = np.loadtxt("scores_DQN_DUAL_MEM.txt")
scores_d2qn_dual_mem = np.loadtxt("scores_DDQN_DUAL_MEM.txt")
scores_d3qn_dual_mem = np.loadtxt("scores_D3QN_DUAL_MEM.txt")

plt.figure(1)


#DQN
plt.subplot(221)
scores_ma_100 = moving_average(scores_dqn, n = 100)
scores_ma_1000 = moving_average(scores_dqn, n = 1000)

plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'r')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'r', label = "DQN")

scores_ma_100 = moving_average(scores_dqn_dual_mem, n = 100)
scores_ma_1000 = moving_average(scores_dqn_dual_mem, n = 1000)

plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'b')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'b', label = "DQN with dual mem")

plt.xlim(xmin=0 , xmax = 100_501) 
plt.ylim(ymin=-335)  

plt.ylabel('Reward', fontsize=5)
plt.xlabel('Episode',fontsize=5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.gca().set_title('DQN comparision', fontsize=10)
plt.legend(facecolor='white',fontsize=7)


#DDQN
plt.subplot(222)
scores_ma_100 = moving_average(scores_d2qn, n = 100)
scores_ma_1000 = moving_average(scores_d2qn, n = 1000)

plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'r')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'r', label = "DQN")

scores_ma_100 = moving_average(scores_d2qn_dual_mem, n = 100)
scores_ma_1000 = moving_average(scores_d2qn_dual_mem, n = 1000)

plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'b')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'b', label = "DQN with dual mem")

plt.xlim(xmin=0 , xmax = 100_501) 
plt.ylim(ymin=-335)  

plt.ylabel('Reward', fontsize=5)
plt.xlabel('Episode',fontsize=5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.gca().set_title('DDQN comparision', fontsize=10)
plt.legend(facecolor='white',fontsize=7)


#D3QN
plt.subplot(223)
scores_ma_100 = moving_average(scores_d3qn, n = 100)
scores_ma_1000 = moving_average(scores_d3qn, n = 1000)

plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'r')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'r', label = "DQN")

scores_ma_100 = moving_average(scores_d3qn_dual_mem, n = 100)
scores_ma_1000 = moving_average(scores_d3qn_dual_mem, n = 1000)

plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'b')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'b', label = "DQN with dual mem")

plt.xlim(xmin=0 , xmax = 100_501) 
plt.ylim(ymin=-335)  

plt.ylabel('Reward', fontsize=5)
plt.xlabel('Episode',fontsize=5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.gca().set_title('D3QN comparision', fontsize=10)
plt.legend(facecolor='white',fontsize=7)


#ALL
plt.subplot(224)
scores_ma_100 = moving_average(scores_dqn, n = 100)
scores_ma_1000 = moving_average(scores_dqn, n = 1000)

plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'b')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'b', label = "DQN with dual mem")

scores_ma_100 = moving_average(scores_d2qn_dual_mem, n = 100)
scores_ma_1000 = moving_average(scores_d2qn_dual_mem, n = 1000)

plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'r')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'r', label = "DDQN with dual mem")

scores_ma_100 = moving_average(scores_d3qn_dual_mem, n = 100)
scores_ma_1000 = moving_average(scores_d3qn_dual_mem, n = 1000)

plt.plot(np.arange(len(scores_ma_100)), scores_ma_100, alpha = 0.1, color = 'g')
plt.plot(np.arange(len(scores_ma_1000)), scores_ma_1000, lw = '0.65', alpha = 1, color = 'g', label = "D3QN with dual mem")

plt.xlim(xmin=0 , xmax = 100_501) 
plt.ylim(ymin=-335)  

plt.ylabel('Reward', fontsize=5)
plt.xlabel('Episode',fontsize=5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.gca().set_title('Comparision of all algorithms', fontsize=10)
#plt.title('All DQN comparison', fontsize=15)
plt.legend(facecolor='white',fontsize=7)


plt.subplots_adjust(left=None,
                    bottom=None, 
                    right=None, 
                    top=None, 
                    wspace=0.07, 
                    hspace=None)
plt.savefig('collage.png',bbox_inches = 'tight', dpi=350)
#plt.show()