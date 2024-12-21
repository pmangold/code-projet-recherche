import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from fedavg import fed_avg, compute_theta_star

import matplotlib

# plot parameters
matplotlib.rc('font', **{'family': 'serif',
                         'serif': ['Computer Modern'],
                         'size': 16})
matplotlib.rc('text', usetex=True)
markers = ['o', '^', 'v', '*', '+', 'x']


# general parameters 
n_local_list = [1, 2, 4, 8, 16, 32]
n_agents = 10
n_runs = 10

n_features = 10
n_data = 100


# Simulation parameters
step = 0.0005
n_rounds = 500
batch_size = 1
lambda_ = 1e-4


# rng for reproducibility
rng = np.random.default_rng(seed=42)

# generate data
for data_type in {"homogeneous", "heterogeneous"}:
    if data_type == "homogeneous":
        datas = [ make_blobs(n_samples=n_data, centers=2,
                             n_features=n_features, cluster_std=10,
                             random_state=42)
                  for j in range(n_agents) ]
        
    else:
        datas = [ make_blobs(n_samples=n_data, centers=2,
                             n_features=n_features, cluster_std=10,
                             random_state=j)
                  for j in range(n_agents) ]
        
    # compute solution
    theta_star = compute_theta_star(datas, lambda_)
    
    
    # create figure
    fig, ax = plt.subplots(figsize=(4,3))

    for i, n_local in enumerate(n_local_list):
        # run algorithm
        thetas_fedavg = fed_avg(clients_data=datas, theta=np.zeros(n_features),
                                step=step, lambda_=lambda_,
                                n_local=n_local, n_rounds=n_rounds)
    
        
        # plot result
        plt.plot(np.linalg.norm(thetas_fedavg - theta_star, axis=1)**2,
                 label="H="+str(n_local), marker=markers[i],
                 markevery=500//10)

    plt.ylim([1e-9, None])
    plt.yscale("log")

    plt.xlabel("Nombre de Communications")
    plt.ylabel("Erreur Quadratique")
    plt.savefig("local_training_" + data_type + ".pdf", bbox_inches="tight")


fig, ax = plt.subplots(figsize=(2,3))
lines = []
for i, n_local in enumerate(n_local_list):
    line, = ax.plot([1], [4], label="H="+str(n_local), marker=markers[i])
    lines.append(line)
labels = [line.get_label() for line in lines]

fig2, ax2 = plt.subplots(figsize=(2,3))
ax2.axis('off')
legend = ax2.legend(lines, labels, loc='center')
plt.savefig("legend.pdf", bbox_inches="tight")

