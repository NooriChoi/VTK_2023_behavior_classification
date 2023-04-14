# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:31:14 2023

@author: Noori
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def umap_figure(embedding):
    e_max = np.amax(embedding, axis=0)
    e_min = np.amin(embedding, axis=0)
    
    # assign colors to time windows by the distance from the origin
    dist_colors = [((x[0] - e_min[0])/(e_max[0] - e_min[0]), 
                    (x[1] - e_min[1])/(e_max[1] - e_min[1]), 
                    0.5) for x in embedding]
    
    # plot UMAP latent projection without clustering
    plt.scatter(embedding[:, 0], embedding[:, 1], c=dist_colors, s=0.1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()
    plt.close()

def hdbscan_figure(embedding, cluster_labels):
    # assign colors to time windows by the clustering results
    color_palette = sns.color_palette('tab10')
    cluster_colors = np.array([color_palette[x] if x >= 0
                               else (0.5, 0.5, 0.5)
                               for x in cluster_labels])
    
    ### plot UMAP latent projection with clustering results
    #### general figure
    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors, s=0.1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()
    
def LSTM_umap_figure(embedding):
    e_max = np.amax(embedding, axis=0)
    e_min = np.amin(embedding, axis=0)
    
    # assign colors to time windows by the distance from the origin
    dist_colors = [((x[0] - e_min[0])/(e_max[0] - e_min[0]), 
                    (x[1] - e_min[1])/(e_max[1] - e_min[1]), 
                    0.5) for x in embedding]
    
    # plot UMAP latent projection without clustering
    plt.scatter(embedding[:, 0], embedding[:, 1], c=dist_colors)
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()
    plt.close()
    
def LSTM_hdbscan_figure(embedding, cluster_labels, true_labels):
    # assign colors to time windows by the clustering results
    color_palette = sns.color_palette('tab10')
    cluster_colors = np.array([color_palette[x] if x >= 0
                               else (0.5, 0.5, 0.5)
                               for x in cluster_labels])
    
    ## plot UMAP latent projection with clustering results
    true_label_unique = list(set(true_labels))
    n_true_label = len(true_label_unique)
    markers = ["o", "x", "^", "s", "*"]
    
    fig, ax = plt.subplots()
    for i in range(n_true_label):
        label_idx = [idx for idx, e in enumerate(true_labels) if e == true_label_unique[i]]
        
        x = embedding[:, 0][label_idx]
        y = embedding[:, 1][label_idx]
        color = cluster_colors[label_idx]
        
        ax.scatter(x, y, c=color, marker=markers[i])
    
    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]

    handles = [f("s", cluster_colors[i]) for i in range(2)]
    handles += [f(markers[i], "k") for i in range(2)]
    
    labels = list(set(cluster_labels)) + true_label_unique
    
    plt.legend(handles, labels, loc = 'upper right')
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()
    plt.close()
