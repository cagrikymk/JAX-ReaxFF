#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 04:02:24 2020

@author: cagri
"""
from scipy.spatial import distance
import numpy as onp
import numba

def calculate_dist(X, Y):
    cost1 = onp.sum(onp.abs(X[:4] - Y[:4]))
    cost2 = abs(X[4] * (X[5] ** 2) - Y[4] * (Y[5] ** 2))

    return cost1 + cost2

@numba.njit
def calc_cost_and_centers(X, labels, k):
    counts = onp.zeros(k)
    max_centers = onp.zeros(shape=(k,len(X[0])))
    for i in range(len(X)):
        counts[labels[i]] += 1
        max_centers[labels[i]] = onp.where(max_centers[labels[i]] > X[i], max_centers[labels[i]], X[i])

    #cost,part1,part2 = calculate_total_cost(X,counts,max_centers)
    return max_centers#,cost

@numba.njit
def calculate_cost_and_new_centers(single_X, centroids, counts):
    # calculate change in total comp.
    change_cost = onp.zeros(len(centroids))
    new_cents = []
    for i in range(len(centroids)):

        max_vals = onp.where(centroids[i] > single_X, centroids[i], single_X)
        change1 = max_vals - centroids[i]
        change2 = max_vals - single_X
        change = onp.sum(change1 * (counts[i]+1))
        #solver_cost_change = change1[-1]**3 * (counts[i]+1) - centroids[i][-1]**3 * (counts[i])
        prev = centroids[i][4] * (centroids[i][5] ** 2) * counts[i]  #nonbounded cost: image count * (atom_count)**2
        curr = max_vals[4] * (max_vals[5] ** 2) *  (counts[i]+1)
        change = change + curr - prev

        change_cost[i] = change# + solver_cost_change
        new_cents.append(max_vals)

    return change_cost,new_cents


def calculate_total_cost(X,counts,centroids):
    total_cost = 0
    total_part1 = 0
    total_part2 = 0

    for i in range(len(counts)):
        part1= onp.sum(centroids[i][:4] * counts[i])
        total_cost+= part1
        part2 = centroids[i][4] * (centroids[i][5] ** 2) * counts[i] #+ (centroids[i][5] ** 3) * counts[i]
        total_cost+= part2
        total_part1 += part1
        total_part2 += part2
    return total_cost,total_part1,total_part2

@numba.njit
def assign_labels(X, centroids):
    counts = onp.zeros(len(centroids))
    labels = onp.zeros(len(X),dtype=onp.int32)
    #iteration order changes the result, so shuffle the order to randomize
    idx = onp.arange(len(X))
    onp.random.shuffle(idx)
    for i in idx:
        costs,new_cents = calculate_cost_and_new_centers(X[i], centroids, counts)
        min_ind = onp.argmin(costs)
        labels[i] = min_ind
        counts[min_ind] += 1
        centroids[min_ind] = new_cents[min_ind]

    for i in range(len(counts)):
        if counts[i] == 0:
             centroids[i,:] = 0.0

    return labels,counts,centroids

def modified_kmeans(systems,k=3,max_iterations=100, rep_count=10, print_mode=True):

    all_lists = []
    for s in systems:
        #2-body, 3-body, 4-body, hbond count, image count, atom count
        my_list = [s.global_body_2_count,s.global_body_3_count,s.global_body_4_count, s.global_hbond_count, len(s.all_shift_comb),(s.real_atom_count)]
        all_lists.append(my_list)

    X = onp.array(all_lists)

    '''
    X: multidimensional data
    k: number of clusters
    max_iterations: number of repetitions before clusters are established

    Steps:
    1. Convert data to numpy aray
    2. Pick indices of k random point without replacement
    3. Find class (P) of each data point using euclidean distance
    4. Stop when max_iteration are reached of P matrix doesn't change

    Return:
    np.array: containg class of each data point
    '''
    min_cost = 999999999999999999999
    idx = onp.arange(len(X))
    centroids = X[idx, :]
    counts = onp.ones(len(centroids))
    cost,part1,part2 = calculate_total_cost(X,counts,centroids)
    if print_mode:
        print("Cost without aligning:", cost)
        print("nonbounded:           ", part2)
        print("bounded:              ", part1)
    for r in range(rep_count):
        idx = onp.random.choice(len(X), k, replace=False)
        centroids = X[idx, :]


        P_labels,counts,centroids = assign_labels(X, centroids)
        #centroids = calc_cost_and_centers(X, P_labels, k)
        cost,part1,part2 = calculate_total_cost(X,counts,centroids)

        for iter_c in range(max_iterations):
            labels,counts,centroids = assign_labels(X, centroids)
            #centroids = calc_cost_and_centers(X, P_labels, k)
            if iter_c != 0 and onp.array_equal(P_labels,labels):break
            P_labels = labels
            cost,part1,part2 = calculate_total_cost(X,counts,centroids)
            #print(cost,part1,part2)
        if cost < min_cost:
            min_cost = cost
            min_labels = P_labels
            min_centr = centroids
            min_counts = counts
            min_part1 = part1
            min_part2 = part2
    if print_mode:
        print("Number of clusters:   ", k)
        print("Cost after aligning:  ", min_cost)
        print("nonbounded:           ", min_part2)
        print("bounded:              ", min_part1)

    return min_labels,min_centr,min_counts,min_cost
def calc_cost(X, labels, k):
    counts = onp.zeros(k)
    max_centers = onp.zeros(shape=(k,len(X[0])))
    for i in range(len(X)):
        counts[labels[i]] += 1
        max_centers[labels[i]] = onp.where(max_centers[labels[i]] > X[i], max_centers[labels[i]], X[i])

    cost,part1,part2 = calculate_total_cost(X,counts,max_centers)
    return cost,part1,part2



