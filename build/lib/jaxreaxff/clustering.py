"""
K-means like clustering code to cluster similar structures together
to minimize memory and computational cost

accelerated using numba

Author: Mehmet Cagri Kaymak
"""
import numpy as onp
import numba

@numba.njit
def calculate_cost_and_new_centers(single_X, centroids, counts):
  '''
  Calculate the cost differences by assigning X to each centroid and
  the updated centroid, if the X becomes a part of that group
  '''
  # calculate change in total comp.
  change_cost = onp.zeros(len(centroids))
  new_cents = []
  for i in range(len(centroids)):
    max_vals = onp.where(centroids[i] > single_X, centroids[i], single_X)

    new_cost = sum(calculate_single_cost(max_vals)) * (counts[i] + 1)
    old_cost = sum(calculate_single_cost(centroids[i])) * counts[i]

    change_cost[i] = new_cost - old_cost
    new_cents.append(max_vals)

  return change_cost, new_cents

@numba.njit
def calculate_single_cost(sizes):
  '''
  Calculate the cost of a group
  '''
  # N x K where N is # atoms and K is # far neighbors
  nonbonded = sizes[0] * sizes[2]
  # N X L + 3_body size + 4_body size
  # where N is # atoms and L is # close neighbors
  bonded = sizes[0] * sizes[3] + sizes[5] + sizes[6]
  # cost of hbond
  hbond = sizes[7] + sizes[8] + sizes[9] * sizes[0] + sizes[10] * sizes[0]
  return bonded + hbond, nonbonded

@numba.njit
def calculate_total_cost(X,counts,centroids):
  '''
  Calculate the cost of current grouping of the structures
  '''
  total_bonded = 0
  total_nonbonded = 0
  for i in range(len(counts)):
    bonded, nonbonded = calculate_single_cost(centroids[i])
    total_bonded += bonded * counts[i]
    total_nonbonded += nonbonded * counts[i]
  return total_bonded, total_nonbonded

@numba.njit
def assign_labels(X, centroids):
  '''
  Assign group label to each system, for given centroids
  '''
  counts = onp.zeros(len(centroids))
  labels = onp.zeros(len(X),dtype=onp.int32)
  #iteration order changes the result, so shuffle the order to randomize
  idx = onp.arange(len(X))
  onp.random.shuffle(idx)
  for i in idx:
    costs, new_cents = calculate_cost_and_new_centers(X[i], centroids, counts)
    min_ind = onp.argmin(costs)
    labels[i] = min_ind
    counts[min_ind] += 1
    centroids[min_ind] = new_cents[min_ind]

  for i in range(len(counts)):
    if counts[i] == 0:
      centroids[i,:] = 0.0

  return labels,counts,centroids

def modified_kmeans(system_size_dicts ,k=3,
                    max_iterations=100, rep_count=10, print_mode=True):
  '''
  X: multidimensional data
  k: number of clusters
  max_iterations: number of repetitions before clusters are established

  Steps:
  1. Convert data to numpy aray
  2. Pick indices of k random point without replacement
  3. Find class (P) of each data point using euclidean distance
  4. Stop when max_iteration are reached of P matrix doesn't change

  '''

  selected_keys = ['num_atoms','periodic_image_count',
                   'far_nbr_size', 'close_nbr_size','filter2_size',
                   'filter3_size','filter4_size',
                   'hbond_size', 'hbond_h_size', 'hbond_filter_far_size',
                   'hbond_filter_close_size']
  all_lists = []
  for s in system_size_dicts:
    my_list = [s[k] for k in selected_keys]
    all_lists.append(my_list)

  X = onp.array(all_lists)

  min_cost = float('inf')
  min_bonded = float('inf')
  min_nonbonded = float('inf')
  idx = onp.arange(len(X))
  centroids = X[idx, :]
  counts = onp.ones(len(centroids))
  bonded, nonbonded = calculate_total_cost(X,counts,centroids)
  if print_mode:
    print("Cost without aligning:", nonbonded + bonded)
    print("nonbonded:            ", nonbonded)
    print("bonded:               ", bonded)
  # Try the clustering rep_count times and pick the one with the lowest cost
  for r in range(rep_count):
    idx = onp.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]

    P_labels,counts,centroids = assign_labels(X, centroids)
    bonded, nonbonded = calculate_total_cost(X,counts,centroids)
    # update the centroids till convergence or reaching max iteration
    for iter_c in range(max_iterations):
      labels,counts,centroids = assign_labels(X, centroids)
      if iter_c != 0 and onp.array_equal(P_labels,labels):break
      P_labels = labels
      bonded, nonbonded = calculate_total_cost(X,counts,centroids)
      cost = bonded + nonbonded
      #print(cost,part1,part2)
    if cost < min_cost:
      min_cost = cost
      min_labels = P_labels
      min_centr = centroids
      min_counts = counts
      min_bonded = bonded
      min_nonbonded = nonbonded
  if print_mode:
    print("Number of clusters:   ", k)
    print("Cost after aligning:  ", min_cost)
    print("nonbonded:            ", min_nonbonded)
    print("bonded:               ", min_bonded)
  cluster_dicts = []
  for i in range(len(min_centr)):
    dict_min_centr = {}
    for j,k in enumerate(selected_keys):
      dict_min_centr[k] = min_centr[i,j]
    cluster_dicts.append(dict_min_centr)

  return min_labels,cluster_dicts,min_counts,min_cost



