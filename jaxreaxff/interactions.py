"""
This file contains the functionality to create the interaction lists
and calculate required distances/angles

Although jax-md contains the same logic, it only support minimum image convention.
While this implementation is sub-obtimal, for training data containing relatively
small geometries, the performance should be enough.
Author: Mehmet Cagri Kaymak

"""
import jax
import jax.numpy as jnp
import numpy as onp
from jax_md.reaxff.reaxff_energy import calculate_bo
from jax_md.reaxff.reaxff_interactions import (filtration,
                                              body_3_candidate_fn, 
                                              body_4_candidate_fn,
                                              hbond_candidate_fn,
                                              ReaxFFNeighborLists,
                                              calculate_angle, 
                                              calculate_all_4_body_angles,
                                              calculate_all_hbond_angles_and_dists)
from jax_md.reaxff.reaxff_helper import safe_sqrt
from jax_md.reaxff.reaxff_energy import calculate_reaxff_energy
from jax_md.reaxff.reaxff_forcefield import ForceField
from jax_md.space import map_neighbor
from jax_md.util import safe_mask
from jax_md import dataclasses, util

DYNAMIC_INTERACTION_KEYS = ['far_nbr_size', 'close_nbr_size', 'filter2_size',
                            'filter3_size', 'filter4_size', 'hbond_size',
                            'hbond_h_size', 'hbond_filter_far_size',
                            'hbond_filter_close_size']

Array = util.Array

@dataclasses.dataclass
class NeighborList(object):
  '''
  Modified neighborList class
  '''
  idx: Array
  shift: Array
  did_buffer_overflow: Array

def calculate_dist(disp):
  '''
  Calculate the distance for given displacement
  '''
  return safe_sqrt(jnp.sum(disp**2))

def calculate_dist_and_angles(positons, structure, nbr_lists):
  '''
  Calculate the required distance and angles for the potential calculation
  '''
  # functions to calculate displacement and distance for neighbor lists
  calc_disp = map_neighbor(lambda x,y: x - y)
  calc_dist = map_neighbor(lambda x,y: calculate_dist(x - y))
  R = positons
  shifts = structure.periodic_image_shifts
  orth_matrix = structure.orth_matrix
  shift_pos = jnp.dot(orth_matrix,shifts.transpose()).transpose()
  N = len(R)

  atom_inds = jnp.arange(N).reshape(-1,1)
  # filtered bonded indices
  filtered_close_idx = nbr_lists.close_nbrs.idx[atom_inds,nbr_lists.filter2.idx]
  filtered_close_idx = jnp.where(nbr_lists.filter2.idx != -1,
                                 filtered_close_idx,
                                 N)
  filtered_close_shifts = nbr_lists.close_nbrs.shift[atom_inds,nbr_lists.filter2.idx]

  far_nbr_id = nbr_lists.far_nbrs.idx
  far_nbr_shifts = nbr_lists.far_nbrs.shift
  # calculate the distance and displacement for long neighbor list
  R_close_nbr = R[filtered_close_idx, :] + shift_pos[filtered_close_shifts]
  close_nbr_disps = calc_disp(R, R_close_nbr)
  close_nbr_dists = calc_dist(R, R_close_nbr)

  R_far_nbr = R[far_nbr_id,:] + shift_pos[far_nbr_shifts]
  far_nbr_dists = calc_dist(R, R_far_nbr)

  center = nbr_lists.filter3.idx[:, 0]
  neigh1_lcl = nbr_lists.filter3.idx[:,1]
  neigh2_lcl = nbr_lists.filter3.idx[:,2]
  # calculate 3-body angles
  body_3_cos_angles = jax.vmap(calculate_angle)(close_nbr_disps[center,
                                                            neigh1_lcl],
                                            close_nbr_disps[center,
                                                            neigh2_lcl])
  body_3_angles = body_3_cos_angles
  body_3_angles = safe_mask((body_3_cos_angles < 1) & (body_3_cos_angles > -1),
                            jnp.arccos, body_3_cos_angles).astype(R.dtype)
  # calculate 4-body angles
  body_4_angles = calculate_all_4_body_angles(nbr_lists.filter4.idx,
                                              filtered_close_idx,
                                              close_nbr_disps)

  if nbr_lists.filter_hb != None:
    hb_inds = nbr_lists.filter_hb.idx
    far_nbr_disps = calc_disp(R, R_far_nbr)
    hb_ang_dist = calculate_all_hbond_angles_and_dists(hb_inds,
                                                       close_nbr_disps,
                                                       far_nbr_disps)
  else:
    # filler arrays when there is no hbond
    hb_inds = jnp.zeros(shape=(0,3),dtype=center.dtype)
    hb_ang_dist = jnp.zeros(shape=(2,0),dtype=R.dtype)

  return (close_nbr_dists,
          far_nbr_dists,
          body_3_angles,
          body_4_angles,
          hb_ang_dist)



def reaxff_interaction_list_generator(force_field,
                                      close_cutoff = 5.0,
                                      far_cutoff = 10.0,
                                      use_hbond = False):
  '''
  Generates neighborlist generation function based on the provided
  cutoffs and info
  '''
  # cutoffs
  cutoff2 = force_field.cutoff2
  cutoff = force_field.cutoff
  FF_types_hb = force_field.nphb
  hbond_dist_cutff = 7.5
  hbond_bo_cutoff = 0.01

  # filter functions
  filter2_fn = filtration(lambda inds,vals: (inds,vals),
                          lambda x: x > 0.0,
                          is_dense=True)

  filter3_fn = filtration(body_3_candidate_fn, lambda x: x > 0.00001)

  filter4_fn = filtration(body_4_candidate_fn, lambda x: x > cutoff2)

  filter_hb_close_fn = filtration(lambda inds, vals, acceptor_mask:
                                  (inds, vals * acceptor_mask[inds]),
                                  lambda x: x > hbond_bo_cutoff,
                                  is_dense=True)

  filter_hb_far_fn = filtration(lambda inds, vals, acceptor_mask:
                                  (inds, vals * acceptor_mask[inds]),
                                  lambda x: (x < hbond_dist_cutff) & (x > 0.0),
                                  is_dense=True)

  filter_hb_fn = filtration(hbond_candidate_fn,
                                  lambda x: x > 0.0,
                                  is_dense=False)
  # Since we use 3 body list to generate 4-body, we need to not filter 3-body
  # interactions if we need them for 4-body
  new_body3_mask = force_field.body34_params_mask | force_field.body3_params_mask


  def create_distance_matrice(shift_pos, tiled_atom_pos1, orth_matrix):
    '''
    Calculate distance matrix for given positions and shift amount
    '''
    tiled_atom_pos1_trans = tiled_atom_pos1.swapaxes(0,1)
    shifted_tiled_atom_pos1_trans = tiled_atom_pos1_trans - shift_pos
    diff = tiled_atom_pos1 - shifted_tiled_atom_pos1_trans
    distance_matrix = safe_sqrt(jnp.square(diff).sum(axis=2))
    return distance_matrix

  def create_distance_matrices(positions, structure):
    '''
    Calculate distance matrices for given positions and shift values
    '''
    atom_positions = positions
    all_shift_comb = structure.periodic_image_shifts
    all_shift_comb = jnp.array(all_shift_comb,dtype=onp.float32)
    orth_matrix = structure.orth_matrix
    # create a mask for atoms
    mask = (structure.atom_types != -1).reshape(-1,1)
    num_atoms = len(atom_positions)

    atom_pos = atom_positions.reshape((num_atoms,1,3))
    shift_pos = jnp.dot(orth_matrix,all_shift_comb.transpose()).transpose()
    tiled_atom_pos1 = jnp.tile(atom_pos,(1,num_atoms,1))
    # create all the distance matrices
    distance_matrices = jnp.stack([create_distance_matrice(pos,
                                                           tiled_atom_pos1,
                                                           orth_matrix)
                                   for pos in shift_pos])
    # dist mat: [N, N, # shifts]
    distance_matrices = distance_matrices.swapaxes(0,2)
    mask_2d = jnp.dot(mask,mask.transpose())
    # zero out the masked distances
    distance_matrices = distance_matrices * mask_2d[..., jnp.newaxis]
    return distance_matrices


  def calculate_dense_nbr_size(dist_mats, cutoff):
    '''
    Calculate the dense neighbor size
    '''
    # dist_mats size: [N, N, # shifts]
    mask = (dist_mats < cutoff) & (dist_mats > 0)
    counts = jnp.sum(mask.astype(jnp.int32), axis=(1,2))
    max_c = jnp.max(counts)
    return max_c


  def create_dense_nbr_list(structure, dist_mats, max_nbr_size, cutoff):
    '''
    Create the dense neighbor list
    '''
    N = len(structure.positions)
    mapped_argwhere = jax.vmap(lambda vec:
                               jnp.argwhere(vec, size=max_nbr_size, fill_value=N))
    sel_inds = mapped_argwhere((dist_mats < cutoff) & (dist_mats > 0))

    neigh_inds = sel_inds[:,:,0]
    shift_inds = sel_inds[:,:,1]
    dists = dist_mats[jnp.arange(N).reshape(-1,1), neigh_inds, shift_inds]
    # zero out the distance for padded values (indicated by N)
    dists = dists * (neigh_inds != N)
    return neigh_inds, shift_inds, dists

  def calculate_filter2_size(close_nbr_inds, bo):
    '''
    Calculate the size of filtered bonded interactions
    Filtering happens based on the bond order term
    '''
    count = filter2_fn.count(candidate_args=((close_nbr_inds,bo)))
    return count

  def create_filter2(close_nbr_inds, bo, filter2_count):
    '''
    Create the filter bonded interaction list
    '''
    filter2 = filter2_fn.allocate_fixed(candidate_args=(close_nbr_inds,bo),
                                  capacity=filter2_count)
    return filter2

  def filter_close_nbr_list(close_nbr_inds, bo, filter2_inds):
    '''
    Apply the filter to close neighbor list
    '''
    N = len(bo)
    atom_inds = jnp.arange(N).reshape(-1,1)
    filtered_close_idx = close_nbr_inds[atom_inds,filter2_inds]
    filtered_close_idx = jnp.where(filter2_inds != -1,
                                   filtered_close_idx,
                                   N)
    filtered_bo = bo[atom_inds,filter2_inds]
    filtered_bo = jnp.where(filter2_inds != -1, filtered_bo, 0.0)

    return filtered_close_idx, filtered_bo

  def calculate_filter3_size(filtered_close_idx,
                             bo,
                             atom_types,
                             cutoff2,
                             new_body3_mask):
    '''
    Calculate the size of 3-body interactions
    '''
    count = filter3_fn.count(candidate_args=(filtered_close_idx,
                                             bo,
                                             atom_types,
                                             force_field.cutoff2,
                                             new_body3_mask))
    return count

  def create_filter3(filtered_close_idx,
                     bo,
                     atom_types,
                     cutoff2,
                     new_body3_mask,
                     filter3_count):
    '''
    Generate 3-body interaction list
    '''
    filter3 = filter3_fn.allocate_fixed(candidate_args=(filtered_close_idx,
                                                 bo,
                                                 atom_types,
                                                 force_field.cutoff2,
                                                 new_body3_mask),
                                        capacity=filter3_count)
    return filter3

  def calculate_filter4_size(filter3_ids,
                             filter2_ids,
                             bo,
                             atom_types,
                             cutoff2,
                             body4_params_mask):
    '''
    Calculate the size of 4-body interactions
    '''
    count = filter4_fn.count(candidate_args=(filter3_ids,
                                             filter2_ids,
                                             bo,
                                             atom_types,
                                             cutoff2,
                                             body4_params_mask))
    return count

  def create_filter4(filter3_ids,
                     filter2_ids,
                     bo,
                     atom_types,
                     cutoff2,
                     body4_params_mask,
                     filter4_count):
    '''
    Generate 4-body interaction list
    '''
    filter4 = filter4_fn.allocate_fixed(candidate_args=(filter3_ids,
                                                        filter2_ids,
                                                        bo,
                                                        atom_types,
                                                        cutoff2,
                                                        body4_params_mask),
                                        capacity=filter4_count)
    return filter4
  '''
  vec_calculate_hb_far_close_sizes(h_nbr_inds,h_nbr_pots,
                                   h_long_nbr_inds,h_long_nbr_dists,
                                   hb_nbr_mask)
  '''
  def calculate_hb_far_close_sizes(h_nbr_inds,h_nbr_pots,
                                   h_long_nbr_inds,h_long_nbr_dists,
                                   hb_nbr_mask):
    '''
    Calculate both close neighbor and far neigbor counts to be used
    for h-bond (acceptor donor pairs)

    donor (close(bonded neighbor) --H atom ------ acceptor (far neighbor)
    '''
    count_close = filter_hb_close_fn.count(candidate_args=(h_nbr_inds,
                                               h_nbr_pots,
                                               hb_nbr_mask))
    count_far = filter_hb_far_fn.count(candidate_args=(h_long_nbr_inds,
                                               h_long_nbr_dists,
                                               hb_nbr_mask))
    return jnp.array([count_far, count_close])

  def create_hb_far_close_filters(h_nbr_inds,h_nbr_pots,
                                  h_long_nbr_inds,h_long_nbr_dists,
                                  hb_acceptor_mask,
                                  max_hb_close_size,
                                  max_hb_far_size):
    '''
    Generate both close neighbor and far neigbor lists to be used
    for h-bond (acceptor donor pairs)

    '''
    filter_hb_close = filter_hb_close_fn.allocate_fixed(
                                          candidate_args=(h_nbr_inds,
                                                          h_nbr_pots,
                                                          hb_acceptor_mask),
                                          capacity=max_hb_close_size)
    filter_hb_far = filter_hb_far_fn.allocate_fixed(
                                          candidate_args=(h_long_nbr_inds,
                                                          h_long_nbr_dists,
                                                          hb_acceptor_mask),
                                          capacity=max_hb_far_size)
    return filter_hb_far, filter_hb_close

  def calculate_hb_count(hb_h_inds,
                         filter2_ids,
                         hb_close_ids,
                         far_nbr_ids,
                         hb_far_ids,
                         atom_types,
                         hb_params_mask):
    '''
    Calculate the size of final h-bond interaction list
    '''

    count = filter_hb_fn.count(candidate_args=(hb_h_inds,
                                             filter2_ids,
                                             hb_close_ids,
                                             far_nbr_ids,
                                             hb_far_ids,
                                             atom_types,
                                             hb_params_mask))
    return count


  def create_hb_filter(hb_h_inds,
                       filter2_ids,
                       hb_close_ids,
                       far_nbr_ids,
                       hb_far_ids,
                       atom_types,
                       hb_params_mask,
                       max_hb_size):
    '''
    Generate the final h-bond interaction list
    '''
    filter_hb = filter_hb_fn.allocate_fixed(candidate_args=(hb_h_inds,
                                                       filter2_ids,
                                                       hb_close_ids,
                                                       far_nbr_ids,
                                                       hb_far_ids,
                                                       atom_types,
                                                       hb_params_mask),
                                        capacity=max_hb_size)
    return filter_hb


  def batched_allocate(positions, structures, force_field, prev_counts=None):
    '''
    Calculate and generate the interaction lists for the given structures.
    Set the overflow flag if the new sizes exceeds the previous one
    '''

    # create the vmapped version of the interaction list count/generate func.
    vec_calc_dense_nbr_size = jax.vmap(calculate_dense_nbr_size,
                                       in_axes=(0,None))
    vec_create_dense_nbr = jax.vmap(create_dense_nbr_list,
                                       in_axes=(0, 0, None, None))
    vec_calculate_bo = jax.vmap(calculate_bo,
                                       in_axes=(0, 0, 0, 0, None))
    vec_calculate_filter2_size = jax.vmap(calculate_filter2_size)
    vec_create_filter2 = jax.vmap(create_filter2,
                                       in_axes=(0, 0, None))
    vec_filter_close_nbr_list = jax.vmap(filter_close_nbr_list)

    vec_calculate_filter3_size = jax.vmap(calculate_filter3_size,
                                          in_axes=(0,0,0,None,None))
    vec_create_filter3 = jax.vmap(create_filter3,
                                  in_axes=(0,0,0,None,None,None))
    vec_calculate_filter4_size = jax.vmap(calculate_filter4_size,
                                          in_axes=(0,0,0,0,None,None))
    vec_create_filter4 = jax.vmap(create_filter4,
                                  in_axes=(0,0,0,0,None,None,None))
    vec_calculate_hb_far_close_sizes = jax.vmap(calculate_hb_far_close_sizes)
    vec_create_hb_far_close_filters = jax.vmap(create_hb_far_close_filters,
                                               in_axes=(0,0,0,0,0,None,None))

    vec_calculate_hb_count = jax.vmap(calculate_hb_count,
                                      in_axes=(0,0,0,0,0,0,None))
    vec_create_hb_filter = jax.vmap(create_hb_filter,
                                      in_axes=(0,0,0,0,0,0,None, None))

    dist_mats = jax.vmap(create_distance_matrices)(positions, structures)
    did_overflow = False
    batch_size = len(dist_mats)

    # calculate the far neighbor size and compare it with the previous one
    far_nbr_sizes = vec_calc_dense_nbr_size(dist_mats,
                                            far_cutoff)
    new_max_far_size = jnp.max(far_nbr_sizes)
    new_counts = {}
    new_counts['far_nbr_size'] = new_max_far_size

    max_far_size = prev_counts['far_nbr_size']
    did_overflow = did_overflow | (new_max_far_size > max_far_size)
    # generate far neighbor interaction list
    [far_nbr_inds,
     far_nbr_shifts,
     far_dists] = vec_create_dense_nbr(structures,
                                        dist_mats,
                                        max_far_size,
                                        far_cutoff)
    far_nbr_list = NeighborList(far_nbr_inds, far_nbr_shifts,
                                jnp.zeros(shape=batch_size, dtype=jnp.bool_))

    # calculate the close neighbor size and compare it with the previous one
    close_nbr_sizes = vec_calc_dense_nbr_size(dist_mats,
                                             close_cutoff)
    new_max_close_size = jnp.max(close_nbr_sizes)
    new_counts['close_nbr_size'] = new_max_close_size
    max_close_size = prev_counts['close_nbr_size']
    did_overflow = did_overflow | (new_max_close_size > max_close_size)
    # generate close neighbor interaction list
    [close_nbr_inds,
     close_nbr_shifts,
     close_dists] = vec_create_dense_nbr(structures,
                                          dist_mats,
                                          max_close_size,
                                          close_cutoff)

    close_nbr_list = NeighborList(close_nbr_inds, close_nbr_shifts,
                                  jnp.zeros(shape=batch_size, dtype=jnp.bool_))

    # calculate the bond order values
    bo = vec_calculate_bo(close_nbr_inds,
                          close_dists,
                          structures.atom_types,
                          structures.atomic_nums,
                          force_field)
    # calculate the size of the filtered bonded interactions and compare with
    # the previous one
    filter2_count = vec_calculate_filter2_size(close_nbr_inds,bo)
    new_max_filter2_size = jnp.max(filter2_count)
    new_counts['filter2_size'] = new_max_filter2_size
    max_filter2_size = prev_counts['filter2_size']
    did_overflow = did_overflow | (new_max_filter2_size > max_filter2_size)

    filter2 = vec_create_filter2(close_nbr_inds,bo, max_filter2_size)
    filtered_close_idx, filtered_bo = vec_filter_close_nbr_list(close_nbr_inds, bo, filter2.idx)
    # calculate the size of the 3-body interactions and compare with
    # the previous one
    filter3_count = vec_calculate_filter3_size(filtered_close_idx,
                                               filtered_bo,
                                               structures.atom_types,
                                               force_field.cutoff2,
                                               new_body3_mask)
    new_max_filter3_size = jnp.max(filter3_count)
    new_counts['filter3_size'] = new_max_filter3_size

    max_filter3_size = prev_counts['filter3_size']
    did_overflow = did_overflow | (new_max_filter3_size > max_filter3_size)

    filter3 = vec_create_filter3(filtered_close_idx,
                                 filtered_bo,
                                 structures.atom_types,
                                 force_field.cutoff2,
                                 new_body3_mask,
                                 max_filter3_size)
    # calculate the size of the 4-body interactions and compare with
    # the previous one
    filter4_count = vec_calculate_filter4_size(filter3.idx,
                                               filtered_close_idx,
                                               filtered_bo,
                                               structures.atom_types,
                                               force_field.cutoff2,
                                               force_field.body4_params_mask)
    new_max_filter4_size = jnp.max(filter4_count)
    new_counts['filter4_size'] = new_max_filter4_size

    max_filter4_size = prev_counts['filter4_size']
    did_overflow = did_overflow | (new_max_filter4_size > max_filter4_size)

    filter4 = vec_create_filter4(filter3.idx,
                                 filtered_close_idx,
                                 filtered_bo,
                                 structures.atom_types,
                                 force_field.cutoff2,
                                 force_field.body4_params_mask,
                                 max_filter4_size)
    # handle hbond if it is available in the given systems
    if use_hbond:
      types_hb = FF_types_hb[structures.atom_types]
      hb_h_mask = jnp.array(types_hb == 1) & (structures.atom_types != -1)
      hb_nbr_mask = jnp.array(types_hb == 2) & (structures.atom_types != -1)
      new_max_hb_h_size = jnp.max(jnp.count_nonzero(hb_h_mask,axis=1), initial=1)
      new_counts['hbond_h_size'] = new_max_hb_h_size

      max_hb_h_size = prev_counts['hbond_h_size']
      did_overflow = did_overflow | (new_max_hb_h_size > max_hb_h_size)

      mapped_argwhere = jax.vmap(lambda vec:
                                 jnp.argwhere(vec, size=max_hb_h_size, fill_value=-1))

      hb_h_inds = mapped_argwhere(hb_h_mask).reshape(batch_size, -1)
      batch_inds = jnp.arange(batch_size).reshape(-1,1)
      h_nbr_inds = filtered_close_idx[batch_inds, hb_h_inds]
      h_nbr_pots = filtered_bo[batch_inds, hb_h_inds] * (hb_h_inds[..., jnp.newaxis] != -1)
      h_long_nbr_inds = far_nbr_inds[batch_inds,hb_h_inds]
      h_long_nbr_dists = far_dists[batch_inds,hb_h_inds] * (hb_h_inds[..., jnp.newaxis] != -1)

      far_close_sizes = vec_calculate_hb_far_close_sizes(h_nbr_inds,h_nbr_pots,
                                       h_long_nbr_inds,h_long_nbr_dists,
                                       hb_nbr_mask)

      new_max_hb_far_size = jnp.max(far_close_sizes[:,0], initial=1)
      new_max_hb_close_size = jnp.max(far_close_sizes[:,1], initial=1)
      new_counts['hbond_filter_far_size'] = new_max_hb_far_size
      new_counts['hbond_filter_close_size'] = new_max_hb_close_size

      max_hb_far_size = prev_counts['hbond_filter_far_size']
      max_hb_close_size = prev_counts['hbond_filter_close_size']
      did_overflow = did_overflow | (new_max_hb_far_size > max_hb_far_size)
      did_overflow = did_overflow | (new_max_hb_close_size > max_hb_close_size)

      [filter_hb_far,
       filter_hb_close] = vec_create_hb_far_close_filters(h_nbr_inds,
                                                        h_nbr_pots,
                                                        h_long_nbr_inds,
                                                        h_long_nbr_dists,
                                                        hb_nbr_mask,
                                                        max_hb_close_size,
                                                        max_hb_far_size)
      hb_sizes = vec_calculate_hb_count(hb_h_inds,
                                               filtered_close_idx,
                                               filter_hb_close.idx,
                                               far_nbr_inds,
                                               filter_hb_far.idx,
                                               structures.atom_types,
                                               force_field.hb_params_mask)
      new_max_hb_size = jnp.max(hb_sizes, initial=1)
      new_counts['hbond_size'] = new_max_hb_size

      max_hb_size = prev_counts['hbond_size']
      did_overflow = did_overflow | (new_max_hb_size > max_hb_size)

      hb_filter = vec_create_hb_filter(hb_h_inds,
                                      filtered_close_idx,
                                      filter_hb_close.idx,
                                      far_nbr_inds,
                                      filter_hb_far.idx,
                                      structures.atom_types,
                                      force_field.hb_params_mask,
                                      max_hb_size)
    else:
      filter_hb_close = None
      filter_hb_far = None
      hb_filter = None
    filter34 = None
    overflow_flags = jnp.ones(shape=batch_size, dtype=jnp.bool_) & did_overflow
    inter_lists = ReaxFFNeighborLists(close_nbr_list,far_nbr_list,filter2,
                                      filter3,filter34,filter4,
                                      filter_hb_close,filter_hb_far,hb_filter,
                                      overflow_flags)

    return inter_lists,new_counts

  return batched_allocate
