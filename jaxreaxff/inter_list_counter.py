"""
This file contains the functionality to calulate the interaction list
sizes before actually creating them.

As jax expects the array shaped to be fixed throughout the computation,
we calculate the array shapes here using regular numpy with multiprocessing

Author: Mehmet Cagri Kaymak

"""
import numpy as onp
onp.seterr(all="ignore")
from functools import partial
import numba as nb

def pool_handler_for_inter_list_count(systems, force_field, 
                                      pool, chunksize=1,
                                      close_cutoff=5.0,
                                      far_cutoff=10.0):
  '''
  Calculate the interaction list sizes for the provided geometries
  using the provided pool and chunk size
  '''
  modified_create_inter_lists = partial(calculate_inter_list_sizes,
                                        force_field=force_field,
                                        close_cutoff=close_cutoff,
                                        far_cutoff=far_cutoff)
  all_sizes = pool.map(modified_create_inter_lists, systems,
                       chunksize=chunksize)
  return all_sizes

def create_dense_nbr_list(dist_mats, cutoff=10.0):
  '''
  Based on the provided distance matrices, create the neighbor lists
  using regular numpy (on CPU)
  dist_mats: [N, N, # shifts]
  '''
  N,N,shifts = dist_mats.shape
  mask = (dist_mats < cutoff) & (dist_mats > 0)
  counts = onp.sum(mask.astype(onp.int32), axis=(1,2))
  max_c = onp.max(counts,initial=0)
  # initialize with -1
  res_inds = onp.zeros(shape=(N,max_c),dtype=onp.int32) - 1
  res_shifts = onp.zeros(shape=(N,max_c),dtype=onp.int32)
  dists = onp.zeros(shape=(N,max_c),dtype=onp.float32)
  # create the dense neighbor lists atom by atom
  for i in range(N):
    inds = onp.argwhere(mask[i])
    if inds.size > 0:
      res_inds[i][:counts[i]] = inds[:,0]
      res_shifts[i][:counts[i]] = inds[:,1]
      dists[i][:counts[i]] = dist_mats[i, inds[:,0], inds[:,1]]
  return res_inds, res_shifts, dists

def filter_dense(nbr_inds,
                 values,
                 threshold = 0.0):
  '''
  Given the neighbor lists (N X K)
  Only keep the indices that have value above the threshold
  Typically, values represent bond strength (or bond order)
  '''
  N,M = nbr_inds.shape
  mask = (values > threshold)
  counts = onp.count_nonzero(mask, axis=1)
  max_c = onp.max(counts,initial=0)
  res_inds = onp.zeros(shape=(N,max_c),dtype=onp.int32) - 1
  filt_vals = onp.zeros(shape=(N,max_c),dtype=onp.float32)
  for i in range(N):
    inds = onp.argwhere(mask[i]).flatten()
    if inds.size != 0:
      res_inds[i][:counts[i]] = nbr_inds[i][inds]
      filt_vals[i][:counts[i]] = values[i][inds]
  return res_inds, filt_vals

@nb.njit
def calculate_filter3_size(atom_types,
                          nbr_inds,
                          bo,
                          cutoff2,
                          body3_mask):
  '''
  Given the filtered neighbor list, bond order values, bond order cutoff
  and 3 body mask based on the available force field parameters,
  count the number of 3 body interactions
  '''
  N = atom_types.shape[0]
  count = 0
  for c in range(N):
    c_t = atom_types[c]
    for i1 in range(len(nbr_inds[c])):
      nbr1 = nbr_inds[c][i1]
      i1_t = atom_types[nbr1]
      if bo[c][i1] <= cutoff2 or i1 == -1:
        continue
      for i2 in range(i1+1,len(nbr_inds[c])):
        nbr2 = nbr_inds[c][i2]
        i2_t = atom_types[nbr2]
        if (bo[c][i2] > cutoff2 and i2 != -1 and body3_mask[i1_t, c_t, i2_t]
            and bo[c][i1] * bo[c][i2] > 0.00001):
          count += 1
  return count

@nb.njit
def calculate_filter4_size(atom_types,
                          nbr_inds,
                          bo,
                          cutoff2,
                          body4_mask):
  '''
  Given the filtered neighbor list, bond order values, bond order cutoff
  and 4 body mask based on the available force field parameters,
  count the number of 4 body interactions
  '''
  N = atom_types.size
  count = 0
  for c1 in range(N):
    c1_t = atom_types[c1]
    for i1,c2 in enumerate(nbr_inds[c1]):
      c2_t = atom_types[c2]
      if bo[c1][i1] <= cutoff2 or i1 == -1 or c2 < c1:
        continue
      for i2,nbr2 in enumerate(nbr_inds[c2]):
        i2_t = atom_types[nbr2]
        if bo[c2][i2] <= cutoff2 or i2 == -1:
          continue
        for i3,nbr1 in enumerate(nbr_inds[c1]):
          i3_t = atom_types[nbr1]
          if (bo[c1][i3] > cutoff2 and i3 != -1
              and body4_mask[i3_t, c1_t, c2_t, i2_t]
              and (bo[c1][i1] * bo[c2][i2] * bo[c1][i3] > cutoff2)
              and c1 != nbr2 and c2 != nbr1 and nbr1 != nbr2):
            # 4-body: nbr1, c1, c2, nbr2
            count += 1
  return count

def calculate_hbond_dense_size(atom_types,
                               nbr_inds,
                               nbr_values,
                               cutoff,
                               nphb):
  '''
  Calculate the number of neighbors available for hbond for each H atom
  Can work for both long range neighbors (the cutoff is typically 7.5 A)
  or short range bonded neighbors
  Returns X,Y where X is the number H atoms and Y is the max number of nbrs
  suitable to form H-bond
  '''
  FF_types_hb = nphb
  types_hb = FF_types_hb[atom_types]
  # mask for hydrogen type
  hydrogen_mask = (types_hb == 1) & (atom_types != -1)
  # atoms that can be involved in h-bond
  hydrogen_pair_mask = (types_hb == 2) & (atom_types != -1)

  h_nbr_inds = nbr_inds[hydrogen_mask]
  h_nbr_vals = nbr_values[hydrogen_mask]
  h_nbr_mask = hydrogen_pair_mask[h_nbr_inds]
  h_val_mask = (h_nbr_vals < cutoff) & (h_nbr_inds != -1) & (h_nbr_vals != 0.0)

  counts = onp.count_nonzero(h_val_mask & h_nbr_mask, axis=1)
  max_c = onp.max(counts,initial=0)
  return (onp.sum(hydrogen_mask), max_c)

@nb.njit
def calculate_hbond_size(atom_types,
                         close_nbr_inds,
                         bo,
                         far_nbr_inds,
                         far_nbr_dists,
                         dist_cutoff,
                         bo_cutoff,
                         nphb,
                         hbond_mask):
  '''
  Calculate the size of the final h-bond interaction list that is created using
  both long range and short range interaction lists
  '''
  N = atom_types.shape[0]
  count = 0
  # a1 is the H ATOM (center)
  for a1 in range(N):
    type1 = atom_types[a1]
    if type1 == -1:
      continue
    # i2 is the long range nbr(acceptor)
    for i2 in range(far_nbr_inds[a1].size):
      a2 = far_nbr_inds[a1][i2]
      dist = far_nbr_dists[a1][i2]
      type2 = atom_types[a2]
      ihhb1 = nphb[type1]
      ihhb2 = nphb[type2]
      if type2 == -1 or ihhb1 != 1 or ihhb2 != 2 or dist > dist_cutoff:
        continue
      #i3 is the short range nbr (donor)
      for i3 in range(close_nbr_inds[a1].size):
        a3 = close_nbr_inds[a1][i3]
        type3 = atom_types[a3]
        ihhb3 = nphb[type3]
        if (ihhb3 == 2 and bo[a1][i3] > bo_cutoff and
            a3 != a2 and hbond_mask[type3,type1,type2]):
          count += 1
  return count

def create_distance_matrix(shift_pos, tiled_atom_pos1, orth_matrix):
  '''
  Create a distance matrix for selected box shift and box matrix
  '''
  tiled_atom_pos1_trans = tiled_atom_pos1.swapaxes(0,1)
  shifted_tiled_atom_pos1_trans = tiled_atom_pos1_trans - shift_pos
  diff = tiled_atom_pos1 - shifted_tiled_atom_pos1_trans
  distance_matrix = onp.sqrt(onp.square(diff).sum(axis=2))
  return distance_matrix

def create_distance_matrices(atom_positions, orth_matrix, all_shift_comb, mask):
  '''
  Create distance matrices for all the box shift options for a given box
  '''
  all_shift_comb = onp.array(all_shift_comb,dtype=onp.float32)
  num_atoms = atom_positions.shape[0]
  atom_pos = atom_positions.reshape((num_atoms,1,3))
  shift_pos = onp.dot(orth_matrix,all_shift_comb.transpose()).transpose()
  tiled_atom_pos1 = onp.tile(atom_pos,(1,num_atoms,1))
  # canculate the distance matrices
  distance_matrices = onp.stack([create_distance_matrix(pos,
                                                        tiled_atom_pos1,
                                                        orth_matrix)
                                 for pos in shift_pos])
  # dist mat: [N, N, # shifts]
  distance_matrices = distance_matrices.swapaxes(0,2)
  # create the mask
  mask = mask.reshape(-1,1)
  mask_2d = onp.dot(mask,mask.transpose())
  # zero out the masked distances
  distance_matrices = distance_matrices * mask_2d[..., onp.newaxis]
  return distance_matrices


def calculate_inter_list_sizes(structure,
                              force_field,
                              close_cutoff=5.0,
                              far_cutoff=10.0):
  '''
  Calculate the all ineraction list sizes for a given structure and force field
  '''
  atom_AN = structure.atomic_nums
  atom_types = structure.atom_types
  atom_mask = atom_types != -1
  # dictionary to hold the final values
  size_dict = {}
  # default cutoffs
  hbond_dist_cutff = 7.5
  hbond_bo_cutoff = 0.01
  dist_mats = create_distance_matrices(structure.positions,
                                       structure.orth_matrix,
                                       structure.periodic_image_shifts,
                                       atom_mask
                                       )
  size_dict["num_atoms"] = len(atom_mask)
  size_dict["periodic_image_count"] = len(structure.periodic_image_shifts)

  [far_nbr_inds,
  far_nbr_shifts,
  far_nbr_dists] = create_dense_nbr_list(dist_mats, cutoff=far_cutoff)
  size_dict["far_nbr_size"] = far_nbr_inds.shape[1]

  [close_nbr_inds,
  close_nbr_shifts,
  close_nbr_dists] = create_dense_nbr_list(dist_mats, cutoff=close_cutoff)
  size_dict["close_nbr_size"] = close_nbr_inds.shape[1]


  atomic_num1 = atom_AN.reshape(-1, 1)
  atomic_num2 = atom_AN[close_nbr_inds]
  # For the special terms between C and O atoms
  # O: 8, C:6
  triple_bond1 = onp.logical_and(atomic_num1 == 8, atomic_num2 == 6)
  triple_bond2 = onp.logical_and(atomic_num1 == 6, atomic_num2 == 8)
  triple_bond = onp.logical_or(triple_bond1, triple_bond2)

  close_nbr_mask = (atom_mask.reshape(-1, 1)
                    & atom_mask[close_nbr_inds]
                    & (close_nbr_inds != -1))

  # calculate the bond strength
  [_, bo, _, _,_] = calculate_covbon_pot(close_nbr_inds,
                       close_nbr_dists,
                       close_nbr_mask,
                       atom_types,
                       triple_bond,
                       force_field)

  # filter out the weak bonds
  [filt2_inds,
  filt2_bo] = filter_dense(close_nbr_inds,
                           bo,
                           threshold = 0.0)

  size_dict["filter2_size"] = filt2_inds.shape[1]

  # calculate the size of 3-body interaction list
  size_filter3 = calculate_filter3_size(atom_types,
                                      filt2_inds,
                                      filt2_bo,
                                      force_field.cutoff2,
                                      force_field.body3_params_mask)
  size_dict["filter3_size"] = size_filter3
  # calculate the size of 4-body interaction list
  size_filter4 = calculate_filter4_size(atom_types,
                                        filt2_inds,
                                        filt2_bo,
                                        force_field.cutoff2,
                                        force_field.body4_params_mask)
  size_dict["filter4_size"] = size_filter4

  # calculate the size of h-bond interaction list
  size_hbond = calculate_hbond_size(atom_types,
                                   filt2_inds,
                                   filt2_bo,
                                   far_nbr_inds,
                                   far_nbr_dists,
                                   hbond_dist_cutff,
                                   hbond_bo_cutoff,
                                   force_field.nphb,
                                   force_field.hb_params_mask)

  size_dict["hbond_size"] = size_hbond

  # calculate the size of far neighbor sub list for h-bond creation
  (far_h_size,
   far_acc_size) = calculate_hbond_dense_size(atom_types,
                                                 far_nbr_inds,
                                                 far_nbr_dists,
                                                 hbond_dist_cutff,
                                                 force_field.nphb)
  size_dict["hbond_h_size"] = far_h_size
  size_dict["hbond_filter_far_size"] = far_acc_size

  # calculate the size of close neighbor sub list for h-bond creation
  # multiply BO and the cutoff by -1 since we check if value < cutoff
  (close_h_size,
   close_donor_size) = calculate_hbond_dense_size(atom_types,
                                                 filt2_inds,
                                                 filt2_bo * -1,
                                                 hbond_bo_cutoff * -1,
                                                 force_field.nphb)

  size_dict["hbond_filter_close_size"] = close_donor_size


  return size_dict

def calculate_covbon_pot(nbr_inds,
                         nbr_dist,
                         nbr_mask,
                         species,
                         triple_bond,
                         force_field):
  '''
  Calculate the bond order for the covolent bonding using regular numpy
  This is required to be able to use multiprocessing library since JAX.numpy
  does not work well in that settings
  '''
  #TODO: This leads to code dublication since the same code exists for both
  # jax and regular numpy, remove the dublication by being able to select the
  # numpy backend
  N = species.size
  nbr_mask = nbr_mask & (nbr_dist > 0)

  neigh_types = species[nbr_inds]
  atom_inds = onp.arange(N).reshape(-1, 1)
  species = species.reshape(-1, 1)
  # since we store the close nbr list full, we later divide the summation by 2
  # to compansate double counting, the self bonds are not double counted
  # so they will be multipled by 0.5 as expected
  symm = 1.0
  my_rob1 = force_field.rob1[neigh_types, species]
  my_rob2 = force_field.rob2[neigh_types, species]
  my_rob3 = force_field.rob3[neigh_types, species]
  my_ptp = force_field.ptp[neigh_types, species]
  my_pdp = force_field.pdp[neigh_types, species]
  my_popi = force_field.popi[neigh_types, species]
  my_pdo = force_field.pdo[neigh_types, species]
  my_bop1 = force_field.bop1[neigh_types, species]
  my_bop2 = force_field.bop2[neigh_types, species]
  my_de1 = force_field.de1[neigh_types, species]
  my_de2 = force_field.de2[neigh_types, species]
  my_de3 = force_field.de3[neigh_types, species]
  my_psp = force_field.psp[neigh_types, species]
  my_psi = force_field.psi[neigh_types, species]

  rhulp = onp.where(my_rob1 > 0, nbr_dist/(my_rob1 + 1e-15), 0.0)
  rhulp2 = onp.where(my_rob2 > 0, nbr_dist/(my_rob2 + 1e-15), 0.0)
  rhulp3 = onp.where(my_rob3 > 0, nbr_dist/(my_rob3 + 1e-15), 0.0)

  rh2p = rhulp2 ** my_ptp
  ehulpp = onp.exp(my_pdp * rh2p)

  rh2pp = rhulp3 ** my_popi
  ehulppp = onp.exp(my_pdo * rh2pp)

  rh2 = rhulp ** my_bop2
  ehulp = (1 + force_field.cutoff) * onp.exp(my_bop1 * rh2)

  mask1 = (my_rob1 > 0) & nbr_mask
  mask2 = (my_rob2 > 0) & nbr_mask
  mask3 = (my_rob3 > 0) & nbr_mask
  full_mask = mask1 | mask2 | mask3

  ehulp = ehulp * mask1
  ehulpp = ehulpp * mask2
  ehulppp = ehulppp * mask3

  bor = ehulp + ehulpp + ehulppp
  bopi = ehulpp
  bopi2 = ehulppp
  bo = bor - force_field.cutoff
  bo = onp.where(bo <= 0, 0.0, bo)
  abo = onp.sum(bo, axis=1)

  bo, bopi, bopi2 = calculate_boncor_pot(nbr_inds,
                                         nbr_mask,
                                         species.flatten(),
                                         bo, bopi, bopi2, abo,
                                         force_field)

  abo = onp.sum(bo * nbr_mask, axis=1)

  bosia = bo - bopi - bopi2
  bosia = onp.clip(bosia, a_min=0, a_max=9999)
  de1h = symm * my_de1
  de2h = symm * my_de2
  de3h = symm * my_de3

  bopo1 = onp.where(bosia != 0, bosia ** my_psp, 0)

  exphu1 = onp.exp(my_psi * (1.0 - bopo1))
  ebh = -de1h * bosia * exphu1 - de2h * bopi - de3h * bopi2
  ebh = onp.where(bo <= 0, 0.0, ebh)
  # Stabilisation terminal triple bond in CO
  ba = (bo - 2.5) * (bo - 2.5)
  exphu = onp.exp(-force_field.trip_stab8 * ba)

  abo_j2 = abo[nbr_inds]
  abo_j1 = abo[atom_inds]

  obo_a = abo_j1 - bo
  obo_b = abo_j2 - bo

  exphua1 = onp.exp(-force_field.trip_stab4*obo_a)
  exphub1 = onp.exp(-force_field.trip_stab4*obo_b)

  my_aval = force_field.aval[species] + force_field.aval[neigh_types]

  triple_bond = onp.where(bo < 1.0, 0.0, triple_bond)
  ovoab = abo_j1 + abo_j2 - my_aval
  exphuov = onp.exp(force_field.trip_stab5 * ovoab)

  hulpov = 1.0/(1.0+25.0*exphuov)

  estriph = force_field.trip_stab11*exphu*hulpov*(exphua1+exphub1)

  eb = (ebh + estriph * triple_bond)
  eb = eb * full_mask

  cov_pot = onp.sum(eb) / 2.0

  return [cov_pot, bo, bopi, bopi2, abo]

def calculate_boncor_pot(nbr_inds,
                         nbr_mask,
                         species,
                         bo,
                         bopi,
                         bopi2,
                         abo,
                         force_field):
  '''
  Calculate the bond order correction for the covolent bonding using regular
  numpy. This is required to be able to use multiprocessing library since
  JAX.numpy does not work well in that settings
  '''

  neigh_types = species[nbr_inds]
  species = species.reshape(-1, 1)

  abo_j2 = abo[nbr_inds]
  abo_j1 = abo.reshape(-1, 1)

  aval_j2 = force_field.aval[neigh_types]
  aval_j1 = force_field.aval[species]

  vp131 = onp.sqrt(force_field.bo131[species] * force_field.bo131[neigh_types])
  vp132 = onp.sqrt(force_field.bo132[species] * force_field.bo132[neigh_types])
  vp133 = onp.sqrt(force_field.bo133[species] * force_field.bo133[neigh_types])

  my_ovc = force_field.ovc[neigh_types, species]

  ov_j1 = abo_j1 - aval_j1
  ov_j2 = abo_j2 - aval_j2

  exp11 = onp.exp(-force_field.over_coord1*ov_j1)
  exp21 = onp.exp(-force_field.over_coord1*ov_j2)
  exphu1 = onp.exp(-force_field.over_coord2*ov_j1)
  exphu2 = onp.exp(-force_field.over_coord2*ov_j2)
  exphu12 = (exphu1+exphu2)

  ovcor = -(1.0/force_field.over_coord2) * onp.log(0.50*exphu12)
  huli = aval_j1+exp11+exp21
  hulj = aval_j2+exp11+exp21

  corr1 = huli/(huli+ovcor)
  corr2 = hulj/(hulj+ovcor)
  corrtot = 0.50*(corr1+corr2)

  corrtot = onp.where(my_ovc > 0.001, corrtot, 1.0)

  my_v13cor = force_field.v13cor[neigh_types, species]

  # update vval3 based on amas value
  vval3 = onp.where(force_field.amas < 21.0,
                   force_field.valf,
                   force_field.vval3)

  vval3_j1 = vval3[species]
  vval3_j2 = vval3[neigh_types]
  ov_j11 = abo_j1 - vval3_j1
  ov_j22 = abo_j2 - vval3_j2
  cor1 = vp131 * bo * bo - ov_j11
  cor2 = vp131 * bo * bo - ov_j22

  exphu3 = onp.exp(-vp132 * cor1 + vp133)
  exphu4 = onp.exp(-vp132 * cor2 + vp133)
  bocor1 = 1.0/(1.0+exphu3)
  bocor2 = 1.0/(1.0+exphu4)

  bocor1 = onp.where(my_v13cor > 0.001, bocor1, 1.0)
  bocor2 = onp.where(my_v13cor > 0.001, bocor2, 1.0)

  bo = bo * corrtot * bocor1 * bocor2
  threshold = 0.0 # fortran threshold: 1e-10
  mask = nbr_mask & (bo > threshold)
  bo = bo * mask
  corrtot2 = corrtot*corrtot
  bopi = bopi*corrtot2*bocor1*bocor2
  bopi2 = bopi2*corrtot2*bocor1*bocor2

  bopi = bopi * mask
  bopi2 = bopi2 * mask

  return bo, bopi, bopi2


