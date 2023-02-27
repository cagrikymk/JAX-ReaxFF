#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of ReaxFF potential using JAX
Ported from the standalone ReaxFF(Fortran)

Author: Mehmet Cagri Kaymak
"""

from jaxreaxff.structure import Structure
from jaxreaxff.forcefield import TYPE,c1c,dgrrdn,rdndgr,preprocess_force_field

import numpy as onp
import jax.numpy as np
import jax

CLIP_MIN = -35
CLIP_MAX = 35
# it fixes nan values issue, from: https://github.com/google/jax/issues/1052
def vectorized_cond(pred, true_fun, false_fun, operand):
  # true_fun and false_fun must act elementwise (i.e. be vectorized)
  #how to use: grad(lambda x: vectorized_cond(x > 0.5, lambda x: np.arctan2(x, x), lambda x: 0., x))(0.)
  true_op = np.where(pred, operand, 0)
  false_op = np.where(pred, 0, operand)
  return np.where(pred, true_fun(true_op), false_fun(false_op))


#https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
@jax.custom_jvp
def safe_sqrt(x):
  return np.sqrt(x)

@safe_sqrt.defjvp
def safe_sqrt_jvp(primals, tangents):
  x = primals[0]
  x_dot = tangents[0]
  #print(x[0])
  primal_out = safe_sqrt(x)
  tangent_out = 0.5 * x_dot / np.where(x > 0, primal_out, np.inf)
  return primal_out, tangent_out


def calculate_total_energy_single(flattened_force_field, flattened_non_dif_params, system):
    #body_2_distances = Structure.calculate_2_body_distances(system.atom_positions,system.box_size, system.global_body_2_inter_list,system.global_body_2_inter_list_mask)
    #body_3_angles = Structure.calculate_3_body_angles(system.atom_positions,system.box_size,system.global_body_2_inter_list,system.global_body_3_inter_list,system.global_body_3_inter_list_mask, system.global_body_3_inter_shift_map)
    #body_4_angles = Structure.calculate_body_4_angles_new(system.atom_positions,system.box_size,system.global_body_4_inter_list,system.global_body_4_inter_shift,system.global_body_4_inter_list_mask)
    #dist_matrices = Structure.create_distance_matrices(system.is_periodic, system.atom_positions,system.box_size)


    return calculate_total_energy(flattened_force_field,flattened_non_dif_params,
                                                    system.atom_types,system.atom_mask, system.total_charge, system.local_body_2_neigh_list, system.distance_matrices,
                                                    system.global_body_2_inter_list,system.global_body_2_inter_list_mask,system.triple_bond_body_2_mask,system.global_body_2_distances,
                                                    system.global_body_3_inter_list,system.global_body_3_inter_list_mask,system.global_body_3_angles,
                                                    system.global_body_4_inter_list,system.global_body_4_inter_list_mask,system.global_body_4_angles,
                                                    system.global_hbond_inter_list,system.global_hbond_inter_list_mask,system.global_hbond_angles_and_dist)

def calculate_total_energy_multi(flattened_force_field, flattened_non_dif_params,
               structured_training_data,
               list_all_atom_pos,
               list_all_type,list_all_mask,
               list_all_total_charge,
               list_all_body_2_neigh_list,
               list_all_dist_mat,
               list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
               list_all_body_3_list,list_all_body_3_map,list_all_body_3_angles,
               list_all_body_4_list,list_all_body_4_map,list_all_body_4_angles,
               list_all_hbond_inter_list,list_all_hbond_inter_list_mask,list_all_angles_and_dist,
               list_bond_rest,list_angle_rest,list_torsion_rest):

    flattened_force_field = preprocess_force_field(flattened_force_field,flattened_non_dif_params)
    list_counts = [len(l) for l in list_all_type]
    total_num_systems = sum(list_counts)
    pot_func = jax.vmap(calculate_total_energy,in_axes=(None,None,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))
    pot_func_pmap = jax.vmap(calculate_total_energy,in_axes=(None,None,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))
    all_pots = np.zeros(total_num_systems,dtype=TYPE)
    all_bo = []
    all_tors = []
    all_val = []
    start = 0
    end = 0
    for i in range(len(list_all_type)):

        start = end
        end = start + list_counts[i]

        pots,charges = pot_func_pmap(flattened_force_field,flattened_non_dif_params,
                                      list_all_type[i],list_all_mask[i],
                                      list_all_total_charge[i],
                                      list_all_body_2_neigh_list[i],
                                      list_all_dist_mat[i],
                                      list_all_body_2_list[i],list_all_body_2_map[i],list_all_body_2_trip_mask[i],list_all_body_2_distances[i],
                                      list_all_body_3_list[i],list_all_body_3_map[i],list_all_body_3_angles[i],
                                      list_all_body_4_list[i],list_all_body_4_map[i],list_all_body_4_angles[i],
                                      list_all_hbond_inter_list[i],list_all_hbond_inter_list_mask[i],list_all_angles_and_dist[i])

        bond_rest_pot = jax.vmap(calculate_bond_restraint_energy)(list_all_atom_pos[i], list_bond_rest[i])
        angle_rest_pot = jax.vmap(calculate_angle_restraint_energy)(list_all_atom_pos[i], list_angle_rest[i])
        torsion_rest_pot = jax.vmap(calculate_torsion_restraint_energy)(list_all_atom_pos[i], list_torsion_rest[i])

        all_pots = jax.ops.index_update(all_pots, jax.ops.index[start:end], pots)
        #all_pots = jax.ops.index_update(all_pots, jax.ops.index[start:end], pots + bond_rest_pot + angle_rest_pot + torsion_rest_pot)
        #all_bo.append(bo)


    return all_pots #,all_bo


def calculate_total_energy(flattened_force_field,flattened_non_dif_params, types,atom_mask,total_charge,
                                    local_body_2_neigh_list,
                                    dist_matrices,
                                    global_body_2_inter_list,global_body_2_inter_list_mask,triple_bond_body_2_mask,global_body_2_distances,
                                    global_body_3_inter_list,global_body_3_inter_list_mask,global_body_3_angles,
                                    global_body_4_inter_list,global_body_4_inter_list_mask,global_body_4_angles,
                                    global_hbond_inter_list,global_hbond_inter_list_mask,hbond_angles_and_dist
                                 ):
    cou_pot = 0
    vdw_pot = 0
    charge_pot = 0
    cov_pot = 0
    lone_pot = 0
    val_pot = 0
    total_penalty = 0
    total_conj = 0
    overunder_pot = 0
    tor_conj = 0
    torsion_pot = 0
    h_pot = 0

    #return (cou_pot + vdw_pot + charge_pot +
    #   cov_pot + lone_pot + val_pot + total_penalty + total_conj + overunder_pot + tor_conj + torsion_pot)
    # shared accross charge calc, coulomb, and vdw
    dist_matrices = dist_matrices + 1e-15 # for numerical issues
    tapering_matrices = taper(dist_matrices, 0.0, 10.0)
    tapering_matrices = np.where((dist_matrices > 10.0) | (dist_matrices < 0.001), 0.0, tapering_matrices)

    # shared accross charge calc and coulomb
    my_gamma = flattened_force_field[0][types] # gamma
    gamma_mat = np.sqrt(my_gamma.reshape(-1,1).dot(my_gamma.reshape(1,-1)))

    hulp1_mat = dist_matrices ** 3 + (1/gamma_mat**3)

    charges = calculate_eem_charges(types,
                                  atom_mask,
                                  total_charge,
                                  hulp1_mat,
                                  tapering_matrices,
                                  flattened_force_field[0], # gamma
                                  flattened_force_field[1], #idempotential
                                  flattened_force_field[2] #electronegativity
                                  )

    cou_pot = calculate_coulomb_pot(types,
                                  hulp1_mat,
                                  tapering_matrices,
                                  charges[:-1],
                                  flattened_force_field[0]) # gamma


    charge_pot = calculate_charge_energy(types,
                                       charges[:-1] * atom_mask,
                                       flattened_force_field[1], #idempotential
                                       flattened_force_field[2]) #electronegativity



    vdw_pot = calculate_vdw_pot(types,
                              atom_mask,
                              dist_matrices,
                              tapering_matrices,
                              flattened_force_field[3],  # p1co
                              flattened_force_field[4],  # p2co
                              flattened_force_field[5],  # p3co
                              flattened_force_field[6],  # vop
                              flattened_force_field[7]  # vdw shielding
                              )


    [cov_pot, bo, bopi,bopi2, abo] = calculate_covbon_pot(types,
                                               global_body_2_inter_list,
                                               global_body_2_inter_list_mask,
                                               global_body_2_distances,
                                               local_body_2_neigh_list,
                                               triple_bond_body_2_mask,
                                               *flattened_force_field[8:33],
                                               flattened_non_dif_params[2],
                                               flattened_non_dif_params[3],
                                               flattened_non_dif_params[4],
                                               flattened_non_dif_params[5],
                                               flattened_non_dif_params[6],
                                               flattened_non_dif_params[7],
                                               flattened_non_dif_params[8]
                                               )

    [lone_pot, vlp] = calculate_lonpar_pot(types,
                                           atom_mask,
                                           abo,
                                           flattened_force_field[26],
                                           flattened_force_field[34],
                                           flattened_force_field[54],
                                           flattened_force_field[55]
                                           )

    overunder_pot = calculate_ovcor_pot(types,
                                      flattened_non_dif_params[18],
                                      atom_mask,
                                      global_body_2_inter_list,
                                      global_body_2_inter_list_mask,
                                      local_body_2_neigh_list,
                                      bo,bopi, bopi2,abo, vlp,
                                      flattened_force_field[34],
                                      flattened_force_field[26],
                                      flattened_force_field[56],
                                      flattened_force_field[57],
                                      flattened_force_field[17],
                                      flattened_force_field[58],
                                      flattened_force_field[59],
                                      *flattened_force_field[60:66])

    [val_pot,total_penalty,total_conj] = calculate_valency_pot(types,
                                                          global_body_3_inter_list,
                                                          global_body_3_inter_list_mask,
                                                          global_body_3_angles,
                                                          local_body_2_neigh_list,
                                                          vlp,
                                                          bo, bopi, bopi2, abo,
                                                          flattened_force_field[26],
                                                          flattened_force_field[27],
                                                          *flattened_force_field[33:54],
                                                          flattened_non_dif_params[9],
                                                          flattened_non_dif_params[11])


    [torsion_pot, tor_conj] = calculate_torsion_pot(types,
                                                  global_body_4_inter_list,
                                                  global_body_4_inter_list_mask,
                                                  global_body_4_angles,
                                                  bo,bopi,abo,
                                                  flattened_force_field[33],
                                                  *flattened_force_field[66:75],
                                                  flattened_non_dif_params[11]
                                                  )


    h_pot = calculate_hb_pot(types,bo,global_hbond_inter_list,global_hbond_inter_list_mask,hbond_angles_and_dist, *flattened_force_field[87:91])

    #print(torsion_pot, tor_conj)
    #print(overunder_pot)
    #print(torsion_pot,tor_conj)
    #print(cou_pot , vdw_pot , charge_pot , cov_pot ,lone_pot , val_pot, total_penalty, total_conj,overunder_pot, tor_conj, torsion_pot,h_pot)
    return (cou_pot + vdw_pot + charge_pot +
         cov_pot + lone_pot + val_pot + total_penalty + total_conj + overunder_pot + tor_conj + torsion_pot + h_pot),charges[:-1]

def calculate_total_energy_for_minim(atom_positions,flattened_force_field,flattened_non_dif_params,all_shift_comb,
                                 orth_matrix, types,atom_mask,list_all_total_charge,
                                 local_body_2_neigh_list,global_body_2_inter_list,global_body_2_inter_list_mask,
                                 triple_bond_body_2_mask,global_body_3_inter_list,
                                 global_body_3_inter_list_mask,global_body_3_inter_shift_map,
                                 global_body_4_inter_list,global_body_4_inter_list_mask,global_body_4_inter_shift,
                                 global_hbond_inter_list,global_hbond_inter_list_mask,hbond_shift,
                                 bond_restraints,angle_restraints,torsion_restraints
                                 ):
    #flattened_force_field = use_selected_parameters(selected_params,selected_inds, flattened_force_field)
    #flattened_force_field = preprocess_force_field(flattened_force_field, flattened_non_dif_params)

    body_2_distances = Structure.calculate_2_body_distances(atom_positions,orth_matrix, global_body_2_inter_list,global_body_2_inter_list_mask)
    body_3_angles = Structure.calculate_3_body_angles(atom_positions,orth_matrix,global_body_2_inter_list,global_body_3_inter_list,global_body_3_inter_list_mask, global_body_3_inter_shift_map)
    body_4_angles = Structure.calculate_body_4_angles_new(atom_positions,orth_matrix,global_body_4_inter_list,global_body_4_inter_list_mask,global_body_4_inter_shift)
    dist_matrices = Structure.create_distance_matrices( atom_positions,orth_matrix, all_shift_comb)
    hbond_angles_and_dist = Structure.calculate_global_hbond_angles_and_dist(atom_positions,orth_matrix,global_hbond_inter_list,hbond_shift,global_hbond_inter_list_mask)

    #dist_matrices = 0
    reax_pot = 0.0
    bond_rest_pot = 0.0
    angle_rest_pot = 0.0
    torsion_rest_pot = 0.0
    reax_pot,_ = calculate_total_energy(flattened_force_field,flattened_non_dif_params, types,atom_mask,list_all_total_charge,
                                 local_body_2_neigh_list,dist_matrices,
                                 global_body_2_inter_list,global_body_2_inter_list_mask,triple_bond_body_2_mask, body_2_distances,
                                 global_body_3_inter_list,global_body_3_inter_list_mask,body_3_angles,
                                 global_body_4_inter_list,global_body_4_inter_list_mask,body_4_angles,
                                 global_hbond_inter_list,global_hbond_inter_list_mask,hbond_angles_and_dist
                                 ) # return only the energy
    bond_rest_pot = calculate_bond_restraint_energy(atom_positions, bond_restraints)
    angle_rest_pot = calculate_angle_restraint_energy(atom_positions, angle_restraints)
    torsion_rest_pot = calculate_torsion_restraint_energy(atom_positions, torsion_restraints)


    return reax_pot + bond_rest_pot + angle_rest_pot + torsion_rest_pot



def calculate_bond_restraint_energy(atom_positions, bond_restraints):
    atom_indices = bond_restraints[:,:2].astype(np.int32)
    forces = bond_restraints[:,2:4]
    target_dist = bond_restraints[:,4]
    bond_restrait_mask = bond_restraints[:,6]

    atoms_i = atom_indices[:,0]
    atoms_j = atom_indices[:,1]

    forces_1 = forces[:,0]
    forces_2 = forces[:,1]

    cur_dist = safe_sqrt(np.sum(np.power(atom_positions[atoms_i] - atom_positions[atoms_j],2),axis=1))
    #print(cur_dist)
    en_restraint = np.sum(bond_restrait_mask * forces_1 * (1.0 - np.exp(-forces_2 * (cur_dist - target_dist)**2)))
    #en_restraint = np.sum(bond_restrait_mask * forces_1 * (cur_dist - target_dist)**2)
    return en_restraint

def calculate_angle_restraint_energy(atom_positions, angle_restraints):
    atom_indices = angle_restraints[:,:3].astype(np.int32)
    forces = angle_restraints[:,3:5]
    target_angle = angle_restraints[:,5]
    angle_restraint_mask = angle_restraints[:,7]

    atoms_i = atom_indices[:,0]
    atoms_j = atom_indices[:,1]
    atoms_k = atom_indices[:,2]
    forces_1 = forces[:,0]
    forces_2 = forces[:,1]
    pos1 = atom_positions[atoms_i]
    pos2 = atom_positions[atoms_j]
    pos3 = atom_positions[atoms_k]

    cur_angle = jax.vmap(Structure.calculate_valence_angle)(pos1,pos2,pos3)
    cur_angle = cur_angle * rdndgr
    # to have periodicity, Ex. diff between 170 and -170 is 20 degree.
    cur_angle = np.where(cur_angle < 0.0, cur_angle+360.0, cur_angle)
    target_angle = np.where(target_angle < 0.0, target_angle+360.0, target_angle)
    diff = (cur_angle - target_angle) * dgrrdn
    en_restraint = np.sum(angle_restraint_mask * forces_1 * (1.0 - np.exp(-forces_2 * (diff)**2)))

    return en_restraint

def calculate_torsion_restraint_energy(atom_positions, torsion_restraints):
    atom_indices = torsion_restraints[:,:4].astype(np.int32)
    forces = torsion_restraints[:,4:6]
    target_torsion = torsion_restraints[:,6]
    torsion_restraint_mask = torsion_restraints[:,8]

    atoms_1 = atom_indices[:,0]
    atoms_2 = atom_indices[:,1]
    atoms_3 = atom_indices[:,2]
    atoms_4 = atom_indices[:,3]
    forces_1 = forces[:,0]
    forces_2 = forces[:,1]
    pos1 = atom_positions[atoms_1]
    pos2 = atom_positions[atoms_2]
    pos3 = atom_positions[atoms_3]
    pos4 = atom_positions[atoms_4]

    cur_torsion = jax.vmap(Structure.calculate_body_4_angles_single)(pos1,pos2,pos3,pos4)[:,-1].reshape(-1)
    # clip the values to not get NaN
    cur_torsion = np.clip(cur_torsion, -1.0 + 1e-7, 1.0 - 1e-7)
    cur_torsion = np.arccos(cur_torsion)
    cur_torsion = cur_torsion * rdndgr
    # to have periodicity, Ex. diff between 170 and -170 is 20 degree.
    cur_torsion = np.where(cur_torsion < 0.0, cur_torsion+360.0, cur_torsion)
    target_torsion = np.where(target_torsion < 0.0, target_torsion+360.0, target_torsion)
    diff = (cur_torsion - target_torsion) * dgrrdn
    en_restraint = np.sum(torsion_restraint_mask * forces_1 * (1.0 - np.exp(-forces_2 * (diff)**2)))

    return en_restraint
    
def calculate_torsion_pot(atom_types, global_body_4_inter_list, global_body_4_inter_list_mask,global_body_4_angles,
                              bo, bopi, abo,
                              valf,
                              v1, v2, v3,v4, vconj,
                              par_24,par_25, par_26,par_28, cutoff2):
    #[ind1,type1,ind2,type2,ind3,type3,ind4,type4,b1,b2,b3]
    # global_body_3_angles = coshd,coshe,sinhd,sinhe,arg
    #type_indices = global_body_4_inter_list[:,[1,3,5,7]]
    atom_indices = global_body_4_inter_list[:,[0,1,2,3]]
    bond_indices = global_body_4_inter_list[:,[4,5,6]]

    #type_indices = type_indices.transpose()
    atom_indices = atom_indices.transpose()
    bond_indices = bond_indices.transpose()
    type_indices = atom_types[atom_indices]


    num_atoms = len(atom_types)

    # for the masking: bopr=bo(i1)*bo(ibo2)*bo(ibo3) > cutoff condition is missing

    my_v1 = v1[type_indices[0],type_indices[1],type_indices[2],type_indices[3]]
    my_v2 = v2[type_indices[0],type_indices[1],type_indices[2],type_indices[3]]
    my_v3 = v3[type_indices[0],type_indices[1],type_indices[2],type_indices[3]]
    my_v4 = v4[type_indices[0],type_indices[1],type_indices[2],type_indices[3]]
    my_vconj = vconj[type_indices[0],type_indices[1],type_indices[2],type_indices[3]]

    exbo1 = abo - valf[atom_types]

    exbo1_2 = exbo1[atom_indices[1]] # second atom
    exbo2_3 = exbo1[atom_indices[2]] # third atom
    htovt = exbo1_2 + exbo2_3
    # max value for float32 is 10^38 = e^80
    expov = np.exp(np.clip(par_26 * htovt,CLIP_MIN, CLIP_MAX))
    expov2 = np.exp(np.clip(-par_25 * htovt,CLIP_MIN, CLIP_MAX))
    # for the numerical problems
    htov1 = 2.0 + expov2
    htov2 = 1.0 + expov + expov2
    etboadj = htov1 / htov2



    bo2t = 2.0 - bopi[bond_indices[1]] - etboadj
    bo2p = bo2t * bo2t

    bocor2 = np.exp(my_v4 * bo2p)
    #print('bocor2',4,1,2,3, bocor2[3,0,1,2])

    hsin =  global_body_4_angles[:,2] * global_body_4_angles[:,3] # sinhd * sinhe
    #print('hsin',4,1,2,3, hsin[3,0,1,2])
    arg = global_body_4_angles[:,4]
    arg2 = arg * arg
    #print('arg',4,1,2,3, arg[3,0,1,2])
    ethhulp = (0.5 * my_v1 * (1.0 + arg) + my_v2 * bocor2 * (1.0 - arg2) +
               my_v3 * (0.5 + 2.0*arg2*arg - 1.5*arg))
    #print('ethhulp',4,1,2,3, ethhulp[3,0,1,2])
    #print('my_v1',4,1,2,3, my_v1[3,0,1,2])
    #print('my_v2',4,1,2,3, my_v2[3,0,1,2])
    #print('my_v3',4,1,2,3, my_v3[3,0,1,2])

    boa = bo[bond_indices[0]]
    bob = bo[bond_indices[1]]
    boc = bo[bond_indices[2]]


    mult_bo_mask = np.where(boa * bob * boc > cutoff2, 1, 0)
    my_mask = global_body_4_inter_list_mask * mult_bo_mask

    boa = boa - cutoff2
    bob = bob - cutoff2
    boc = boc - cutoff2

    #print('par_24',par_24)

    bo_mask = np.where(boa > 0, 1, 0)
    bo_mask = np.where(bob > 0, bo_mask, 0)
    bo_mask = np.where(boc > 0, bo_mask, 0)
    my_mask = bo_mask * my_mask
    # ORIGINAL
    exphua = np.exp(-par_24 * boa)
    exphub = np.exp(-par_24 * bob)
    exphuc = np.exp(-par_24 * boc)
    #TORSION2013
    #exphua = np.exp(-2*par_24 * boa**2)
    #exphub = np.exp(-2*par_24 * bob**2)
    #exphuc = np.exp(-2*par_24 * boc**2)
    bocor4 = (1.0 - exphua) * (1.0 - exphub) * (1.0 - exphuc)
    #print('bocor4',4,1,2,3, bocor4[3,0,1,2])
    eth = hsin * ethhulp * bocor4



    #eth = np.nan_to_num(eth)
    eth = eth * my_mask
    '''
    for i1 in range(num_atoms):
        for i2 in range(num_atoms):
            for i3 in range(num_atoms):
                for i4 in range(num_atoms):
                    if eth[i1,i2,i3,i4] != 0:
                        print('eth',i1,i2,i3,i4,eth[i1,i2,i3,i4])
    '''
    tors_pot = np.sum(eth)

    # calculate conjugation pot
    ba=(boa-1.50)*(boa-1.50)
    bb=(bob-1.50)*(bob-1.50)
    bc=(boc-1.50)*(boc-1.50)

    exphua1=np.exp(-par_28*ba)
    exphub1=np.exp(-par_28*bb)
    exphuc1=np.exp(-par_28*bc)
    sbo=exphua1*exphub1*exphuc1
    sbo = sbo * my_mask

    arghu0=(arg2-1.0) * hsin # hsin = sinhd*sinhe
    ehulp = my_vconj*(arghu0+1.0)

    #ehulp = np.nan_to_num(ehulp)
    ecoh = ehulp*sbo
    conj_pot = np.sum(ecoh)

    return [tors_pot, conj_pot]


#@jax.jit
def calculate_ovcor_pot(atom_types,
                            name_to_index,
                            atom_mask,
                            global_body_2_inter_list,
                            global_body_2_inter_list_mask,
                            local_neigh_list,
                            bo,bopi, bopi2,abo, vlp,
                            stlp, aval, amas, vover,
                            de1,valp1, vovun,
                            par_6, par_7, par_9, par_10, par_32,par_33 ):

    num_atoms = len(atom_types)
    my_stlp = stlp[atom_types]
    my_aval = aval[atom_types]
    my_amas = amas[atom_types]
    my_valp1 = valp1[atom_types]
    my_vovun = vovun[atom_types]

    #print("abo",abo)
    #print("my_aval",my_aval)

    vlptemp = np.where(my_amas > 21.0, 0.50*(my_stlp-my_aval), vlp)
    dfvl = np.where(my_amas > 21.0, 0.0, 1.0)
    #  Calculate overcoordination energy
    #  Valency is corrected for lone pairs
    voptlp = 0.50*(my_stlp-my_aval)
    diffvlph = dfvl*(voptlp-vlptemp)
    #print("diffvlph",diffvlph)
    #Determine coordination neighboring atoms
    sumov = np.zeros(num_atoms)
    sumov2 = np.zeros(num_atoms)
    neigh_indices = local_neigh_list[:,:,0].astype(np.int32)
    bond_indices = local_neigh_list[:,:,1].astype(np.int32)

    neigh_types = atom_types[neigh_indices]

    part_1 = bopi[bond_indices] + bopi2[bond_indices]
    part_2 = abo[neigh_indices] - aval[neigh_types]     - diffvlph[neigh_indices]
    sumov = np.sum(part_1 * part_2, axis=1)
    #TODO: this part can be improved (learn more about numpy views)
    row_indices = np.arange(num_atoms)
    num_neigh = neigh_indices.shape[-1]
    row_indices = np.tile(row_indices.reshape(-1,1),(1, num_neigh))

    mult_vov_de1 = vover[atom_types] * de1[atom_types]
    selected_mult_vov_de1 = mult_vov_de1[row_indices,neigh_types]


    sumov2 = np.sum(selected_mult_vov_de1 * bo[bond_indices], axis=1)
    #print("bopi + bopi2",bopi + bopi2)
    #print("sumov",sumov)
    #print("sumov2",sumov2)
    # Gradient non issue fix
    exphu1 = np.exp(np.clip(par_32 * sumov,CLIP_MIN,CLIP_MAX))
    vho = 1.0 / (1.0+par_33*exphu1)
    diffvlp = diffvlph * vho

    vov1 = abo - my_aval - diffvlp
    # to solve the nan issue
    exphuo = np.exp(np.clip(my_vovun*vov1,CLIP_MIN,CLIP_MAX))
    hulpo = 1.0/(1.0+exphuo)

    hulpp = (1.0/(vov1+my_aval+1e-8))

    eah = sumov2*hulpp*hulpo*vov1

    ea = np.sum(eah * atom_mask)

    # Calculate undercoordination energy
    # Gradient non issue fix
    exphu2 = np.exp(np.clip(par_10*sumov,CLIP_MIN,CLIP_MAX))
    vuhu1=1.0+par_9*exphu2
    hulpu2=1.0/vuhu1

    exphu3=-np.exp(par_7*vov1)
    hulpu3=-(1.0+exphu3)

    dise2=my_valp1
    # Gradient non issue fix
    exphuu = np.exp(np.clip(-my_vovun*vov1,CLIP_MIN,CLIP_MAX))
    hulpu=1.0/(1.0+exphuu)
    eahu=dise2*hulpu*hulpu2*hulpu3

    eahu = np.where(my_valp1 < 0, 0, eahu)

    ea = ea + np.sum(eahu * atom_mask)

    #TODO: Calculate correction for C2 PART effecting (eplh) is missing (related to vpar(6))
    if 'C' in name_to_index:
        C_ind = name_to_index['C']
        par6_mask = np.where(np.abs(par_6) > 0.001, 1.0, 0.0)
        C_C_bonds_mask = (np.where(global_body_2_inter_list[:,1] == C_ind, 1.0, 0.0) *
                         np.where(global_body_2_inter_list[:,3] == C_ind, 1.0, 0.0) *
                         global_body_2_inter_list_mask *
                         par6_mask)
        vov4 = abo[global_body_2_inter_list[:,1]] - aval[global_body_2_inter_list[:,1]]
        vov3 = (bo - vov4 - 0.040 * (vov4 ** 4))
        vov3_mask = np.where(vov3 > 3.0, 1.0, 0.0)
        elph = par_6 * (vov3 -3.0)**2
        elph = elph * vov3_mask * C_C_bonds_mask
        ea = ea + np.sum(elph)

    return ea

def smooth_lone_pair_casting(number, p_lambda=0.9999, l1=-1.3, l2=-0.3, r1=0.3, r2=1.3):

    f_R = number - 1/2 - (1/np.pi)*(np.arctan(p_lambda *
                      np.sin(2*np.pi * number) /
                      (p_lambda * np.cos(2*np.pi * number) - 1)))

    f_L = number + 1/2 - (1/np.pi)*(np.arctan(p_lambda *
                      np.sin(2*np.pi * number) /
                      (p_lambda * np.cos(2*np.pi * number) - 1)))

    result = np.where(number < l1, f_L,
                    np.where(number < l2, f_L * taper(number,l1,l2),
                    np.where(number < r1, 0,
                    np.where(number <= r2, f_R * taper2(number,r1,r2),f_R))))

    return result


def calculate_lonpar_pot(atom_types,
                             atom_mask,
                             abo,
                             aval, stlp,
                             vlp1, par_16):
    #Determine number of lone pairs on atoms
    voptlp = 0.5 * (stlp[atom_types] - aval[atom_types])
    vund = abo - stlp[atom_types]
    vund_div2 = smooth_lone_pair_casting(vund/2.0) # (vund/2.0).astype(np.int32)
    #vund_div2 = (vund/2.0).astype(np.int32)
    vlph = 2.0 * vund_div2
    vlpex = vund - vlph

    vp16h = par_16 - 1.0
    expvlp = np.exp(-par_16 * (2.0 + vlpex) * (2.0 + vlpex))
    vlp = expvlp - vund_div2

    #Calculate lone pair energy
    diffvlp = voptlp-vlp
    exphu1 = np.exp(np.clip(-75.0*diffvlp,CLIP_MIN,CLIP_MAX))
    hulp1 = 1.0/(1.0+exphu1)
    elph = vlp1[atom_types] * diffvlp * hulp1

    elp = np.sum(elph * atom_mask)

    return [elp, vlp]


#@jax.jit
def calculate_valency_pot(atom_types, global_body_3_inter_list,
                              global_body_3_inter_list_mask,
                              global_body_3_angles,
                              body_2_local_list,
                              vlp,
                              bo,bopi, bopi2, abo,
                              aval,vval3,
                              valf, stlp,vval1,vval2 , vval4,vkac,
                              th0, vka,vkap, vka3, vka8,
                              val_par3,
                              val_par15,val_par17,val_par18,
                              val_par20,val_par21,val_par22,
                              val_par31,val_par34,val_par39,
                              valency_param_mask,cutoff2):
    ##[ind1, type1, ind2, type2, ind3, type3, bond_ind1,bond_ind2]
    #type_indices = global_body_3_inter_list[:,[1,3,5]]
    atom_indices = global_body_3_inter_list[:,[0,1,2]]
    # = type_indices.transpose()
    atom_indices = atom_indices.transpose()
    type_indices = atom_types[atom_indices]
    val_angles = global_body_3_angles
    num_atoms = len(atom_types)
    #my_valency_param_mask = valency_param_mask[atom_types,:,:][:,atom_types,:][:,:,atom_types]
    #complete_mask = valency_system_mask * my_valency_param_mask
    # first create the required data structures
    boa = bo[global_body_3_inter_list[:, -2]]
    bob = bo[global_body_3_inter_list[:, -1]]

    new_mask = np.where(boa * bob < 0.00001, 0, 1) #!Scott Habershon recommendation March 2009
    complete_mask = global_body_3_inter_list_mask * new_mask
    
    boa = boa - cutoff2
    bob = bob - cutoff2
    # thresholding
    boa = np.clip(boa, a_min=0) #if (boa.lt.zero.or.bob.lt.zero) then skip
    bob = np.clip(bob, a_min=0)
    # calculate SBO term
    # calculate sbo2 and vmbo for every atom in the sim.sys.

    neigh_indices = body_2_local_list[:,:,1].astype(np.int32)
    sbo2 = np.sum(bopi[neigh_indices],axis=1) + np.sum(bopi2[neigh_indices],axis=1)
    vmbo = np.prod(np.exp(-bo ** 8)[neigh_indices],axis=1)

    my_abo = abo[atom_indices[1]]

    exbo = abo - valf[atom_types] # calculate for every atom in the sim.sys.
    my_exbo = exbo[atom_indices[1]]

    my_vkac = vkac[type_indices[0],type_indices[1],type_indices[2]]
    evboadj = 1.0 # why?
    # to solve the nan issue, clip the vlaues
    expun = np.exp(np.clip(-my_vkac * my_exbo,CLIP_MIN,CLIP_MAX))
    expun2 = np.exp(np.clip(val_par15 * my_exbo,CLIP_MIN,CLIP_MAX))

    htun1 = 2.0 + expun2
    htun2 = 1.0 + expun + expun2
    my_vval4 = vval4[type_indices[1]]
    evboadj2 = my_vval4-(my_vval4-1.0)*np.clip(htun1/htun2,-1e15,+1e15)


    exlp1 = abo - stlp[atom_types] # calculate for every atom in the sim.sys.
    exlp2 = 2.0 * ((exlp1/2.0).astype(np.int32))# integer casting
    exlp = exlp1 - exlp2
    # fix after lone pair part
    vlpadj = 0.0
    vlpadj = np.where(exlp < 0.0, vlp, 0.0) # vlp comes from lone pair
    # TODO: temorary change related to new dataset
    sbo2 = sbo2 + (1 - vmbo) * (-exbo - val_par34 * vlpadj) # calculate for every atom in the sim.sys.
    #sbo2 = sbo2 + (1 - vmbo) * (- val_par34 * vlpadj) # calculate for every atom in the sim.sys.
    sbo2 = np.clip(sbo2, 0, 2.0)
    sbo2 = vectorized_cond(sbo2 < 1, lambda x: (x+1e-30) ** val_par17, lambda x: sbo2, sbo2) # +1e-30 to not have ln(0)

    sbo2 = vectorized_cond(sbo2 >= 1, lambda x: 2.0-(2.0-x+1e-30)**val_par17, lambda x: sbo2, sbo2) # +1e-30 to not have ln(0)
    '''
    sbo2 = np.where(sbo2 < 0.0, 0.0,
           np.where(sbo2 < 1.0, sbo2 ** val_par17,
           np.where(sbo2 < 2.0, 2.0-(2.0-sbo2)**val_par17, 2.0)))
    '''
    expsbo = np.exp(-val_par18*(2.0-sbo2))

    my_expsbo = expsbo[atom_indices[1]]
    thba = th0[type_indices[0],type_indices[1],type_indices[2]]

    thetao = 180.0 - thba * (1.0-my_expsbo)
    thetao = thetao * dgrrdn
    thdif = (thetao - val_angles)
    thdi2 = thdif * thdif

    my_vka = vka[type_indices[0],type_indices[1],type_indices[2]]
    my_vka3 = vka3[type_indices[0],type_indices[1],type_indices[2]]
    exphu = my_vka * np.exp(-my_vka3 * thdi2)

    exphu2 = my_vka - exphu
    # !To avoid linear Me-H-Me angles (6/6/06)
    exphu2 = np.where(my_vka < 0.0, exphu2 - my_vka, exphu2)

    my_vval2 = vval2[type_indices[0],type_indices[1],type_indices[2]]

    boap = (boa+1e-30) ** my_vval2 #+1e-30 so that ln(0) is not non

    bobp = (bob+1e-30) ** my_vval2 #+1e-30 so that ln(0) is not non

    my_vval1 = vval1[type_indices[1]]


    exa = np.exp(-my_vval1*boap)
    exb = np.exp(-my_vval1*bobp)

    exa2 = (1.0-exa)
    exb2 = (1.0-exb)

    evh = evboadj2*evboadj*exa2*exb2*exphu2
    evh = np.where(boa == 0, 0.0, evh)
    evh = np.where(bob == 0, 0.0, evh)
    #print(evh.shape)
    #evh = np.nan_to_num(evh)
    total_pot = np.sum(evh*complete_mask)

    '''
    mult = evh*complete_mask
    for j in range(num_atoms):
        for i in range(num_atoms):
            for k in range(i+1, num_atoms):
                if mult[i,j,k] != 0:
                    print(i+1,j+1,k+1,mult[i,j,k])
    '''


    #Calculate penalty for two double bonds in valency angle
    exbo = abo - aval[atom_types] # calculate for every atom in the sim.sys.
    expov = np.exp(val_par22 * exbo)
    expov2 = np.exp(-val_par21 * exbo)

    htov1=2.0+expov2
    htov2=1.0+expov+expov2

    ecsboadj = htov1/htov2
    my_ecsboadj = ecsboadj[atom_indices[1]] # for the center atom

    my_vkap = vkap[type_indices[0],type_indices[1],type_indices[2]]
    exphu1=np.exp(-val_par20*(boa-2.0)*(boa-2.0))
    exphu2=np.exp(-val_par20*(bob-2.0)*(bob-2.0))
    epenh=my_vkap*my_ecsboadj*exphu1*exphu2

    epenh = epenh * complete_mask
    total_penalty = np.sum(epenh)

    #Calculate valency angle conjugation energy
    abo_i = abo[atom_indices[0]]

    abo_k = abo[atom_indices[2]] # (i,j,k) will give abo for k

    unda = abo_i - boa
    ovb = my_abo - vval3[type_indices[1]]
    #print("ovb",ovb)

    undc = abo_k - bob
    #print("undc",undc)
    ba=(boa-1.50)*(boa-1.50)
    bb=(bob-1.50)*(bob-1.50)

    exphua = np.exp(np.clip(-val_par31*ba,CLIP_MIN,CLIP_MAX))
    exphub=np.exp(np.clip(-val_par31*bb,CLIP_MIN,CLIP_MAX))
    exphuua=np.exp(np.clip(-val_par39*unda*unda,CLIP_MIN,CLIP_MAX))
    exphuob=np.exp(np.clip(val_par3*ovb,CLIP_MIN,CLIP_MAX))
    #print(val_par3)
    exphuuc=np.exp(np.clip(-val_par39*undc*undc,CLIP_MIN,CLIP_MAX))
    hulpob=1.0/(1.0+exphuob)
    #print('exphuua',exphuua)
    my_vka8 = vka8[type_indices[0],type_indices[1],type_indices[2]]
    ecoah=my_vka8*exphua*exphub*exphuua*exphuuc*hulpob

    ecoah = ecoah * complete_mask
    #print("ecoah",ecoah)
    total_conj = np.sum(ecoah)


    return [total_pot,total_penalty,total_conj]


def calculate_boncor_pot(num_atoms,body_2_global_list,body_2_global_list_mask,body_2_local_list, bo, bopi, bopi2, abo, aval,vval3, bo131, bo132, bo133, ovc, v13cor, ov_coord_1, ov_coord_2):
    #Content of the body_2_global_list:
    #[ind1, type1, ind2, type2, shift[0], shift[1], shift[2]]

    #my_mask[np.diag_indices(num_atoms)] = 0.0

    type_indices = body_2_global_list[:,[1,3]]
    type_indices = type_indices.transpose()

    atom_indices = body_2_global_list[:,[0,2]]
    atom_indices = atom_indices.transpose()

    abo_j2 = abo[atom_indices[1]]
    abo_j1 = abo[atom_indices[0]]

    aval_j2 = aval[type_indices[1]]
    aval_j1 = aval[type_indices[0]]

    vp131 = safe_sqrt(bo131[type_indices[0]] * bo131[type_indices[1]])
    vp132 = safe_sqrt(bo132[type_indices[0]] * bo132[type_indices[1]])
    vp133 = safe_sqrt(bo133[type_indices[0]] * bo133[type_indices[1]])

    my_ovc = ovc[type_indices[0], type_indices[1]]

    ov_j1 = abo_j1 - aval_j1
    ov_j2 = abo_j2 - aval_j2

    #exphu1 = np.exp(-ov_coord_2*ov_j1)
    #exphu2 = np.exp(-ov_coord_2*ov_j2)
    # clipping to solve nan grad values
    exphu1 = np.exp(np.clip(-ov_coord_2*ov_j1,CLIP_MIN,CLIP_MAX))
    exphu2 = np.exp(np.clip(-ov_coord_2*ov_j2,CLIP_MIN,CLIP_MAX))
    #print("exphu1",exphu1)
    #print("exphu2",exphu2)
    exp11 = np.exp(np.clip(-ov_coord_1*ov_j1,CLIP_MIN,CLIP_MAX))
    exp21 = np.exp(np.clip(-ov_coord_1*ov_j2,CLIP_MIN,CLIP_MAX))
    exphu12 = (exphu1+exphu2)
    ovcor = -(1.0/ov_coord_2) * np.clip(np.log(0.50*exphu12),-1e+15, +1e+15)
    ovcor = np.clip(ovcor, -1e+15, +1e+15)
    huli = aval_j1+exp11+exp21
    hulj = aval_j2+exp11+exp21

    corr1 = huli/(huli+ovcor)
    corr2 = hulj/(hulj+ovcor)
    corrtot = 0.50*(corr1+corr2)

    corrtot = np.where(my_ovc > 0.001, corrtot, 1.0)

    my_v13cor = v13cor[type_indices[0], type_indices[1]]


    #v13cor_mask = np.where(my_v13cor <= 0.001)
    #print("v13cor",my_v13cor)

    vval3_j1 = vval3[type_indices[0]]
    vval3_j2 = vval3[type_indices[1]]
    #print("abo_j2", abo_j2)
    #print("vval3_j2 1 4", vval3_j2[0,3])
    #print("abo_j2 1 4", abo_j2[0,3])
    ov_j11 = abo_j1 - vval3_j1
    ov_j22 = abo_j2 - vval3_j2
    #print("abo_j1", abo_j1)
    #print("ov_j22 1 4", ov_j22[0,3])
    cor1 = vp131 * bo * bo - ov_j11
    cor2 = vp131 * bo * bo - ov_j22
    #print("cor1", cor1)
    #print("cor2 1 4", cor2[0,3])
    #print("vp132", vp132)
    #print("vp133", vp133)

    exphu3 = np.exp(np.clip(-vp132 * cor1 + vp133,CLIP_MIN,CLIP_MAX))
    exphu4 = np.exp(np.clip(-vp132 * cor2 + vp133,CLIP_MIN,CLIP_MAX))
    #print("exphu3", exphu3)
    #print("exphu4 1 4", exphu4[0,3])

    bocor1=1.0/(1.0+exphu3)
    bocor2=1.0/(1.0+exphu4)

    bocor1 = np.where(my_v13cor > 0.001, bocor1, 1.0)
    bocor2 = np.where(my_v13cor > 0.001, bocor2, 1.0)

    #print("corrtot",corrtot)
    #print("bocor1",bocor1)
    #print("bocor2",bocor2)

    bo = bo * corrtot * bocor1 * bocor2  * body_2_global_list_mask
    bo = np.where(bo < 1e-10, 0.0, bo)


    corrtot2=corrtot*corrtot
    bopi=bopi*corrtot2*bocor1*bocor2 * body_2_global_list_mask
    bopi2=bopi2*corrtot2*bocor1*bocor2 * body_2_global_list_mask

    bopi = np.where(bopi < 1e-10, 0.0, bopi)
    bopi2 = np.where(bopi2 < 1e-10, 0.0, bopi2)
    #print("bopi",bopi)
    #print("bopi2",bopi2)

    #bo = np.nan_to_num(bo)
    abo = np.sum(bo[body_2_local_list[:,:,1].astype(np.int32)],axis=1)

    return bo, abo, bopi, bopi2


def calculate_covbon_pot(atom_types,
                            body_2_global_list, body_2_global_list_mask,
                            global_body_2_distances,
                                body_2_local_list,
                            triple_bond_body_2_mask,
                            rob1, rob2, rob3,
                            ptp, pdp, popi, pdo, bop1, bop2,
                            de1, de2, de3, psp, psi,
                            trip_stab4,trip_stab5,trip_stab8,trip_stab11,
                            aval,vval3, bo131, bo132, bo133,
                            ov_coord_1, ov_coord_2,
                                bond_params_mask, cutoff,
                                rob1_mask,rob2_mask,rob3_mask,ovc, v13cor):

    num_atoms = len(atom_types)
    type_indices = body_2_global_list[:,[1,3]]
    type_indices = type_indices.transpose()

    atom_indices = body_2_global_list[:,[0,2]]
    atom_indices = atom_indices.transpose()

    symm = (atom_indices[0] == atom_indices[1]).astype(TYPE) + 1
    symm = 1.0 / symm
    distance = global_body_2_distances

    my_rob1 = rob1[type_indices[0], type_indices[1]]
    my_rob2 = rob2[type_indices[0], type_indices[1]]
    my_rob3 = rob3[type_indices[0], type_indices[1]]
    my_ptp = ptp[type_indices[0], type_indices[1]]
    my_pdp = pdp[type_indices[0], type_indices[1]]
    my_popi = popi[type_indices[0], type_indices[1]]
    my_pdo = pdo[type_indices[0], type_indices[1]]
    my_bop1 = bop1[type_indices[0], type_indices[1]]
    my_bop2 = bop2[type_indices[0], type_indices[1]]
    my_de1 = de1[type_indices[0], type_indices[1]]
    my_de2 = de2[type_indices[0], type_indices[1]]
    my_de3 = de3[type_indices[0], type_indices[1]]
    my_psp = psp[type_indices[0], type_indices[1]]
    my_psi = psi[type_indices[0], type_indices[1]]

    #rhulp = distance / my_rob1
    rhulp = vectorized_cond(my_rob1 == 0, lambda x: 0.,lambda x: distance / (x+1e-10), my_rob1)
    rhulp2 = vectorized_cond(my_rob2 == 0, lambda x: 0., lambda x: distance / (x+1e-10), my_rob2)
    rh2p = np.clip(rhulp2 ** my_ptp, -1e15,1e15)
    ehulpp = np.exp(np.clip(my_pdp * rh2p,CLIP_MIN,CLIP_MAX))
    ehulpp = np.where(my_rob2 == 0, 0.0, ehulpp)

    rhulp3 = vectorized_cond(my_rob3 == 0, lambda x: 0., lambda x: distance / (x+1e-10), my_rob3)
    #rhulp3 = distance / my_rob3
    rh2pp = np.clip((rhulp3+1e-30) ** my_popi, -1e15,1e15) # problem: if rhulp3 is 0, ln(0) is not defined
    #gradient non issue fix
    ehulppp = np.exp(my_pdo*rh2pp)
    ehulppp = np.where(my_rob3 == 0, 0.0, ehulppp)
    #ehulppp = ehulppp * my_rob3_mask



    rh2 = np.clip((rhulp+1e-30) ** my_bop2, -1e15,1e15) # problem: if rhulp is 0, ln(0) is not defined
    #gradient non issue fix
    ehulp = (1 + cutoff) * np.exp(np.clip(my_bop1 * rh2,CLIP_MIN,CLIP_MAX))
    #ehulp = jax.ops.index_update(ehulp, mask_rob1, 0.0)
    ehulp = np.where(my_rob1 == 0, 0.0, ehulp)
    bor = ehulp + ehulpp + ehulppp
    bopi = ehulpp
    bopi2 = ehulppp
    bo = bor - cutoff

    #cutoff_mask = np.where(bo <= 0) # if (bor.gt.cutoff) then
    #bo = bo * my_mask
    bo = np.where(bo <= 0, 0.0, bo)
    bo = bo * body_2_global_list_mask
    #bo = jax.ops.index_update(bo, cutoff_mask, 0.0)
    #print("bo", bo)
    # TODO: this part needs to be improved
    #print(bo[15])
    abo = np.sum(bo[body_2_local_list[:,:,1].astype(np.int32)],axis=1)

    #bo_old = bo
    bo, abo, bopi, bopi2 = calculate_boncor_pot(num_atoms,body_2_global_list, body_2_global_list_mask,body_2_local_list, bo, bopi, bopi2, abo,
                                                aval,vval3, bo131, bo132, bo133, ovc, v13cor,
                                                ov_coord_1, ov_coord_2)

    bosia = bo - bopi - bopi2
    #bosia = bosia.clip(min=0)
    bosia = np.clip(bosia, a_min=0)
    de1h = symm * my_de1
    de2h = symm * my_de2
    de3h = symm * my_de3
    #return [np.sum(bosia), [],[],[],[]] #no nan value
    bopo1 = np.clip((bosia+1e-30) ** my_psp,-1e15,+1e15)
    #return [np.sum((bosia+1e-15) ** my_psp), [],[],[],[]] #1 nan value -> problem: if bosia is 0, ln(0) is not defined
    #gradient non issue fix
    exphu1 = np.exp(np.clip(my_psi * (1.0 - bopo1),CLIP_MIN,CLIP_MAX))
    #exphu1 = np.exp(my_psi * (1.0 - bopo1))
    ebh= -de1h * bosia * exphu1 - de2h * bopi - de3h * bopi2
    #ebh = jax.ops.index_update(ebh, cutoff_mask, 0.0)

    ebh = np.where(bo <= 0, 0.0, ebh)
    #Stabilisation terminal triple bond in CO

    ba = (bo - 2.5) * (bo - 2.5)
    #print("ba",ba)
    #gradient non issue fix
    exphu = np.exp(np.clip(-trip_stab8 * ba,CLIP_MIN,CLIP_MAX))
    #exphu = np.exp(-trip_stab8 * ba)
    #print("exphu", exphu)

    abo_j2 = abo[atom_indices[1]]
    abo_j1 = abo[atom_indices[0]]
    #print("abo", abo)
    obo_a = abo_j1 - bo
    obo_b = abo_j2 - bo
    #gradient non issue fix
    exphua1 = np.exp(np.clip(-trip_stab4*obo_a,CLIP_MIN,CLIP_MAX))
    exphub1 = np.exp(np.clip(-trip_stab4*obo_b,CLIP_MIN,CLIP_MAX))
    #exphua1 = np.exp(-trip_stab4*obo_a)
    #exphub1 = np.exp(-trip_stab4*obo_b)

    #print("exphua1", exphua1)
    my_aval = aval[type_indices[0]] + aval[type_indices[1]]



    #bo_mask = np.where(bo < 1.0)
    triple_bond_body_2_mask = np.where(bo < 1.0, 0.0, triple_bond_body_2_mask)
    #triple_bond_mask[bo_mask] = 0
    ovoab = abo_j1 + abo_j2 - my_aval
    #print("ovoab", ovoab)
    #gradient nan issue fix
    exphuov = np.exp(np.clip(trip_stab5 * ovoab,CLIP_MIN,CLIP_MAX))

    hulpov=1.0/(1.0+25.0*exphuov)
    #print("hulpov",hulpov)

    estriph=trip_stab11*exphu*hulpov*(exphua1+exphub1)

    eb = ebh + estriph * triple_bond_body_2_mask
    #print("estriph",estriph)

    # nan values only appear on the diag.
    # harmless for this routine but just to make sure they disappear
    bo = np.nan_to_num(bo * body_2_global_list_mask)
    bopi = np.nan_to_num(bopi * body_2_global_list_mask)
    bopi2 = np.nan_to_num(bopi2 * body_2_global_list_mask)
    return [np.sum(eb), bo, bopi, bopi2, abo]

def calculate_bo_single(distance,
                rob1, rob2, rob3,
                ptp, pdp, popi, pdo, bop1, bop2,
                cutoff):
    '''
    to get the highest bor:
        rob1:
        bop2: group 2, line 2, col. 6
        bop1: group 2, line 2, col. 5

        rob2:
        ptp: group 2, line 2, col. 3
        pdp: group 2, line 2, col. 2

        rob3:
        popi: group 2, line 1, col. 7
        pdo: group 2, line 1, col. 5
    '''

    rhulp = np.where(rob1 == 0, 0, distance / rob1)
    rh2 = rhulp ** bop2
    ehulp = (1 + cutoff) * np.exp(bop1 * rh2)

    rhulp2 = np.where(rob1 == 0, 0, distance / rob2)
    rh2p = rhulp2 ** ptp
    ehulpp = np.exp(pdp * rh2p)


    rhulp3 = np.where(rob1 == 0, 0, distance / rob3)
    rh2pp = rhulp3 ** popi
    ehulppp = np.exp(pdo*rh2pp)

    bor = ehulp + ehulpp + ehulppp


    return bor # if < cutoff, will be ignored


def calculate_bo(body_2_global_list, body_2_global_list_mask,
                 distances,
                rob1, rob2, rob3,
                ptp, pdp, popi, pdo, bop1, bop2,
                cutoff):
    '''
    to get the highest bor:
        rob1:
        bop2: group 2, line 2, col. 6
        bop1: group 2, line 2, col. 5

        rob2:
        ptp: group 2, line 2, col. 3
        pdp: group 2, line 2, col. 2

        rob3:
        popi: group 2, line 1, col. 7
        pdo: group 2, line 1, col. 5
    '''
    type_indices = body_2_global_list[:,[1,3]]
    type_indices = type_indices.transpose()

    my_rob1 = rob1[type_indices[0], type_indices[1]]
    my_rob2 = rob2[type_indices[0], type_indices[1]]
    my_rob3 = rob3[type_indices[0], type_indices[1]]

    my_ptp = ptp[type_indices[0], type_indices[1]]
    my_pdp = pdp[type_indices[0], type_indices[1]]
    my_popi = popi[type_indices[0], type_indices[1]]

    my_pdo = pdo[type_indices[0], type_indices[1]]
    my_bop1 = bop1[type_indices[0], type_indices[1]]
    my_bop2 = bop2[type_indices[0], type_indices[1]]

    rhulp = onp.where(my_rob1 <= 0, 0, distances / (my_rob1+1e-10))
    rh2 = rhulp ** my_bop2
    ehulp = (1 + cutoff) * onp.exp(my_bop1 * rh2)

    ehulp = onp.where(my_rob1 <= 0, 0.0, ehulp)

    rhulp2 = onp.where(my_rob2 <= 0, 0, distances / (my_rob2+1e-10))
    rh2p = rhulp2 ** my_ptp
    ehulpp = onp.exp(my_pdp * rh2p)

    ehulpp = onp.where(my_rob2 <= 0, 0.0, ehulpp)

    rhulp3 = onp.where(my_rob3 <= 0, 0, distances / (my_rob3+1e-10))
    rh2pp = rhulp3 ** my_popi
    ehulppp = onp.exp(my_pdo*rh2pp)

    ehulppp = onp.where(my_rob3 <= 0, 0.0, ehulppp)

    bor = ehulp + ehulpp + ehulppp


    return bor * body_2_global_list_mask # if < cutoff, will be ignored

def calculate_vdw_pot(atom_types,atom_mask,distance_matrices,tapering_matrices, p1co, p2co, p3co, vop, vdw_shiedling):
    num_atoms = len(atom_types)
    my_vop = vop[atom_types]
    gamwh_mat = np.sqrt(my_vop.reshape(-1,1).dot(my_vop.reshape(1,-1)))
    gamwco_mat = (1.0 / gamwh_mat) ** vdw_shiedling

    # select the required values
    p1_mat = p1co[atom_types,:][:,atom_types]
    p2_mat = p2co[atom_types,:][:,atom_types]
    p3_mat = p3co[atom_types,:][:,atom_types]

    di = np.diag_indices(num_atoms)


    hulpw_mat = distance_matrices ** vdw_shiedling + gamwco_mat
    rrw_mat = hulpw_mat ** (1.0 / vdw_shiedling)
    # if p = 0 -> gradient will be 0
    temp_val2 =  p3_mat * ((1.0 - rrw_mat / p1_mat))
    #h1_mat = np.exp(temp_val)
    #h2_mat = np.exp(0.5 * temp_val)
    #gradient nan issue fix
    h1_mat = np.exp(temp_val2)
    h2_mat = np.exp(0.5 * temp_val2)
    ewh_mat = p2_mat * (h1_mat - 2.0 * h2_mat)
    ewhtap_mat = ewh_mat * tapering_matrices

    #half the self potential
    atom_mask_2d = atom_mask.reshape(-1,1) * atom_mask.reshape(1,-1)
    self_multip = np.ones((num_atoms,num_atoms))
    self_multip = jax.ops.index_update(self_multip, jax.ops.index[di], 0.5)
    ewhtap_mat = ewhtap_mat * (self_multip * atom_mask_2d)

    total_pot = np.sum(np.triu(np.sum(ewhtap_mat,axis=0)))
    return total_pot



def calculate_coulomb_pot(atom_types, hulp1_mat,tapering_matrices, charges, gamma):
    num_atoms = len(atom_types)
    charge_mat = charges.reshape(-1,1).dot(charges.reshape(1,-1))
    di = np.diag_indices(num_atoms)

    #eph_mat = np.where(hulp1_mat == 0, 0.0,  c1c * charge_mat / (hulp1_mat ** (1.0/3.0)))
    eph_mat = c1c * charge_mat / (hulp1_mat ** (1.0/3.0))
    ephtap_mat = eph_mat * tapering_matrices
    #half the self potential
    self_multip = np.ones((num_atoms,num_atoms))
    self_multip = jax.ops.index_update(self_multip, jax.ops.index[di], 0.5)
    ephtap_mat = ephtap_mat * self_multip
    total_pot = np.sum(np.triu(np.sum(ephtap_mat,axis=0)))

    return total_pot
def calculate_charge_energy(atom_types, charges, idempotential, electronegativity):

    ech = np.sum(23.02 * (electronegativity[atom_types] * charges +
                 idempotential[atom_types] * np.square(charges)))
    return ech

def calculate_eem_charges(atom_types,atom_mask,total_charge,hulp1_mat,tapering_matrices, gamma, idempotential, electronegativity):

    num_atoms = len(atom_types)

    # create the matrix
    A = np.zeros(shape=(num_atoms + 1, num_atoms + 1), dtype=TYPE)
    A = jax.ops.index_update(A, jax.ops.index[num_atoms, :num_atoms], atom_mask)
    A = jax.ops.index_update(A, jax.ops.index[:num_atoms, num_atoms], atom_mask)

    hulp2_mat = hulp1_mat**(1.0/3.0)

    #new_sub_A = tapering_matrices * 14.4 / hulp2_mat
    new_sub_A = vectorized_cond(hulp2_mat == 0.0, lambda x: 0.0, lambda x: tapering_matrices * 14.4 / hulp2_mat, hulp2_mat)
    #new_sub_A = np.where(hulp2_mat == 0, 0.0,  tapering_matrices * 14.4 / hulp2_mat)
    A_sub_mat = np.sum(new_sub_A, axis=0)

    di = np.diag_indices(num_atoms)
    A_sub_mat = jax.ops.index_update(A_sub_mat, jax.ops.index[di], 2.0 * idempotential[atom_types] + A_sub_mat[di])

    A = jax.ops.index_update(A, jax.ops.index[:num_atoms,:num_atoms], A_sub_mat)

    A = jax.ops.index_update(A, jax.ops.index[num_atoms, num_atoms], 0.0)
    #A[num_atoms][num_atoms] = 0.0
    # create b
    b = np.zeros(shape=(num_atoms+1), dtype=TYPE)
    #b[-1] = 0 # total charge is 0
    b = jax.ops.index_update(b, jax.ops.index[:num_atoms], -1 * electronegativity[atom_types])
    # total charge
    b = jax.ops.index_update(b, jax.ops.index[num_atoms], total_charge)

    atom_charges = np.linalg.solve(A, b)
    #atom_charges = jax.scipy.linalg.solve_triangular(A, b)
    #atom_charges = jax.scipy.linalg.solve(A,b,sym_pos=True)
    atom_charges = jax.ops.index_update(atom_charges, jax.ops.index[:-1], atom_charges[:-1] * atom_mask)
    return atom_charges


def calculate_hb_pot(atom_types,bo,global_hbond_inter_list,global_hbond_inter_list_mask,hbond_angles_and_dist, rhb, dehb, vhb1, vhb2):

    type_indices = global_hbond_inter_list[:,[1,3,5]].transpose()

    my_rhb = rhb[type_indices[0],type_indices[1],type_indices[2]]
    my_dehb = dehb[type_indices[0],type_indices[1],type_indices[2]]
    my_vhb1 = vhb1[type_indices[0],type_indices[1],type_indices[2]]
    my_vhb2 = vhb2[type_indices[0],type_indices[1],type_indices[2]]

    boa = bo[global_hbond_inter_list[:,-1]]
    boa = np.where(boa > 0.01, boa, 0.0)

    angles = hbond_angles_and_dist[:,0]
    dist = hbond_angles_and_dist[:,1]

    global_hbond_inter_list_mask = np.logical_and(global_hbond_inter_list_mask, dist < 7.5) # TODO: put this value in the non diff. section of force field

    # to not get divide by zero
    rhu1 = my_rhb / (dist + 1e-10)
    rhu2 = dist / (my_rhb + 1e-10)

    exphu1 = np.exp(-my_vhb1 * boa)
    exphu2 = np.exp(-my_vhb2 * (rhu1 + rhu2 - 2.0))

    ehbh = (1.0-exphu1) * my_dehb * exphu2 * np.power(np.sin((angles + 1e-10)/2.0), 4) * global_hbond_inter_list_mask

    return np.sum(ehbh)

def taper(dist, low_tap_rad, up_tap_rad):

    dist = dist - low_tap_rad
    up_tap_rad = up_tap_rad - low_tap_rad
    low_tap_rad = 0

    R = dist
    R2 = dist * dist
    R3 = R2 * R


    SWB = up_tap_rad
    SWA = low_tap_rad

    D1 = SWB-SWA
    D7 = D1**7.0
    SWA2 = SWA*SWA
    SWA3 = SWA2*SWA
    SWB2 = SWB*SWB
    SWB3 = SWB2*SWB

    SWC7 =  20.0/D7
    SWC6 = -70.0*(SWA+SWB)/D7
    SWC5 =  84.0*(SWA2+3.0*SWA*SWB+SWB2)/D7
    SWC4 = -35.0*(SWA3+9.0*SWA2*SWB+9.0*SWA*SWB2+SWB3)/D7
    SWC3 = 140.0*(SWA3*SWB+3.0*SWA2*SWB2+SWA*SWB3)/D7
    SWC2 = -210.0*(SWA3*SWB2+SWA2*SWB3)/D7
    SWC1 = 140.0*SWA3*SWB3/D7
    SWC0 = (-35.0*SWA3*SWB2*SWB2+21.0*SWA2*SWB3*SWB2+ \
        7.0*SWA*SWB3*SWB3+SWB3*SWB3*SWB)/D7

    SW = SWC7*R3*R3*R+SWC6*R3*R3+SWC5*R3*R2+SWC4*R2*R2+SWC3*R3+SWC2*R2+ \
       SWC1*R+SWC0

    return SW

def taper2(dist, low_tap_rad=-2, up_tap_rad=2):

    return 1 - taper(dist, low_tap_rad, up_tap_rad)


def taper_BO(x,low_dist, high_dist):
    x_r = high_dist - low_dist
    S1 =  140.0 / x_r
    S2 = -210.0 / np.power(x_r,2)
    S3 =  140.0 / np.power(x_r,3)
    S4 = -35.0 /  np.power(x_r,4)
    S5 =  84.0 /  np.power(x_r,5)
    S6 = -70.0 /  np.power(x_r,6)
    S7 =  20.0 /  np.power(x_r,7)

    tap = S7 * np.power(x,7) + S6 * np.power(x,6) + S5 * np.power(x,5) +  \
        S4 * np.power(x,4) + S3 * np.power(x,3) + S2 * np.power(x,2) + \
        S1 * x + 1.0
    return tap

def S(x,low_dist, high_dist):
    return np.where(x <= low_dist, 0.0,
                (np.where(x <= high_dist,
                  taper(x, low_dist, high_dist),
                  1.0)))
