#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains helper functions for I/O and training

Author: Mehmet Cagri Kaymak
"""

import  os
from jaxreaxff.forcefield import ForceField
from jaxreaxff.forcefield import body_3_indices_src,body_3_indices_dst,body_4_indices_src,body_4_indices_dst,TYPE
from jaxreaxff.structure import Structure,CLOSE_NEIGH_CUTOFF,BUFFER_DIST
from jaxreaxff.reaxffpotential import calculate_bo
from jaxreaxff.forcefield import MY_ATOM_INDICES,preprocess_force_field
import jax
#for multicore
import jax.numpy as np
import numpy as onp
import time
import sys
from functools import partial
import copy
from multiprocessing import Pool
from jaxreaxff.clustering import modified_kmeans
from tabulate import tabulate


# Produces a report with item based error (similar to what the standalone code does)
def produce_error_report(filename, tranining_items, tranining_items_str, indiv_error):
    fptr = open(filename, 'w')
    headers = ["Item Text", "Weight", "Target", "Prediction", "Error", "Cum. Sum."]
    data_to_print = []
    cumulative_err = 0.0
    if "ENERGY" in tranining_items:

        [preds,error_vals] = [indiv_error['ENERGY'][0], indiv_error['ENERGY'][-1]]
        for i,strr in enumerate(tranining_items_str['ENERGY']):
            parts = strr.strip().split()
            weight = parts[0]
            target = parts[-1]
            rest = " ".join(parts[1:-1])
            cumulative_err += error_vals[i]
            row = [rest, weight, target, preds[i], round(error_vals[i],2), round(cumulative_err,2)]
            data_to_print.append(row)

    if "CHARGE" in tranining_items:

        [preds,error_vals] = [indiv_error['CHARGE'][0], indiv_error['CHARGE'][-1]]
        for i,strr in enumerate(tranining_items_str['CHARGE']):
            parts = strr.strip().split()
            weight = parts[1]
            target = parts[-1]
            rest = " ".join([parts[0],] + parts[2:-1])
            cumulative_err += error_vals[i]
            row = [rest, weight, target, preds[i], round(error_vals[i],2), round(cumulative_err,2)]
            data_to_print.append(row)

    if "GEOMETRY-2" in tranining_items:

        [preds,error_vals] = [indiv_error['GEOMETRY-2'][0], indiv_error['GEOMETRY-2'][-1]]
        for i,strr in enumerate(tranining_items_str['GEOMETRY-2']):
            parts = strr.strip().split()
            weight = parts[1]
            target = parts[-1]
            rest = " ".join([parts[0],] + parts[2:-1])
            cumulative_err += error_vals[i]
            row = [rest, weight, target, preds[i], round(error_vals[i],2), round(cumulative_err,2)]
            data_to_print.append(row)

    if "GEOMETRY-3" in tranining_items:
        [preds,error_vals] = [indiv_error['GEOMETRY-3'][0], indiv_error['GEOMETRY-3'][-1]]
        for i,strr in enumerate(tranining_items_str['GEOMETRY-3']):
            parts = strr.strip().split()
            weight = parts[1]
            target = parts[-1]
            rest = " ".join([parts[0],] + parts[2:-1])
            cumulative_err += error_vals[i]
            row = [rest, weight, target, preds[i], round(error_vals[i],2), round(cumulative_err,2)]
            data_to_print.append(row)

    if "GEOMETRY-4" in tranining_items:
        [preds,error_vals] = [indiv_error['GEOMETRY-4'][0], indiv_error['GEOMETRY-4'][-1]]
        for i,strr in enumerate(tranining_items_str['GEOMETRY-4']):
            parts = strr.strip().split()
            weight = parts[1]
            target = parts[-1]
            rest = " ".join([parts[0],] + parts[2:-1])
            cumulative_err += error_vals[i]
            row = [rest, weight, target, preds[i], round(error_vals[i],2), round(cumulative_err,2)]
            data_to_print.append(row)

    if "FORCE-RMSG" in tranining_items:
        [preds,error_vals] = [indiv_error['FORCE-RMSG'][0], indiv_error['FORCE-RMSG'][-1]]
        for i,strr in enumerate(tranining_items_str['FORCE-RMSG']):
            parts = strr.strip().split()
            weight = parts[1]
            target = parts[-1]
            rest = " ".join([parts[0],] + parts[2:-1])
            cumulative_err += error_vals[i]
            row = [rest, weight, target, preds[i], round(error_vals[i],2), round(cumulative_err,2)]
            data_to_print.append(row)
    if "FORCE-ATOM" in tranining_items:
        [preds,error_vals] = [indiv_error['FORCE-ATOM'][0], indiv_error['FORCE-ATOM'][-1]]
        for i,strr in enumerate(tranining_items_str['FORCE-ATOM']):
            parts = strr.strip().split()
            weight = parts[1]
            targets = parts[3:6] # x-y-z
            rest = " ".join([parts[0],parts[2]])
            cumulative_err += error_vals[i][0]
            row = [rest + " X", weight, targets[0], preds[i][0], round(error_vals[i][0],2), round(cumulative_err,2)]
            data_to_print.append(row)
            cumulative_err += error_vals[i][1]
            row = [rest + " Y", weight, targets[1], preds[i][1], round(error_vals[i][1],2), round(cumulative_err,2)]
            data_to_print.append(row)
            cumulative_err += error_vals[i][2]
            row = [rest + " Z", weight, targets[2], preds[i][2], round(error_vals[i][2],2), round(cumulative_err,2)]
            data_to_print.append(row)
    table = tabulate(data_to_print, headers, floatfmt=".2f")
    print(table, file=fptr)
    fptr.close()
    

def generate_BGF_file(file_name, geo_name, num_atoms, positions, str_types, box_dim, box_ang):
    lines =  ["XTLGRF 200"]
    lines.append("DESCRP {}".format(geo_name))
    box_line = "CRYSTX  {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f}".format(box_dim[0],
                       box_dim[1],box_dim[2],box_ang[0],box_ang[1],box_ang[2])
    format_line = "FORMAT ATOM   (a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)"
    lines.append(box_line)
    lines.append(format_line)
    for i in range(num_atoms):
        atom_line = "HETATM {:5} {:2}               {:10.5f}{:10.5f}{:10.5f}    {:2}  1 1  0.00000".format((i+1),str_types[i],
                                                                                      positions[i][0],positions[i][1],positions[i][2],str_types[i])
        lines.append(atom_line)
    lines.append('END')

    with open(file_name, 'w') as f:
        for line in lines:
            f.write("%s\n" % line)

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

    rhulp = np.where(rob1 <= 0, 0, distance / rob1)
    rh2 = rhulp ** bop2
    ehulp = (1 + cutoff) * np.exp(bop1 * rh2)
    ehulp = np.where(rob1 <= 0, 0.0, ehulp)

    rhulp2 = np.where(rob2 == 0, 0, distance / rob2)
    rh2p = rhulp2 ** ptp
    ehulpp = np.exp(pdp * rh2p)
    ehulpp = np.where(rob2 <= 0, 0.0, ehulpp)


    rhulp3 = np.where(rob3 == 0, 0, distance / rob3)
    rh2pp = rhulp3 ** popi
    ehulppp = np.exp(pdo*rh2pp)
    ehulppp = np.where(rob3 <= 0, 0.0, ehulppp)

    #print(ehulp , ehulpp , ehulppp)
    bor = ehulp + ehulpp + ehulppp


    return bor # if < cutoff, will be ignored

def set_flattened_force_field(flattened_force_field,param_indices, bounds):
    copy_ff = copy.deepcopy(flattened_force_field)
    bond_param_indices_max = set([8,15,9,12,10,14,75,76,77,81,82,83]) #rob 75,76,77 rob_off 81 82 83
    bond_param_indices_min = set([16,11,13])
    for i,p in enumerate(param_indices):
        bound = bounds[i]
        if p[0] in bond_param_indices_max:
            copy_ff[p[0]][p[1]] = bound[1]
        if p[0] in bond_param_indices_min:
            copy_ff[p[0]][p[1]] = bound[0]
    return copy_ff

def find_limits(type1, type2, flattened_force_field, cutoff):
    vect_bo_function = jax.jit(jax.vmap(calculate_bo_single,in_axes=(0,None,None,None,None,None,None,None,None,None,None)),backend='cpu')
    rob1 = flattened_force_field[8][type1,type2] # select max, typical values (0.1, 2)
    bop2 = flattened_force_field[16][type1,type2] # select min, typical values (1,10) (negative doesnt make sense)
    bop1 = flattened_force_field[15][type1,type2] # select max, typical values (-0.2, -0.01)

    rob2 = flattened_force_field[9][type1,type2]
    ptp = flattened_force_field[11][type1,type2]
    pdp = flattened_force_field[12][type1,type2]

    rob3 = flattened_force_field[10][type1,type2]
    popi = flattened_force_field[13][type1,type2]
    pdo = flattened_force_field[14][type1,type2]

    distance = np.linspace(0.0, 10, 1000)

    res = vect_bo_function(distance,
                    rob1, rob2, rob3,
                    ptp, pdp, popi, pdo, bop1, bop2,
                    cutoff)

    ind = np.sum(res > cutoff)

    return distance[ind]
# return a matrix where (i,j) is the cutoff dist. for a bond between type i and type j
def find_all_cutoffs(flattened_force_field,flattened_non_dif_params,cutoff,atom_indices):
    lenn = len(atom_indices)
    cutoff_dict = dict()
    flattened_force_field = preprocess_force_field(flattened_force_field, flattened_non_dif_params)
    for i in range(lenn):
        type_i = atom_indices[i]
        for j in range(i,lenn):
            type_j = atom_indices[j]
            dist = find_limits(type_i,type_j, flattened_force_field, cutoff)
            cutoff_dict[(type_i,type_j)] = dist
            cutoff_dict[(type_j,type_i)] = dist

            if dist > CLOSE_NEIGH_CUTOFF and type_i != -1 and type_j!=-1: #-1 TYPE is for the filler atom
                print("[WARNING] between type {0:d} and type {1:d}, the bond length could be greater than {2:.1f} A! ({3:.1f} A)".format(type_i,type_j,CLOSE_NEIGH_CUTOFF,dist))
                cutoff_dict[(type_i,type_j)] = CLOSE_NEIGH_CUTOFF
                cutoff_dict[(type_j,type_i)] = CLOSE_NEIGH_CUTOFF
    return cutoff_dict

def process_and_cluster_geos(systems,force_field,param_indices,bounds,max_num_clusters=10,all_cut_indices=None, num_threads=1, list_prev_max_dict=None):
    start = time.time()
    saved_all_pots = []
    saved_all_total_pots = []
    h_pot = []
    all_atom_types = set()
    start = time.time()
    all_time_data = onp.zeros((len(systems),6))
    for i,s in enumerate(systems):
        s.fill_atom_types(force_field)
        for a_type in s.atom_types:
            all_atom_types.add(a_type)

    cutoff_dict = find_all_cutoffs(force_field.flattened_force_field, force_field.non_dif_params,force_field.cutoff,list(all_atom_types))
    force_field.cutoff_dict = cutoff_dict
    # copy ff with bond values to max. bond length
    copy_ff = set_flattened_force_field(force_field.flattened_force_field,param_indices, bounds)
    copy_ff = preprocess_force_field(copy_ff, force_field.non_dif_params)

    end = time.time()
    pool = Pool(num_threads)
    start = time.time()
    pool_handler_for_inter_list_generation(systems,copy_ff,force_field,pool)
    end = time.time()
    pool.terminate()

    print("Multithreaded interaction list generation took {:.2f} secs with {} threads".format(end-start,num_threads))

    if all_cut_indices == None:
        all_costs_old = []
        prev = -1
        selected_n_cut = 0
        for n_cut in range(1,max_num_clusters+1):
            all_cut_indices,cost_total = cluster_systems_for_aligning(systems,num_cuts=n_cut,max_iterations=1000,rep_count=1000,print_mode=False)
            #print("Cost with {} clusters: {}".format(n_cut, cost_total))
            all_costs_old.append(cost_total)
            if prev != -1 and cost_total > prev or (prev-cost_total) / prev < 0.15:
                selected_n_cut = n_cut - 1
                break
            prev = cost_total
        #sys.exit()
        if selected_n_cut == 0:
            selected_n_cut = max_num_clusters
        all_cut_indices,cost_total = cluster_systems_for_aligning(systems,num_cuts=selected_n_cut,max_iterations=1000,rep_count=1000,print_mode=True)
    
    globally_sorted_indices = []
    for l in all_cut_indices:
        for ind in l:
            globally_sorted_indices.append(ind)


    [list_all_type,
    list_all_mask,
    list_all_total_charge,
    #list_all_dist_mat,
    list_all_body_2_list,
    list_all_body_2_map,
    list_all_body_2_neigh_list,
    list_all_body_2_trip_mask,
    list_all_body_3_list,
    list_all_body_3_map,
    list_all_body_3_shift,
    list_all_body_4_list,
    list_all_body_4_map,
    list_all_body_4_shift,
    list_all_hbond_list,
    list_all_hbond_mask,
    list_all_hbond_shift,
    list_real_atom_counts,
    list_orth_matrices,
    list_all_pos,
    list_is_periodic,
    list_all_shift_combs,
    list_bond_rest,
    list_angle_rest,
    list_torsion_rest,
    list_do_minim,
    list_num_minim_steps],cur_max_dict = align_system_inter_lists(systems, all_cut_indices,list_prev_max_dict=list_prev_max_dict)

    ordered_systems = [systems[i] for i in globally_sorted_indices]
    orig_list_pos = copy.deepcopy(list_all_pos)
    ordered_names = [s.name for s in ordered_systems]



    ##############################################################################

    list_all_dist_mat = [jax.vmap(Structure.create_distance_matrices)(list_all_pos[i],list_orth_matrices[i],list_all_shift_combs[i])  for i in range(len(list_all_type))]
    list_all_body_2_distances = [jax.vmap(Structure.calculate_2_body_distances)(list_all_pos[i],list_orth_matrices[i],list_all_body_2_list[i],list_all_body_2_map[i]) for i in range(len(list_all_type))]
    list_all_body_3_angles = [jax.vmap(Structure.calculate_3_body_angles)(list_all_pos[i],list_orth_matrices[i],list_all_body_2_list[i],list_all_body_3_list[i],list_all_body_3_map[i],list_all_body_3_shift[i]) for i in range(len(list_all_type))]
    list_all_body_4_angles = [jax.vmap(Structure.calculate_body_4_angles_new)(list_all_pos[i],list_orth_matrices[i],list_all_body_4_list[i],list_all_body_4_map[i],list_all_body_4_shift[i]) for i in range(len(list_all_type))]
    list_all_angles_and_dist = [jax.vmap(Structure.calculate_global_hbond_angles_and_dist)(list_all_pos[i],list_orth_matrices[i],list_all_hbond_list[i],list_all_hbond_shift[i],list_all_hbond_mask[i]) for i in range(len(list_all_type))]
    

    return (ordered_systems,[list_all_type,
                            list_all_mask,
                            list_all_total_charge,
                            #list_all_dist_mat,
                            list_all_body_2_list,
                            list_all_body_2_map,
                            list_all_body_2_neigh_list,
                            list_all_body_2_trip_mask,
                            list_all_body_3_list,
                            list_all_body_3_map,
                            list_all_body_3_shift,
                            list_all_body_4_list,
                            list_all_body_4_map,
                            list_all_body_4_shift,
                            list_all_hbond_list,
                            list_all_hbond_mask,
                            list_all_hbond_shift,
                            list_real_atom_counts,
                            list_orth_matrices,
                            list_all_pos,
                            list_is_periodic,
                            list_all_shift_combs,
                            list_bond_rest,
                            list_angle_rest,
                            list_torsion_rest,
                            list_do_minim,
                            list_num_minim_steps],
                            [list_all_dist_mat,
                             list_all_body_2_distances,
                             list_all_body_3_angles,
                             list_all_body_4_angles,
                             list_all_angles_and_dist])

def pool_handler_for_inter_list_generation(systems,flattened_force_field,force_field,pool):
    # prepare input
    all_flat_systems = [s.flatten_no_inter_list() for s in systems]

    modified_create_inter_lists = partial(create_inter_lists,force_field=force_field,flattened_force_field=flattened_force_field)
    all_inter_lists = pool.starmap(modified_create_inter_lists, all_flat_systems)

    for i,s in enumerate(systems):
        s.distance_matrices = all_inter_lists[i][0]

        [s.local_body_2_neigh_list,s.local_body_2_neigh_counts] = all_inter_lists[i][1]

        [s.global_body_2_inter_list,s.global_body_2_inter_list_mask,
        s.triple_bond_body_2_mask,s.global_body_2_distances,
        s.global_body_2_count] = all_inter_lists[i][2]

        [s.global_body_3_inter_list,
         s.global_body_3_inter_list_mask,
         s.global_body_3_inter_shift_map,
         s.global_body_3_count] = all_inter_lists[i][3]

        [s.global_body_4_inter_list,
         s.global_body_4_inter_shift,
         s.global_body_4_inter_list_mask,
         s.global_body_4_count]= all_inter_lists[i][4]

        [s.global_hbond_inter_list,
         s.global_hbond_shift_list,
         s.global_hbond_inter_list_mask,
         s.global_hbond_count] = all_inter_lists[i][5]

# can be usedfor multi processing
def create_inter_lists(is_periodic,
                    do_minim,
                    num_atoms,
                    real_atom_count,
                    atom_types,
                    atom_names,
                    atom_positions,
                    orth_matrix,
                    all_shift_comb,
                    flattened_force_field,
                    force_field):

    #start = time.time()
    distance_matrices = Structure.create_distance_matrices_onp(atom_positions,orth_matrix, all_shift_comb)
    #end = time.time()
    #print("distance_matrices", end-start)
    #start = time.time()
    local_neigh_arrays = [local_body_2_neigh_list,
                        local_body_2_neigh_counts] = Structure.create_local_neigh_list(num_atoms,
                                                        real_atom_count,
                                                        atom_types,
                                                        distance_matrices,
                                                        all_shift_comb,
                                                        force_field.cutoff_dict,
                                                        do_minim)
    #end = time.time()
    #print("local_neigh_arrays", end-start)
    #start = time.time()
    body_2_arrays = [global_body_2_inter_list,global_body_2_inter_list_mask,
    triple_bond_body_2_mask,global_body_2_distances,
    global_body_2_count] = Structure.create_global_body_2_inter_list(real_atom_count,
                            atom_types,atom_names,
                            atom_positions,orth_matrix,
                            local_body_2_neigh_counts,local_body_2_neigh_list,
                            force_field.bond_params_mask)

    #start = time.time()
    modified_dist = global_body_2_distances - BUFFER_DIST * do_minim
    modified_dist = onp.where(modified_dist < 1e-5, 1e-5,modified_dist)

    bo = calculate_bo(global_body_2_inter_list,
                    global_body_2_inter_list_mask,
                    modified_dist,
                    flattened_force_field[8], flattened_force_field[9], flattened_force_field[10],
                    flattened_force_field[11], flattened_force_field[12], flattened_force_field[13],
                    flattened_force_field[14], flattened_force_field[15], flattened_force_field[16],
                    force_field.cutoff)

    bo = bo - force_field.cutoff
    bo = onp.where(bo > 0.0, bo, 0.0)
    #end = time.time()
    #print("calculate_bo", end-start)

    #start = time.time()
    body_3_arrays = [global_body_3_inter_list,
     global_body_3_inter_list_mask,
     global_body_3_inter_shift_map,
     global_body_3_count] = Structure.create_body_3_inter_list(is_periodic,
                            real_atom_count,atom_types,atom_names,atom_positions,orth_matrix,
                            local_body_2_neigh_counts,local_body_2_neigh_list,
                            global_body_2_inter_list,global_body_2_distances,bo,
                            force_field.valency_params_mask,
                            force_field.cutoff2)
    #end = time.time()
    #print("body_3_arrays", end-start)

    #start = time.time()
    body_4_arrays = [global_body_4_inter_list,
     global_body_4_inter_shift,
     global_body_4_inter_list_mask,
     global_body_4_count]= Structure.create_body_4_inter_list_fast(is_periodic,
                               real_atom_count,atom_types,atom_names,atom_positions,orth_matrix,
                               local_body_2_neigh_counts,local_body_2_neigh_list,
                               global_body_2_inter_list,global_body_2_distances,bo,global_body_2_count,
                               force_field.torsion_params_mask,
                               force_field.cutoff2)
    #end = time.time()
    #print("body_4_arrays", end-start)

    #start = time.time()
    body_h_arrays = [global_hbond_inter_list,
     global_hbond_shift_list,
     global_hbond_inter_list_mask,
     global_hbond_count] = Structure.create_global_hbond_inter_list(is_periodic,do_minim,
                               real_atom_count,atom_types,atom_names,
                               atom_positions,orth_matrix,
                               distance_matrices,all_shift_comb,
                               local_body_2_neigh_counts,local_body_2_neigh_list,
                               global_body_2_inter_list,global_body_2_distances,global_body_2_count,
                               force_field.nphb,force_field.hbond_params_mask)
    #end = time.time()
    #print("body_h_arrays", end-start)

    return [distance_matrices,local_neigh_arrays,body_2_arrays,body_3_arrays,body_4_arrays,body_h_arrays]


def print_extra_rmsg_items(systems,folder):
    with open("{}/new_items.txt".format(folder), 'w') as out_file:
        out_file.write("RMSG-NEW\n")
        for s in systems:
            if s.do_minimization:
                line = "{} {} {}\n".format(s.name, 1.0, 25.0)
                out_file.write(line)
        out_file.write("ENDRMSG-NEW\n")


def align_atom_counts_and_local_neigh(systems,all_type,all_mask,positions,all_body_2_neigh_list):

    for i,s in enumerate(systems):
        cur_neigh_cnt = s.local_body_2_neigh_list.shape[1]
        all_body_2_neigh_list[i,:s.num_atoms,:cur_neigh_cnt,:] = s.local_body_2_neigh_list[:s.num_atoms,:cur_neigh_cnt,:]
        all_body_2_neigh_list[i,s.num_atoms:,:,:2] = -1
        all_body_2_neigh_list[i, :s.num_atoms,cur_neigh_cnt:,:2] = -1

        all_mask[i,:s.num_atoms] = s.atom_mask[:s.num_atoms]

        all_type[i,:s.num_atoms] = s.atom_types[:s.num_atoms]
        all_type[i,s.num_atoms:] = -1

        positions[i,:s.num_atoms,:] = s.atom_positions[:s.num_atoms,:]
        diff = positions.shape[1] - s.num_atoms
        cur = onp.array([10000,10000,10000])
        incr = onp.array([100,100,100])
        for k in range(diff):
            cur = cur+incr
            positions[i,s.num_atoms + k,:] = cur[:]


        #distance matrices need to be created again
        #s.distance_matrices = Structure.create_distance_matrices(s.atom_positions,s.box_size, s.all_shift_comb)

def align_body_2_inter_list(systems,all_body_2_list,all_body_2_mask,all_body_2_trip_mask):
    for i,s in enumerate(systems):
        cur_count = s.global_body_2_count

        all_body_2_list[i,:cur_count,:] = s.global_body_2_inter_list[:cur_count,:]
        all_body_2_list[i,cur_count:,:4] = [-1,-1,-1,-1]
        all_body_2_mask[i,:cur_count] = s.global_body_2_inter_list_mask[:cur_count]
        all_body_2_trip_mask[i,:cur_count] = s.triple_bond_body_2_mask[:cur_count]


def align_body_3_inter_list(systems,all_body_3_list,all_body_3_mask,all_body_3_shift):
    for i,s in enumerate(systems):
        cur_count = s.global_body_3_count

        all_body_3_list[i,:cur_count,:] = s.global_body_3_inter_list[:cur_count,:]
        all_body_3_list[i,cur_count:,:] = -1
        all_body_3_mask[i,:cur_count] = s.global_body_3_inter_list_mask[:cur_count]
        all_body_3_shift[i,:cur_count,:] = s.global_body_3_inter_shift_map[:cur_count,:]

def align_body_4_inter_list(systems,all_body_4_list,all_body_4_mask,all_body_4_shift):
    for i,s in enumerate(systems):
        cur_count = s.global_body_4_count

        all_body_4_list[i,:cur_count,:] = s.global_body_4_inter_list[:cur_count,:]
        all_body_4_list[i,cur_count:,:] = -1
        all_body_4_mask[i,:cur_count] = s.global_body_4_inter_list_mask[:cur_count]
        all_body_4_shift[i,:cur_count,:] = s.global_body_4_inter_shift[:cur_count,:]


def align_hbond_inter_list(systems,all_hbond_list,all_hbond_mask,all_hbond_shift):
    for i,s in enumerate(systems):
        cur_count = s.global_hbond_count

        all_hbond_list[i,:cur_count,:] = s.global_hbond_inter_list[:cur_count,:]
        all_hbond_list[i,cur_count:,:] = -1
        all_hbond_mask[i,:cur_count] = s.global_hbond_inter_list_mask[:cur_count]
        all_hbond_shift[i,:cur_count,:] = s.global_hbond_shift_list[:cur_count,:]


def align_all_shift_combinations(systems, shift_combs):
    for i,s in enumerate(systems):
        cur_len = s.all_shift_comb.shape[0]
        shift_combs[i,:cur_len,:] = s.all_shift_comb[:cur_len,:]
        shift_combs[i, cur_len:,:] = 999 # so that all the values will be truncated
    # distance matrices will be recreated in align_atom_counts_and_local_neigh

def cluster_systems_for_aligning(systems,num_cuts=5,max_iterations=100,rep_count=20, print_mode=True):
    # from size arrays for clustering

    labels,min_centr,min_counts,min_cost = modified_kmeans(systems,k=num_cuts,max_iterations=max_iterations, rep_count=rep_count, print_mode=print_mode)

    all_cut_indices = [[] for i in range(num_cuts)]
    for i,s in enumerate(systems):
        label = labels[i]
        all_cut_indices[label].append(i)

    return all_cut_indices, min_cost

def calculate_max_counts(systems):
    cur_max_dict = dict()
    list_max_atom_cnt = []
    list_max_neigh_cnt = []
    list_max_shift_cnt = []
    list_max_body_2_cnt = []
    list_max_body_3_cnt = []
    list_max_body_4_cnt = []
    list_max_hbond_cnt = []

    atom_counts = [s.num_atoms for s in systems]
    max_num_atoms = max(atom_counts)
    #list_max_atom_cnt.append(max_num_atoms)

    neigh_counts = [s.local_body_2_neigh_list.shape[1] for s in systems]
    max_neigh_count = max(neigh_counts)
    #list_max_neigh_cnt.append(max_neigh_count)

    len_shift_combs = [len(s.all_shift_comb) for s in systems]
    max_len_shift_count = max(len_shift_combs)
    #list_max_shift_cnt.append(max_len_shift_count)

    body_2_counts = [s.global_body_2_count for s in systems]
    max_body_2_count = max(body_2_counts)
    #list_max_body_2_cnt.append(max_body_2_count)

    body_3_counts = [s.global_body_3_count for s in systems]
    max_body_3_count = max(body_3_counts)
    #list_max_body_3_cnt.append(max_body_3_count)

    body_4_counts = [s.global_body_4_count for s in systems]
    max_body_4_count = max(body_4_counts)
    #list_max_body_4_cnt.append(max_body_4_count)

    hbond_counts = [s.global_hbond_count for s in systems]
    max_hbond_count = max(hbond_counts)
    #list_max_hbond_cnt.append(max_hbond_count)

    return max_num_atoms,max_len_shift_count, max_neigh_count,max_body_2_count,max_body_3_count,max_body_4_count,max_hbond_count



def align_system_inter_lists(systems, all_cut_indices,list_prev_max_dict=None):

    list_all_type = []
    list_all_mask = []
    list_all_total_charge = []
    list_all_dist_mat = []
    list_all_body_2_list = []
    list_all_body_2_map = []
    list_all_body_2_neigh_list = []
    list_all_body_2_trip_mask = []
    list_all_body_3_list = []
    list_all_body_3_map = []
    list_all_body_3_shift = []
    list_all_body_4_list = []
    list_all_body_4_map = []
    list_all_body_4_shift = []
    list_all_hbond_list = []
    list_all_hbond_mask = []
    list_all_hbond_shift = []
    list_real_atom_counts = []
    list_orth_matrices = []
    list_positions = []
    list_is_periodic = []
    list_all_shift_combs = []
    list_bond_rest = []
    list_angle_rest = []
    list_torsion_rest = []
    list_do_minim = []
    list_num_minim_steps = []


    cur_max_dict = dict()
    list_max_atom_cnt = []
    list_max_neigh_cnt = []
    list_max_shift_cnt = []
    list_max_body_2_cnt = []
    list_max_body_3_cnt = []
    list_max_body_4_cnt = []
    list_max_hbond_cnt = []

    num_cuts = len(all_cut_indices)
    cur_max_dict['max_atom_cnt'] = list_max_atom_cnt
    cur_max_dict['max_neigh_cnt'] = list_max_neigh_cnt
    cur_max_dict['max_shift_cnt'] = list_max_shift_cnt
    cur_max_dict['max_body_2_cnt'] = list_max_body_2_cnt
    cur_max_dict['max_body_3_cnt'] = list_max_body_3_cnt
    cur_max_dict['max_body_4_cnt'] = list_max_body_4_cnt
    cur_max_dict['max_hbond_cnt'] = list_max_hbond_cnt
    for c in range(num_cuts):

        selected_sys = [systems[i] for i in all_cut_indices[c]]
        size = len(selected_sys)
        # align atom count
        atom_counts = [s.num_atoms for s in selected_sys]
        max_num_atoms = max(atom_counts)

        if list_prev_max_dict != None and 'max_atom_cnt' in list_prev_max_dict:
            if max_num_atoms > list_prev_max_dict['max_atom_cnt'][c]:
                print('[WARNING] max_num_atoms_cur > max_num_atoms_prev')
            else:
                max_num_atoms = list_prev_max_dict['max_atom_cnt'][c]

        list_max_atom_cnt.append(max_num_atoms)

        neigh_counts = [s.local_body_2_neigh_list.shape[1] for s in selected_sys]
        max_neigh_count = max(neigh_counts)
        list_max_neigh_cnt.append(max_neigh_count)

        if list_prev_max_dict != None and 'max_neigh_cnt' in list_prev_max_dict:
            if max_neigh_count > list_prev_max_dict['max_neigh_cnt'][c]:
                print('[WARNING] max_neigh_cnt_cur > max_neigh_cnt_prev')
                #list_prev_max_dict['max_neigh_cnt'][c] = max_neigh_count
            else:
                max_neigh_count = list_prev_max_dict['max_neigh_cnt'][c]

        len_shift_combs = [len(s.all_shift_comb) for s in selected_sys]
        max_len_shift_comb = max(len_shift_combs)

        if list_prev_max_dict != None and 'max_shift_cnt' in list_prev_max_dict:
            if max_len_shift_comb > list_prev_max_dict['max_shift_cnt'][c]:
                print('[WARNING] max_shift_cnt_cur > max_shift_cnt_prev')
            else:
                max_len_shift_comb = list_prev_max_dict['max_shift_cnt'][c]

        list_max_shift_cnt.append(max_len_shift_comb)
        #allocate memory
        shift_combs = onp.zeros(shape=(size, max_len_shift_comb, 3), dtype=onp.int8)
        align_all_shift_combinations(selected_sys,shift_combs)

        all_type = onp.zeros(shape=(size, max_num_atoms), dtype=onp.int32)
        all_mask = onp.zeros(shape=(size, max_num_atoms), dtype=onp.bool)
        positions = onp.zeros(shape=(size, max_num_atoms, 3), dtype=onp.float32)
        all_body_2_neigh_list = onp.zeros(shape=(size, max_num_atoms, max_neigh_count,5), dtype=onp.int32)

        align_atom_counts_and_local_neigh(selected_sys,all_type,all_mask,positions,all_body_2_neigh_list)

        # align 2-body
        body_2_counts = [s.global_body_2_count for s in selected_sys]
        max_body_2_count = max(body_2_counts)

        if list_prev_max_dict != None and 'max_body_2_cnt' in list_prev_max_dict:
            if max_body_2_count > list_prev_max_dict['max_body_2_cnt'][c]:
                #list_prev_max_dict['max_body_2_cnt'][c] = max_body_2_count
                print('[WARNING] max_body_2_cnt_cur > max_body_2_cnt_prev')
            else:
                max_body_2_count = list_prev_max_dict['max_body_2_cnt'][c]

        list_max_body_2_cnt.append(max_body_2_count)

        all_body_2_list = onp.zeros(shape=(size,max_body_2_count,7), dtype=onp.int32)
        all_body_2_mask = onp.zeros(shape=(size,max_body_2_count), dtype=onp.bool)
        all_body_2_trip_mask = onp.zeros(shape=(size,max_body_2_count), dtype=onp.bool)

        align_body_2_inter_list(selected_sys,all_body_2_list,all_body_2_mask,all_body_2_trip_mask)

        # align 3-body
        body_3_counts = [s.global_body_3_count for s in selected_sys]
        max_body_3_count = max(body_3_counts)

        if list_prev_max_dict != None and 'max_body_3_cnt' in list_prev_max_dict:
            if max_body_3_count > list_prev_max_dict['max_body_3_cnt'][c]:
                print('[WARNING] max_body_3_cnt_cur > max_body_3_cnt_prev')
                #list_prev_max_dict['max_body_3_cnt'][c] = max_body_3_count
            else:
                max_body_3_count = list_prev_max_dict['max_body_3_cnt'][c]

        list_max_body_3_cnt.append(max_body_3_count)

        all_body_3_list = onp.zeros(shape=(size,max_body_3_count,5), dtype=onp.int32)
        all_body_3_mask = onp.zeros(shape=(size,max_body_3_count), dtype=onp.bool)
        all_body_3_shift = onp.zeros(shape=(size,max_body_3_count,2), dtype=onp.bool)

        align_body_3_inter_list(selected_sys,all_body_3_list,all_body_3_mask,all_body_3_shift)
        # align 4-body
        body_4_counts = [s.global_body_4_count for s in selected_sys]
        max_body_4_count = max(body_4_counts)

        if list_prev_max_dict != None and 'max_body_4_cnt' in list_prev_max_dict:
            if max_body_4_count > list_prev_max_dict['max_body_4_cnt'][c]:
                print('[WARNING] max_body_4_cnt_cur > max_body_4_cnt_prev')
                #list_prev_max_dict['max_body_4_cnt'][c] = max_body_4_count
            else:
                max_body_4_count = list_prev_max_dict['max_body_4_cnt'][c]


        list_max_body_4_cnt.append(max_body_4_count)

        all_body_4_list = onp.zeros(shape=(size,max_body_4_count,7), dtype=onp.int32)
        all_body_4_mask = onp.zeros(shape=(size,max_body_4_count), dtype=onp.bool)
        all_body_4_shift = onp.zeros(shape=(size,max_body_4_count,12), dtype=onp.int8)

        align_body_4_inter_list(selected_sys,all_body_4_list,all_body_4_mask,all_body_4_shift)

        hbond_counts = [s.global_hbond_count for s in selected_sys]
        max_hbond_count = max(hbond_counts)

        if list_prev_max_dict != None and 'max_hbond_cnt' in list_prev_max_dict:
            if max_hbond_count > list_prev_max_dict['max_hbond_cnt'][c]:
                print('[WARNING] max_hbond_cnt_cur > max_hbond_cnt_prev')
                #list_prev_max_dict['max_hbond_cnt'][c] = max_hbond_count
            else:
                max_hbond_count = list_prev_max_dict['max_hbond_cnt'][c]

        list_max_hbond_cnt.append(max_hbond_count)
        all_hbond_list = onp.zeros(shape=(size,max_hbond_count,7), dtype=onp.int32)
        all_hbond_mask = onp.zeros(shape=(size,max_hbond_count), dtype=onp.bool)
        all_hbond_shift = onp.zeros(shape=(size,max_hbond_count,6), dtype=onp.int8)
        align_hbond_inter_list(selected_sys,all_hbond_list,all_hbond_mask,all_hbond_shift)

        orth_matrices = onp.array([s.orth_matrix for s in selected_sys])
        is_periodic = onp.array([s.is_periodic for s in selected_sys])
        bond_rest = onp.array([s.bond_restraints for s in selected_sys])
        angle_rest = onp.array([s.angle_restraints for s in selected_sys])
        torsion_rest = onp.array([s.torsion_restraints for s in selected_sys])
        do_minim = onp.array([s.do_minimization for s in selected_sys])
        minim_steps = onp.array([s.num_min_steps for s in selected_sys])

        counts = onp.array([s.real_atom_count for s in selected_sys])

        all_total_charge = onp.array([s.total_charge for s in selected_sys])

        list_all_type.append(all_type)
        list_all_mask.append(all_mask)
        list_all_total_charge.append(all_total_charge)
        #list_all_dist_mat.append(all_dist_mat)
        list_all_body_2_list.append(all_body_2_list)
        list_all_body_2_map.append(all_body_2_mask)
        list_all_body_2_neigh_list.append(all_body_2_neigh_list)
        list_all_body_2_trip_mask.append(all_body_2_trip_mask)
        list_all_body_3_list.append(all_body_3_list)
        list_all_body_3_map.append(all_body_3_mask)
        list_all_body_3_shift.append(all_body_3_shift)
        list_all_body_4_list.append(all_body_4_list)
        list_all_body_4_map.append(all_body_4_mask)
        list_all_body_4_shift.append(all_body_4_shift)
        list_all_hbond_list.append(all_hbond_list)
        list_all_hbond_mask.append(all_hbond_mask)
        list_all_hbond_shift.append(all_hbond_shift)
        list_orth_matrices.append(orth_matrices)
        list_positions.append(positions)
        list_is_periodic.append(is_periodic)
        list_all_shift_combs.append(shift_combs)
        list_bond_rest.append(bond_rest)
        list_angle_rest.append(angle_rest)
        list_torsion_rest.append(torsion_rest)
        list_do_minim.append(do_minim)
        list_num_minim_steps.append(minim_steps)
        list_real_atom_counts.append(counts)


    cur_max_dict['max_atom_cnt'] = list_max_atom_cnt
    cur_max_dict['max_neigh_cnt'] = list_max_neigh_cnt
    cur_max_dict['max_shift_cnt'] = list_max_shift_cnt
    cur_max_dict['max_body_2_cnt'] = list_max_body_2_cnt
    cur_max_dict['max_body_3_cnt'] = list_max_body_3_cnt
    cur_max_dict['max_body_4_cnt'] = list_max_body_4_cnt
    cur_max_dict['max_hbond_cnt'] = list_max_hbond_cnt

    return     [list_all_type,
            list_all_mask,
            list_all_total_charge,
            #list_all_dist_mat,
            list_all_body_2_list,
            list_all_body_2_map,
            list_all_body_2_neigh_list,
            list_all_body_2_trip_mask,
            list_all_body_3_list,
            list_all_body_3_map,
            list_all_body_3_shift,
            list_all_body_4_list,
            list_all_body_4_map,
            list_all_body_4_shift,
            list_all_hbond_list,
            list_all_hbond_mask,
            list_all_hbond_shift,
            list_real_atom_counts,
            list_orth_matrices,
            list_positions, # should be mutable
            list_is_periodic,
            list_all_shift_combs,
            list_bond_rest,
            list_angle_rest,
            list_torsion_rest,
            list_do_minim,
            list_num_minim_steps], cur_max_dict


def gradient_list_nan_detection(grads, print_extra=True):
    total = 0
    for i,g in enumerate(grads):
        count = np.sum(np.isnan(g))
        count_inf = np.sum(np.isinf(g))
        maxx = np.max(g)
        minn =np.min(g)
        total = total + count
        if print_extra and count > 0 or count_inf > 0:
            print("Nan detected at index: {}, count: {}, count_inf: {}, shape:{}, min: {}, max:{}".format(i,count,count_inf, g.shape, minn, maxx))
    print('Total:{}'.format(total))


def read_charges(filename):
    charges = onp.loadtxt(filename)
    charges = charges.reshape(-1, 2)
    charges = charges[:, 1]
    return charges

def read_energy(filename):
    energy_vals = []
    f = open(filename)

    total = float(f.readline())
    energy_vals.append(total)
    for line in f:
        val = float(line.split()[1])
        energy_vals.append(val)

    f.close()
    return onp.array(energy_vals)


def collect_data(systems):
    import pickle
    data_list = []
    for s in systems:

        f = open('fort.3', 'w')
        f.write(s.bgf_file)
        f.close()
        os.system('/mnt/home/kaymakme/force_field_optimization/reaxFF_optim/reac')
        charges = read_charges('reax_charges.txt')
        energy_vals = read_energy('reax_energy.txt')
        data_list.append((charges,energy_vals))

    with open('data_list.pkl', 'wb') as f:
        pickle.dump(data_list, f)

def preprocess_trainset_line(line):
    # to make sure everything is nicely seperated by a space
    line = line.replace('/', ' / ')
    # it changes the weight as well, so removed
    #line = line.replace('+', ' + ')
    #line = line.replace('-', ' - ')

    return line

def filter_geo_items(systems, trainset_items):
    name_index_dict = dict()
    for i,s in enumerate(systems):
        name_index_dict[s.name] = i
    all_trainset_geo_names = set()
    for key, sub_items in trainset_items.items():
        for item in sub_items:
            if key == "ENERGY":
                names = item[0]
            else:
                names = [item[0],]
            for n in names:
                all_trainset_geo_names.add(n)
    final_systems = []
    for n in all_trainset_geo_names:
        if n not in name_index_dict:
            print(f"[INFO] {n} does not exist!")
        else:
            final_systems.append(systems[name_index_dict[n]])

    return final_systems

def read_train_set(train_in):
    f = open(train_in, 'r')
    training_items = {}
    training_items_str = {}
    energy_flag = 0
    charge_flag = 0
    geo_flag = 0
    force_flag = 0
    new_RMSG_flag = 0 # use to minimize forces
    geo2_items = []
    geo3_items = []
    geo4_items = []
    force_RMSG_items = []
    force_atom_items = []
    new_RMSG_items = []
    energy_items = []
    charge_items = []

    geo2_items_str = []
    geo3_items_str = []
    geo4_items_str = []
    force_RMSG_items_str = []
    force_atom_items_str = []
    new_RMSG_items_str = []
    energy_items_str = []
    charge_items_str = []
    for line in f:
        #print(line)
        line = line.strip()
        # ignore everything after #
        line = line.split('#', 1)[0]
        line = line.split('!', 1)[0]
        if len(line) == 0 or line.startswith("#"):
            continue
        elif line.startswith("ENERGY"):
            energy_flag = 1

        elif line.startswith("CHARGE"):
            charge_flag = 1

        elif line.startswith("GEOMETRY"):
            geo_flag = 1
        elif line.startswith('FORCES'):
            force_flag = 1

        elif line.startswith('RMSG-NEW'):
            new_RMSG_flag = 1

        elif line.startswith("ENDENERGY"):
            #training_items['ENERGY'] = energy_items
            energy_flag = 0

        elif line.startswith("ENDCHARGE"):
            #training_items['CHARGE'] = charge_items
            charge_flag = 0

        elif line.startswith("ENDGEOMETRY"):
            #training_items['GEOMETRY-2'] = geo2_items
            #training_items['GEOMETRY-3'] = geo3_items
            #training_items['GEOMETRY-4'] = geo4_items
            #training_items['FORCE-RMSG'] = force_RMSG_items

            geo_flag = 0

        elif line.startswith("ENDRMSG-NEW"):
            #training_items['RMSG-NEW'] = new_RMSG_items
            new_RMSG_flag = 0

        elif line.startswith("ENDFORCES"):
            #training_items['FORCE-ATOM'] = force_atom_items
            force_flag = 0
        elif energy_flag == 1:
            line = preprocess_trainset_line(line)
            split_line = line.split()
            num_ref_items = int((len(split_line) - 2) / 4) # w and energy + 4 items per ref. item

            name_list = []
            multiplier_list = []

            w = float(split_line[0])
            for i in range(num_ref_items):
                div = float(split_line[4 * i + 4].strip())
                mult = 1/div
                if split_line[1 + 4*i].strip() == '+':
                    multiplier_list.append(mult)
                else:
                    multiplier_list.append(-mult)


                name_list.append(split_line[4 * i + 2].strip())

            energy = float(split_line[-1])

            energy_items.append((name_list,w,multiplier_list, energy))
            energy_items_str.append(line)

        elif charge_flag == 1:
            line = preprocess_trainset_line(line)
            split_line = line.split()
            name = split_line[0].strip()
            weight = float(split_line[1])
            index = int(split_line[2])
            charge = float(split_line[3])
            charge_items.append((name,weight,index,charge))
            charge_items_str.append(line)


        elif geo_flag == 1:
            line = preprocess_trainset_line(line)
            split_line = line.split()
            name = split_line[0].strip()
            weight = float(split_line[1])
            target = float(split_line[-1])
            # 2-body
            if len(split_line) == 5:
                index1 = int(split_line[2])
                index2 = int(split_line[3])
                geo2_items.append((name,weight,index1,index2,target))
                geo2_items_str.append(line)

            # 3-body
            if len(split_line) == 6:
                index1 = int(split_line[2])
                index2 = int(split_line[3])
                index3 = int(split_line[4])
                geo3_items.append((name,weight,index1,index2,index3,target))
                geo3_items_str.append(line)
            # 4-body
            if len(split_line) == 7:
                index1 = int(split_line[2])
                index2 = int(split_line[3])
                index3 = int(split_line[4])
                index4 = int(split_line[5])
                geo4_items.append((name,weight,index1,index2,index3,index4,target))
                geo4_items_str.append(line)
            #RMSG
            if len(split_line) == 3:
                force_RMSG_items_str.append(line)
                force_RMSG_items.append((name,weight,target))

        elif force_flag == 1:
            line = preprocess_trainset_line(line)
            split_line = line.split()
            name = split_line[0].strip()
            weight = float(split_line[1])
            #force on indiv. atoms
            if len(split_line) == 6:
                index = int(split_line[2])
                f1 = float(split_line[3])
                f2 = float(split_line[4])
                f3 = float(split_line[5])
                force_atom_items.append((name,weight,index,f1,f2,f3))
                force_atom_items_str.append(line)
        elif new_RMSG_flag == 1:
            line = preprocess_trainset_line(line)
            split_line = line.split()
            name = split_line[0].strip()
            weight = float(split_line[1])
            target = float(split_line[-1])
            new_RMSG_items.append((name,weight,target))
            new_RMSG_items_str.append(line)

    if len(energy_items) > 0:
        training_items['ENERGY'] = energy_items
        training_items_str['ENERGY'] = energy_items_str

    if len(charge_items) > 0:
        training_items['CHARGE'] = charge_items
        training_items_str['CHARGE'] = charge_items_str


    if len(geo2_items) > 0:
        training_items['GEOMETRY-2'] = geo2_items
        training_items_str['GEOMETRY-2'] = geo2_items_str


    if len(geo3_items) > 0:
        training_items['GEOMETRY-3'] = geo3_items
        training_items_str['GEOMETRY-3'] = geo3_items_str


    if len(geo4_items) > 0:
        training_items['GEOMETRY-4'] = geo4_items
        training_items_str['GEOMETRY-4'] = geo4_items_str


    if len(force_RMSG_items) > 0:
        training_items['FORCE-RMSG'] = force_RMSG_items
        training_items_str['FORCE-RMSG'] = force_RMSG_items_str

    if len(force_atom_items) > 0:
        training_items['FORCE-ATOM'] = force_atom_items
        training_items_str['FORCE-ATOM'] = force_atom_items_str


    return training_items,training_items_str

def structure_energy_training_data(name_dict, training_items):
    import copy
    max_len = 5

    all_weights = []
    all_energy_vals = []

    sys_list_of_lists = []
    multip_list_of_lists = []
    for i, item in enumerate(training_items):

        name_list, w, multiplier_list, energy = item
        # deep copy not to affect the orig. data structures
        multiplier_list = copy.deepcopy(multiplier_list)
        index_list = []
        new_energy = energy
        exist = True
        for multip,name in zip(multiplier_list,name_list):
            if name not in name_dict:
                exist = False
                print("{} does not exist in the geo file, skipping!", name)
                break
        if exist:
            for multip,name in zip(multiplier_list,name_list):
                ind = name_dict[name]


                index_list.append(ind)
                # just to have fixed size length, filler ones will be zeroed out
            if len(index_list) <= max_len:
                cur_len = len(index_list)
                for _ in range(max_len - cur_len):
                    index_list.append(0)
                    multiplier_list.append(0)
                sys_list_of_lists.append(index_list)
                multip_list_of_lists.append(multiplier_list)
            all_weights.append(w)
            all_energy_vals.append(new_energy)
    return onp.array(sys_list_of_lists,dtype=onp.int32), onp.array(multip_list_of_lists,dtype=TYPE), onp.array(all_weights,dtype=TYPE), onp.array(all_energy_vals,dtype=TYPE)

def structure_charge_training_data(name_dict,training_items):
    all_weights = []
    all_charge_vals = []

    sys_index_list = []
    atom_index_list = []
    for i, item in enumerate(training_items):
        name,weight,atom_index,charge = item
        if name in name_dict:
            ind = name_dict[name]
        else:
            print("{} does not exist in the geo file, skipping!", name)
            continue
        sys_index_list.append(ind)
        all_weights.append(weight)
        atom_index_list.append(atom_index - 1)
        all_charge_vals.append(charge)

    return onp.array(sys_index_list,dtype=onp.int32), onp.array(atom_index_list,dtype=onp.int32), onp.array(all_weights,dtype=TYPE), onp.array(all_charge_vals,dtype=TYPE)

def structure_geo2_training_data(name_dict,training_items):
    all_weights = []
    all_target_vals = []
    sys_index_list = []
    atom_index_list = []
    for i, item in enumerate(training_items):
        name,weight,atom_index1,atom_index2,target = item
        if name in name_dict:
            ind = name_dict[name]
        else:
            print("{} does not exist in the geo file, skipping!", name)
            continue
        sys_index_list.append(ind)
        all_weights.append(weight)
        atom_index_list.append((atom_index1 - 1, atom_index2 - 1))
        all_target_vals.append(target)

    return onp.array(sys_index_list,dtype=onp.int32), onp.array(atom_index_list,dtype=onp.int32), onp.array(all_weights,dtype=TYPE), onp.array(all_target_vals,dtype=TYPE)

def structure_geo3_training_data(name_dict,training_items):
    all_weights = []
    all_target_vals = []
    sys_index_list = []
    atom_index_list = []
    for i, item in enumerate(training_items):
        name,weight,atom_index1,atom_index2,atom_index3,target = item
        if name in name_dict:
            ind = name_dict[name]
        else:
            print("{} does not exist in the geo file, skipping!", name)
            continue
        sys_index_list.append(ind)
        all_weights.append(weight)
        atom_index_list.append((atom_index1 - 1, atom_index2 - 1,atom_index3 - 1))
        all_target_vals.append(target)

    return onp.array(sys_index_list,dtype=onp.int32), onp.array(atom_index_list,dtype=onp.int32), onp.array(all_weights,dtype=TYPE), onp.array(all_target_vals,dtype=TYPE)

def structure_geo4_training_data(name_dict,training_items):
    all_weights = []
    all_target_vals = []
    sys_index_list = []
    atom_index_list = []
    for i, item in enumerate(training_items):
        name,weight,atom_index1,atom_index2,atom_index3,atom_index4,target = item
        if name in name_dict:
            ind = name_dict[name]
        else:
            print("{} does not exist in the geo file, skipping!", name)
            continue
        sys_index_list.append(ind)
        all_weights.append(weight)
        atom_index_list.append((atom_index1 - 1, atom_index2 - 1,atom_index3 - 1,atom_index4 - 1))
        all_target_vals.append(target)

    return onp.array(sys_index_list,dtype=onp.int32), onp.array(atom_index_list,dtype=onp.int32), onp.array(all_weights,dtype=TYPE), onp.array(all_target_vals,dtype=TYPE)

def structure_geo_RMSG_training_data(name_dict,training_items):
    sys_index_list = []
    all_weights = []
    all_target_vals = []
    for i, item in enumerate(training_items):
        name,weight,target = item
        if name in name_dict:
            ind = name_dict[name]
        else:
            print("{} does not exist in the geo file, skipping!", name)
            continue
        sys_index_list.append(ind)
        all_weights.append(weight)
        all_target_vals.append(target)
    return onp.array(sys_index_list,dtype=onp.int32), onp.array(all_weights,dtype=TYPE), onp.array(all_target_vals,dtype=TYPE)

def structure_force_training_data(name_dict,training_items):
    all_weights = []
    all_target_vals = []
    sys_index_list = []
    atom_index_list = []
    for i, item in enumerate(training_items):
        name,weight,atom_index,f1,f2,f3 = item
        if name in name_dict:
            ind = name_dict[name]
        else:
            print("{} does not exist in the geo file, skipping!", name)
            continue
        sys_index_list.append(ind)
        all_weights.append(weight)
        atom_index_list.append(atom_index-1)
        all_target_vals.append((f1,f2,f3))

    return onp.array(sys_index_list,dtype=onp.int32), onp.array(atom_index_list,dtype=onp.int32), onp.array(all_weights,dtype=TYPE), onp.array(all_target_vals,dtype=TYPE)

def structure_training_data(sim_systems, all_training_items):
    name_dict = {}
    for i,s in enumerate(sim_systems):
        name_dict[s.name] = i

    structured_training_data = dict()
    if 'ENERGY' in all_training_items and len(all_training_items['ENERGY']) > 0:
        energy_sys_list_of_lists, energy_multip_list_of_lists, energy_all_weights, energy_all_energy_vals =  structure_energy_training_data(name_dict, all_training_items['ENERGY'])
        structured_training_data['ENERGY'] = (energy_sys_list_of_lists, energy_multip_list_of_lists, energy_all_weights, energy_all_energy_vals)

    if 'CHARGE' in all_training_items and len(all_training_items['CHARGE']) > 0:
        chg_sys_index_list, chg_atom_index_list, chg_all_weights, chg_all_charge_vals = structure_charge_training_data(name_dict,all_training_items['CHARGE'])
        structured_training_data['CHARGE'] = (chg_sys_index_list, chg_atom_index_list, chg_all_weights, chg_all_charge_vals)

    if 'GEOMETRY-2' in all_training_items and len(all_training_items['GEOMETRY-2']) > 0:
        geo2_sys_index_list, geo2_atom_index_list, geo2_all_weights, geo2_all_target_vals = structure_geo2_training_data(name_dict,all_training_items['GEOMETRY-2'])
        structured_training_data['GEOMETRY-2'] = (geo2_sys_index_list, geo2_atom_index_list, geo2_all_weights, geo2_all_target_vals)

    if 'GEOMETRY-3' in all_training_items and len(all_training_items['GEOMETRY-3']) > 0:
        geo3_sys_index_list, geo3_atom_index_list, geo3_all_weights, geo3_all_target_vals = structure_geo3_training_data(name_dict,all_training_items['GEOMETRY-3'])
        structured_training_data['GEOMETRY-3'] = (geo3_sys_index_list, geo3_atom_index_list, geo3_all_weights, geo3_all_target_vals)

    if 'GEOMETRY-4' in all_training_items and len(all_training_items['GEOMETRY-4']) > 0:
        geo4_sys_index_list, geo4_atom_index_list, geo4_all_weights, geo4_all_target_vals = structure_geo4_training_data(name_dict,all_training_items['GEOMETRY-4'])
        structured_training_data['GEOMETRY-4'] = (geo4_sys_index_list, geo4_atom_index_list, geo4_all_weights, geo4_all_target_vals)

    if 'FORCE-RMSG' in all_training_items and len(all_training_items['FORCE-RMSG']) > 0:
        force_sys_index_list, force_all_weights, force_all_target_vals = structure_geo_RMSG_training_data(name_dict,all_training_items['FORCE-RMSG'])
        structured_training_data['FORCE-RMSG'] = (force_sys_index_list, force_all_weights, force_all_target_vals)
    if 'RMSG-NEW' in all_training_items and len(all_training_items['RMSG-NEW']) > 0:
        force_sys_index_list, force_all_weights, force_all_target_vals = structure_geo_RMSG_training_data(name_dict,all_training_items['RMSG-NEW'])
        structured_training_data['RMSG-NEW'] = (force_sys_index_list, force_all_weights, force_all_target_vals)
    if 'FORCE-ATOM' in all_training_items and len(all_training_items['FORCE-ATOM']) > 0:
        force_sys_index_list, force_atom_index_list, force_all_weights, force_all_target_vals = structure_force_training_data(name_dict,all_training_items['FORCE-ATOM'])
        structured_training_data['FORCE-ATOM'] = (force_sys_index_list, force_atom_index_list, force_all_weights, force_all_target_vals)

    return structured_training_data



def parse_and_save_force_field(old_ff_file, new_ff_file,force_field):
    output = ""
    f = open(old_ff_file, 'r')
    line = f.readline()
    output = output + line
    header = line.strip()

    line = f.readline()
    output = output + line
    num_params = int(line.strip().split()[0])
    general_params = np.zeros(shape=(num_params,1), dtype=np.float64)
    ff = force_field
    for i in range(num_params):
        line = f.readline()
        line = list(line)
        #-------------------------------------------------------------
        if i == 0:
            line[:10] = "{:10.4f}".format(ff.over_coord1[0])  #overcoord1
        if i == 1:
            line[:10] = "{:10.4f}".format(ff.over_coord2[0]) #overcoord2
        #-------------------------------------------------------------

        #-------------------------------------------------------------
        if i == 3:
            line[:10] = "{:10.4f}".format(ff.trip_stab4[0])  #trip_stab4
        if i == 4:
            line[:10] = "{:10.4f}".format(ff.trip_stab5[0]) #trip_stab5
        if i == 7:
            line[:10] = "{:10.4f}".format(ff.trip_stab8[0])  #trip_stab8
        if i == 10:
            line[:10] = "{:10.4f}".format(ff.trip_stab11[0]) #trip_stab11
        #-------------------------------------------------------------
        #valency related parameters
        if i == 2:
            line[:10] = "{:10.4f}".format(ff.val_par3[0])  #val_par3
        if i == 14:
            line[:10] = "{:10.4f}".format(ff.val_par15[0]) #val_par15
        if i == 15:
            line[:10] = "{:10.4f}".format(ff.par_16[0])  #par_16
        if i == 16:
            line[:10] = "{:10.4f}".format(ff.val_par17[0]) #val_par17
        if i == 17:
            line[:10] = "{:10.4f}".format(ff.val_par18[0])  #val_par18
        if i == 19:
            line[:10] = "{:10.4f}".format(ff.val_par20[0]) #val_par20
        if i == 20:
            line[:10] = "{:10.4f}".format(ff.val_par21[0])  #val_par21
        if i == 30:
            line[:10] = "{:10.4f}".format(ff.val_par31[0]) #val_par31
        if i == 33:
            line[:10] = "{:10.4f}".format(ff.val_par34[0])  #val_par34
        if i == 38:
            line[:10] = "{:10.4f}".format(ff.val_par39[0]) #val_par39
        #-------------------------------------------------------------

        #-------------------------------------------------------------
        #over-under coord.
        if i == 5:
            line[:10] = "{:10.4f}".format(ff.par_6[0])  #par_6
        if i == 6:
            line[:10] = "{:10.4f}".format(ff.par_7[0]) #par_7
        if i == 8:
            line[:10] = "{:10.4f}".format(ff.par_9[0])  #par_9
        if i == 9:
            line[:10] = "{:10.4f}".format(ff.par_10[0]) #par_10
        if i == 31:
            line[:10] = "{:10.4f}".format(ff.par_32[0])  #par_32
        if i == 32:
            line[:10] = "{:10.4f}".format(ff.par_33[0]) #par_33

        #-------------------------------------------------------------
        #torsion
        if i == 23:
            line[:10] = "{:10.4f}".format(ff.par_24[0])  #par_24
        if i == 24:
            line[:10] = "{:10.4f}".format(ff.par_25[0]) #par_25
        if i == 25:
            line[:10] = "{:10.4f}".format(ff.par_26[0])  #par_26
        if i == 27:
            line[:10] = "{:10.4f}".format(ff.par_28[0]) #par_28

        #-------------------------------------------------------------
        # vdw
        if i == 28:
            line[:10] = "{:10.4f}".format(ff.vdw_shiedling[0]) #vdw_shiedling
        output = output + ''.join(line)

    line = f.readline()
    output = output + line

    num_atom_types = int(line.strip().split()[0])
    # skip 3 lines of comment
    output = output + f.readline()
    output = output + f.readline()
    output = output + f.readline()

    atom_names = []
    line_ctr = 0
    for i in range(num_atom_types):
        # first line
        line = f.readline()
        line = list(line)
        if i in MY_ATOM_INDICES:
            line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.rat[i]) #rat - rob1
            line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.rvdw[i]) #rvdw
            line[3 + 9 * 4:3 + 9 * 5] = "{:9.4f}".format(ff.eps[i]) #eps
            line[3 + 9 * 5:3 + 9 * 6] = "{:9.4f}".format(ff.gamma[i]) #gamma
            line[3 + 9 * 6:3 + 9 * 7] = "{:9.4f}".format(ff.rapt[i]) #rapt - rob2
            line[3 + 9 * 7:3 + 9 * 8] = "{:9.4f}".format(ff.stlp[i]) #stlp

        output = output + ''.join(line)

        # second line
        line = f.readline()
        line = list(line)
        if i in MY_ATOM_INDICES:
            line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.alf[i]) #alf
            line[3 + 9 * 1:3 + 9 * 2] = "{:9.4f}".format(ff.vop[i]) #vop
            line[3 + 9 * 2:3 + 9 * 3] = "{:9.4f}".format(ff.valf[i]) #valf
            line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.valp1[i]) #valp1
            line[3 + 9 * 5:3 + 9 * 6] = "{:9.4f}".format(ff.electronegativity[i]) #electronegativity
            line[3 + 9 * 6:3 + 9 * 7] = "{:9.4f}".format(ff.idempotential[i]) #idempotential

        output = output + ''.join(line)
        # third line
        line = f.readline()
        line = list(line)
        if i in MY_ATOM_INDICES:
            line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.vnq[i]) #vnq - rob3
            line[3 + 9 * 1:3 + 9 * 2] = "{:9.4f}".format(ff.vlp1[i]) #vlp1
            line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.bo131[i]) #bo131
            line[3 + 9 * 4:3 + 9 * 5] = "{:9.4f}".format(ff.bo132[i]) #bo132
            line[3 + 9 * 5:3 + 9 * 6] = "{:9.4f}".format(ff.bo133[i]) #bo133

        output = output + ''.join(line)

        # fourth line
        line = f.readline()
        line = list(line)
        if i in MY_ATOM_INDICES:
            line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.vovun[i])
            line[3 + 9 * 1:3 + 9 * 2] = "{:9.4f}".format(ff.vval1[i])
            line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.vval3[i])
            line[3 + 9 * 4:3 + 9 * 5] = "{:9.4f}".format(ff.vval4[i])

        output = output + ''.join(line)



    line = f.readline()  # num_bonds
    output = output + line

    line = line.strip()
    num_bonds = int(line.split()[0])
    output = output + f.readline() # skip next line (comment)
    for _ in range(num_bonds):
        # first line
        line = f.readline()
        tmp = line.strip().split()
        line = list(line)
        i = int(tmp[0]) - 1 # index starts at 0
        j = int(tmp[1]) - 1

        if i in MY_ATOM_INDICES and j in MY_ATOM_INDICES:
            line[6 + 9 * 0:6 + 9 * 1] = "{:9.4f}".format(ff.de1[i,j])
            line[6 + 9 * 1:6 + 9 * 2] = "{:9.4f}".format(ff.de2[i,j])
            line[6 + 9 * 2:6 + 9 * 3] = "{:9.4f}".format(ff.de3[i,j])
            line[6 + 9 * 3:6 + 9 * 4] = "{:9.4f}".format(ff.psi[i,j])
            line[6 + 9 * 4:6 + 9 * 5] = "{:9.4f}".format(ff.pdo[i,j])
            line[6 + 9 * 5:6 + 9 * 6] = "{:9.4f}".format(ff.v13cor[i,j])
            line[6 + 9 * 6:6 + 9 * 7] = "{:9.4f}".format(ff.popi[i,j])
            line[6 + 9 * 7:6 + 9 * 8] = "{:9.4f}".format(ff.vover[i,j])
        #print(''.join(line))
        output = output + ''.join(line)
        # second line
        line = f.readline()
        line = list(line)

        if i in MY_ATOM_INDICES and j in MY_ATOM_INDICES:
            line[6 + 9 * 0:6 + 9 * 1] = "{:9.4f}".format(ff.psp[i,j])
            line[6 + 9 * 1:6 + 9 * 2] = "{:9.4f}".format(ff.pdp[i,j])
            line[6 + 9 * 2:6 + 9 * 3] = "{:9.4f}".format(ff.ptp[i,j])
            line[6 + 9 * 4:6 + 9 * 5] = "{:9.4f}".format(ff.bop1[i,j])
            line[6 + 9 * 5:6 + 9 * 6] = "{:9.4f}".format(ff.bop2[i,j])
            line[6 + 9 * 6:6 + 9 * 7] = "{:9.4f}".format(ff.ovc[i,j])
        #print(''.join(line))
        output = output + ''.join(line)
    line = f.readline()  # num_off_diag
    output = output + line

    line = line.strip()
    num_off_diag = int(line.split()[0])

    for _ in range(num_off_diag):
        # first line
        # first line
        line = f.readline()
        tmp = line.strip().split()
        line = list(line)
        i = int(tmp[0]) - 1 # index starts at 0
        j = int(tmp[1]) - 1

        if i in MY_ATOM_INDICES and j in MY_ATOM_INDICES:
            line[6 + 9 * 0:6 + 9 * 1] = "{:9.4f}".format(ff.p2co_off[i,j])
            line[6 + 9 * 1:6 + 9 * 2] = "{:9.4f}".format(ff.p1co_off[i,j])  # was /2
            line[6 + 9 * 2:6 + 9 * 3] = "{:9.4f}".format(ff.p3co_off[i,j])
            line[6 + 9 * 3:6 + 9 * 4] = "{:9.4f}".format(ff.rob1_off[i,j])
            line[6 + 9 * 4:6 + 9 * 5] = "{:9.4f}".format(ff.rob2_off[i,j])
            line[6 + 9 * 5:6 + 9 * 6] = "{:9.4f}".format(ff.rob3_off[i,j])

        output = output + ''.join(line)

    #valency angle parameters
    line = f.readline()  # num_val_params
    output = output + line

    line = line.strip()
    num_val_params = int(line.split()[0])

    for _ in range(num_val_params):
        # first line
        line = f.readline()
        tmp = line.strip().split()
        line = list(line)
        i = int(tmp[0]) - 1 # index starts at 0
        j = int(tmp[1]) - 1
        k = int(tmp[2]) - 1

        if i in MY_ATOM_INDICES and j in MY_ATOM_INDICES and k in MY_ATOM_INDICES:
            line[9 + 9 * 0:9 + 9 * 1] = "{:9.4f}".format(ff.th0[i,j,k])
            line[9 + 9 * 1:9 + 9 * 2] = "{:9.4f}".format(ff.vka[i,j,k])
            line[9 + 9 * 2:9 + 9 * 3] = "{:9.4f}".format(ff.vka3[i,j,k])
            line[9 + 9 * 3:9 + 9 * 4] = "{:9.4f}".format(ff.vka8[i,j,k])
            line[9 + 9 * 4:9 + 9 * 5] = "{:9.4f}".format(ff.vkac[i,j,k])
            line[9 + 9 * 5:9 + 9 * 6] = "{:9.4f}".format(ff.vkap[i,j,k])
            line[9 + 9 * 6:9 + 9 * 7] = "{:9.4f}".format(ff.vval2[i,j,k])
        output = output + ''.join(line)

    #torsion parameters
    line = f.readline()  # num_tors_params
    output = output + line

    line = line.strip()
    num_tors_params = int(line.split()[0])

    for _ in range(num_tors_params):
        # first line
        line = f.readline()
        tmp = line.strip().split()
        line = list(line)
        i1 = int(tmp[0]) - 1 # index starts at 0
        i2 = int(tmp[1]) - 1
        i3 = int(tmp[2]) - 1
        i4 = int(tmp[3]) - 1

        if i1 in MY_ATOM_INDICES and i2 in MY_ATOM_INDICES and i3 in MY_ATOM_INDICES and i4 in MY_ATOM_INDICES and i1 != -1 and i4 != -1:
            line[12 + 9 * 0:12 + 9 * 1] = "{:9.4f}".format(ff.v1[i1,i2,i3,i4])
            line[12 + 9 * 1:12 + 9 * 2] = "{:9.4f}".format(ff.v2[i1,i2,i3,i4])
            line[12 + 9 * 2:12 + 9 * 3] = "{:9.4f}".format(ff.v3[i1,i2,i3,i4])
            line[12 + 9 * 3:12 + 9 * 4] = "{:9.4f}".format(ff.v4[i1,i2,i3,i4])
            line[12 + 9 * 4:12 + 9 * 5] = "{:9.4f}".format(ff.vconj[i1,i2,i3,i4])

        if i2 in MY_ATOM_INDICES and i3 in MY_ATOM_INDICES and i1 == -1 and i4 == -1:
            sel_ind = force_field.total_num_atom_types - 1
            line[12 + 9 * 0:12 + 9 * 1] = "{:9.4f}".format(ff.v1[sel_ind,i2,i3,sel_ind])
            line[12 + 9 * 1:12 + 9 * 2] = "{:9.4f}".format(ff.v2[sel_ind,i2,i3,sel_ind])
            line[12 + 9 * 2:12 + 9 * 3] = "{:9.4f}".format(ff.v3[sel_ind,i2,i3,sel_ind])
            line[12 + 9 * 3:12 + 9 * 4] = "{:9.4f}".format(ff.v4[sel_ind,i2,i3,sel_ind])
            line[12 + 9 * 4:12 + 9 * 5] = "{:9.4f}".format(ff.vconj[sel_ind,i2,i3,sel_ind])
        output = output + ''.join(line)

    # hbond parameters
    #torsion parameters
    line = f.readline()  # num_tors_params
    output = output + line

    line = line.strip()
    num_hbond_params = int(line.split()[0])

    for i in range(num_hbond_params):
        line = f.readline()
        tmp = line.strip().split()
        line = list(line)
        i1 = int(tmp[0]) - 1
        i2 = int(tmp[1]) - 1
        i3 = int(tmp[2]) -1
        if i1 in MY_ATOM_INDICES and i2 in MY_ATOM_INDICES and i3 in MY_ATOM_INDICES:
            line[9 + 9 * 0:9 + 9 * 1] = "{:9.4f}".format(ff.rhb[i1,i2,i3])
            line[9 + 9 * 1:9 + 9 * 2] = "{:9.4f}".format(ff.dehb[i1,i2,i3])
            line[9 + 9 * 2:9 + 9 * 3] = "{:9.4f}".format(ff.vhb1[i1,i2,i3])
            line[9 + 9 * 3:9 + 9 * 4] = "{:9.4f}".format(ff.vhb2[i1,i2,i3])
        output = output + ''.join(line)



    # need to append some extra lines because of 0 values
    for line in f:
       output = output + line

    file_new = open(new_ff_file,"w")
    file_new.write(output)
    file_new.close()

    f.close()


def parse_force_field(force_field_file, cutoff2):


    f = open(force_field_file, 'r')
    header = f.readline().strip()

    num_params = int(f.readline().strip().split()[0])
    general_params = onp.zeros(shape=(num_params,1), dtype=np.float64)

    body_3_indices_src = [[],[],[]]
    body_3_indices_dst = [[],[],[]]
    body_4_indices_src = [[],[],[],[]]
    body_4_indices_dst = [[],[],[],[]]

    for i in range(num_params):
        line = f.readline().strip()
        #to seperate the comment
        line = line.replace('!', ' ! ')
        general_params[i] = float(line.split()[0])
    num_atom_types = int(f.readline().strip().split()[0])

    force_field = ForceField(cutoff2=cutoff2) # + 1 for the filler type

    force_field.low_tap_rad = general_params[11] # nondiff
    force_field.up_tap_rad = general_params[12] # nondiff
    force_field.vdw_shiedling = general_params[28] # index=7

    force_field.params_to_indices[(1,29,1)] = (7, 0)

    force_field.cutoff = general_params[29] * 0.01

    force_field.over_coord1 = general_params[0] # index=31
    force_field.over_coord2 = general_params[1] # index=32

    force_field.params_to_indices[(1,1,1)] = (31, 0)
    force_field.params_to_indices[(1,2,1)] = (32, 0)

    force_field.trip_stab4 = general_params[3] # index=22
    force_field.trip_stab5 = general_params[4] # index=23
    force_field.trip_stab8 = general_params[7] # index=24
    force_field.trip_stab11 = general_params[10] # index=25

    force_field.params_to_indices[(1,4,1)] = (22, 0)
    force_field.params_to_indices[(1,5,1)] = (23, 0)
    force_field.params_to_indices[(1,8,1)] = (24, 0)
    force_field.params_to_indices[(1,11,1)] = (25, 0)

    force_field.val_par3 = general_params[2]
    force_field.val_par15 = general_params[14]
    force_field.par_16 = general_params[15]
    force_field.val_par17 = general_params[16]
    force_field.val_par18 = general_params[17]
    force_field.val_par20 = general_params[19]
    force_field.val_par21 = general_params[20]
    force_field.val_par22 = general_params[21]
    force_field.val_par31 = general_params[30]
    force_field.val_par34 = general_params[33]
    force_field.val_par39 = general_params[38]

    force_field.params_to_indices[(1,3,1)] = (44, 0)
    force_field.params_to_indices[(1,15,1)] = (45, 0)
    force_field.params_to_indices[(1,16,1)] = (55, 0)
    force_field.params_to_indices[(1,17,1)] = (46, 0)
    force_field.params_to_indices[(1,18,1)] = (47, 0)
    force_field.params_to_indices[(1,20,1)] = (48, 0)
    force_field.params_to_indices[(1,21,1)] = (49, 0)
    force_field.params_to_indices[(1,22,1)] = (50, 0)
    force_field.params_to_indices[(1,31,1)] = (51, 0)
    force_field.params_to_indices[(1,34,1)] = (52, 0)
    force_field.params_to_indices[(1,39,1)] = (53, 0)

    # over under
    force_field.par_6 = general_params[5]
    force_field.par_7 = general_params[6]
    force_field.par_9 = general_params[8]
    force_field.par_10 = general_params[9]
    force_field.par_32 = general_params[31]
    force_field.par_33 = general_params[32]

    force_field.params_to_indices[(1,6,1)] = (60, 0)
    force_field.params_to_indices[(1,7,1)] = (61, 0)
    force_field.params_to_indices[(1,9,1)] = (62, 0)
    force_field.params_to_indices[(1,10,1)] = (63, 0)
    force_field.params_to_indices[(1,32,1)] = (64, 0)
    force_field.params_to_indices[(1,33,1)] = (65, 0)

    # torsion par_24,par_25, par_26,par_28
    force_field.par_24 = general_params[23]
    force_field.par_25 = general_params[24]
    force_field.par_26 = general_params[25]
    force_field.par_28 = general_params[27]

    force_field.params_to_indices[(1,24,1)] = (71, 0)
    force_field.params_to_indices[(1,25,1)] = (72, 0)
    force_field.params_to_indices[(1,26,1)] = (73, 0)
    force_field.params_to_indices[(1,28,1)] = (74, 0)


    # skip 3 lines of comment
    f.readline()
    f.readline()
    f.readline()

    atom_names = []
    line_ctr = 0
    for i in range(num_atom_types):

        # first line
        line = f.readline().strip()
        split_line = line.split()
        atom_names.append(str(split_line[0]))
        if i in MY_ATOM_INDICES:
            force_field.name_2_index[atom_names[i]] = i
            force_field.rat[i] = float(split_line[1])
            force_field.aval[i] = float(split_line[2])
            force_field.amas[i] = float(split_line[3])
            force_field.rvdw[i] = float(split_line[4]) #vdw
            force_field.eps[i] = float(split_line[5]) #vdw
            force_field.gamma[i] = float(split_line[6]) #coulomb
            force_field.rapt[i] = float(split_line[7])
            force_field.stlp[i] = float(split_line[8]) #valency


            force_field.params_to_indices[(2,i+1,1)] = (75, i) #rob1
            force_field.params_to_indices[(2,i+1,2)] = (26, i) # aval
            force_field.params_to_indices[(2,i+1,3)] = (56, i) #amas
            force_field.params_to_indices[(2,i+1,4)] = (78, i) #rvdw p1co
            force_field.params_to_indices[(2,i+1,5)] = (79, i) #eps p2co
            force_field.params_to_indices[(2,i+1,6)] = (0, i) #gamma
            force_field.params_to_indices[(2,i+1,7)] = (76,i) # rob2
            force_field.params_to_indices[(2,i+1,8)] = (34, i) #stlp



        # second line
        line = f.readline().strip()
        split_line = line.split()
        if i in MY_ATOM_INDICES:
            force_field.alf[i] = float(split_line[0]) #vdw
            force_field.vop[i] = float(split_line[1]) #vdw
            force_field.valf[i] = float(split_line[2]) # valency
            force_field.valp1[i] = float(split_line[3]) #over-under coord

            force_field.electronegativity[i] = float(split_line[5]) #coulomb
            force_field.idempotential[i] = float(split_line[6]) # eta will be mult. by 2
            force_field.nphb[i] = int(float(split_line[7])) # needed for hbond #needed to find acceptor-donor

            force_field.params_to_indices[(2,i+1,9)] = (80, i) #alf p3co
            force_field.params_to_indices[(2,i+1,10)] = (6, i)
            force_field.params_to_indices[(2,i+1,11)] = (33, i)
            force_field.params_to_indices[(2,i+1,12)] = (58, i)
            force_field.params_to_indices[(2,i+1,14)] = (2, i)
            force_field.params_to_indices[(2,i+1,15)] = (1, i)
        # third line
        line = f.readline().strip()
        split_line = line.split()
        if i in MY_ATOM_INDICES:
            force_field.vnq[i] = float(split_line[0])
            force_field.vlp1[i] = float(split_line[1])

            force_field.bo131[i] = float(split_line[3])
            force_field.bo132[i] = float(split_line[4])
            force_field.bo133[i] = float(split_line[5])

            force_field.params_to_indices[(2,i+1,17)] = (77, i) #rob3
            force_field.params_to_indices[(2,i+1,18)] = (54, i)
            force_field.params_to_indices[(2,i+1,20)] = (28, i)
            force_field.params_to_indices[(2,i+1,21)] = (29, i)
            force_field.params_to_indices[(2,i+1,22)] = (30, i)

        # fourth line
        line = f.readline().strip()
        split_line = line.split()
        if i in MY_ATOM_INDICES:
            force_field.vovun[i] = float(split_line[0]) #over-under coord
            force_field.vval1[i] = float(split_line[1])
            force_field.vval3[i] = float(split_line[3])
            force_field.vval4[i] = float(split_line[4])

            force_field.params_to_indices[(2,i+1,25)] = (59, i) #vovun
            force_field.params_to_indices[(2,i+1,26)] = (35,i)
            force_field.params_to_indices[(2,i+1,28)] = (27, i)
            force_field.params_to_indices[(2,i+1,29)] = (37, i)

            if force_field.amas[i] < 21.0:
                force_field.vval3[i] = force_field.valf[i]

    # vdw related parameters
    '''
    force_field.p1co = onp.sqrt(4.0 * force_field.rvdw.reshape(-1,1).dot(force_field.rvdw.reshape(1,-1)))
    force_field.p2co = onp.sqrt(force_field.eps.reshape(-1,1).dot(force_field.eps.reshape(1,-1)))
    force_field.p3co = onp.sqrt(force_field.alf.reshape(-1,1).dot(force_field.alf.reshape(1,-1)))

    # bond order related parameters
    for i in range(TOTAL_ATOM_TYPES):
        for j in range(i,TOTAL_ATOM_TYPES):
            if i in MY_ATOM_INDICES and j in MY_ATOM_INDICES:
                if force_field.rat[i] > 0 and force_field.rat[j] > 0:
                    force_field.rob1[i,j] = 0.5 * (force_field.rat[i] + force_field.rat[j])
                    #force_field.rob1[j,i] = 0.5 * (force_field.rat[i] + force_field.rat[j])
                if force_field.rapt[i] > 0 and force_field.rapt[j] > 0:
                    force_field.rob2[i,j] = 0.5 * (force_field.rapt[i] + force_field.rapt[j])
                    #force_field.rob2[j,i] = 0.5 * (force_field.rapt[i] + force_field.rapt[j])
                if force_field.vnq[i] > 0 and force_field.vnq[j] > 0:
                    force_field.rob3[i,j] = 0.5 * (force_field.vnq[i] + force_field.vnq[j])
                    #force_field.rob3[j,i] = 0.5 * (force_field.vnq[i] + force_field.vnq[j])
    '''
    line = f.readline().strip()
    num_bonds = int(line.split()[0])
    f.readline() # skip next line (comment)
    for b in range(num_bonds):
        # first line
        line = f.readline().strip()
        split_line = line.split()
        i = int(split_line[0]) - 1 # index starts at 0
        j = int(split_line[1]) - 1

        if i in MY_ATOM_INDICES and j in MY_ATOM_INDICES:
            force_field.bond_params_mask[i,j] = 1
            force_field.bond_params_mask[j,i] = 1


            force_field.de1[i,j] = float(split_line[2])
            force_field.de2[i,j] = float(split_line[3])
            force_field.de3[i,j] = float(split_line[4])
            '''
            force_field.de1[j,i] = float(split_line[2])
            force_field.de2[j,i] = float(split_line[3])
            force_field.de3[j,i] = float(split_line[4])
            '''
            force_field.psi[i,j] = float(split_line[5])
            force_field.pdo[i,j] = float(split_line[6])
            force_field.v13cor[i,j] = float(split_line[7])
            force_field.popi[i,j] = float(split_line[8])
            force_field.vover[i,j] = float(split_line[9])
            '''
            force_field.psi[j,i] = float(split_line[5])
            force_field.pdo[j,i] = float(split_line[6])
            force_field.popi[j,i] = float(split_line[8])
            force_field.vover[j,i] = float(split_line[9])
            '''
            force_field.v13cor[j,i] = float(split_line[7])

            force_field.params_to_indices[(3,b+1, 1)] = (17, (i, j))
            force_field.params_to_indices[(3,b+1, 2)] = (18, (i, j))
            force_field.params_to_indices[(3,b+1, 3)] = (19, (i, j))
            force_field.params_to_indices[(3,b+1, 4)] = (21, (i, j))
            force_field.params_to_indices[(3,b+1, 5)] = (14, (i, j))
            force_field.params_to_indices[(3,b+1, 7)] = (13, (i, j))
            force_field.params_to_indices[(3,b+1, 8)] = (57, (i, j))

        # second line
        line = f.readline().strip()
        split_line = line.split()
        if i in MY_ATOM_INDICES and j in MY_ATOM_INDICES:
            force_field.psp[i,j] = float(split_line[0])
            force_field.pdp[i,j] = float(split_line[1])
            force_field.ptp[i,j] = float(split_line[2])
            force_field.bop1[i,j] = float(split_line[4])
            force_field.bop2[i,j] = float(split_line[5])
            force_field.ovc[i,j] = float(split_line[6])
            #force_field.vuncor[i,j] = float(split_line[7])
            '''
            force_field.psp[j,i] = float(split_line[0])
            force_field.pdp[j,i] = float(split_line[1])
            force_field.ptp[j,i] = float(split_line[2])
            force_field.bop1[j,i] = float(split_line[4])
            force_field.bop2[j,i] = float(split_line[5])
            '''
            force_field.ovc[j,i] = float(split_line[6])
            #force_field.vuncor[j,i] = float(split_line[7])

            force_field.params_to_indices[(3,b+1, 9)] = (20, (i, j))
            force_field.params_to_indices[(3,b+1, 10)] = (12, (i, j))
            force_field.params_to_indices[(3,b+1, 11)] = (11, (i, j))
            force_field.params_to_indices[(3,b+1, 13)] = (15, (i, j))
            force_field.params_to_indices[(3,b+1, 14)] = (16, (i, j))


    line = f.readline().strip()
    num_off_diag = int(line.split()[0])

    for i in range(num_off_diag):
        line = f.readline().strip()
        split_line = line.split()
        nodm1 = int(split_line[0])
        nodm2 = int(split_line[1])
        deodmh = float(split_line[2])
        rodmh = float(split_line[3])
        godmh = float(split_line[4])
        rsig = float(split_line[5])
        rpi = float(split_line[6])
        rpi2 = float(split_line[7])
        #TODO: handle the mapping of the "params" later
        nodm1 = nodm1 - 1 #index starts from 0
        nodm2 = nodm2 - 1 #index starts from 0
        if nodm1 in MY_ATOM_INDICES and nodm2 in MY_ATOM_INDICES:
            if rsig > 0 and force_field.rat[nodm1] > 0 and force_field.rat[nodm2] > 0:
                force_field.rob1_off[nodm1,nodm2] = rsig
                force_field.rob1_off_mask[nodm1,nodm2] = 1
                #force_field.rob1[nodm2,nodm1] = rsig
                force_field.params_to_indices[(4,i+1,4)] = (81, (nodm1,nodm2))

            if rpi > 0 and force_field.rapt[nodm1] > 0 and force_field.rapt[nodm2] > 0:
                force_field.rob2_off[nodm1,nodm2] = rpi
                force_field.rob2_off_mask[nodm1,nodm2] = 1
                #force_field.rob2[nodm2,nodm1] = rpi
                force_field.params_to_indices[(4,i+1,5)] = (82, (nodm1,nodm2))

            if rpi2 > 0 and force_field.vnq[nodm1] > 0 and force_field.vnq[nodm2] > 0:
                force_field.rob3_off[nodm1,nodm2] = rpi2
                force_field.rob3_off_mask[nodm1,nodm2] = 1
                #force_field.rob3[nodm2,nodm1] = rpi2
                force_field.params_to_indices[(4,i+1,6)] = (83, (nodm1,nodm2))

            if (rodmh > 0):
                force_field.p1co_off[nodm1,nodm2] = rodmh # was 2.0 * rodmh
                force_field.p1co_off_mask[nodm1,nodm2] = 1
                #force_field.p1co[nodm2,nodm1] = 2.0 * rodmh
                force_field.params_to_indices[(4,i+1,2)] = (84, (nodm1,nodm2))

            if (deodmh > 0):
                force_field.p2co_off_mask[nodm1,nodm2] = 1
                force_field.p2co_off[nodm1,nodm2] = deodmh
                #force_field.p2co[nodm2,nodm1] = deodmh
                force_field.params_to_indices[(4,i+1,1)] = (85, (nodm1,nodm2))

            if (godmh > 0):
                force_field.p3co_off_mask[nodm1,nodm2] = 1
                force_field.p3co_off[nodm1,nodm2] = godmh
                #force_field.p3co[nodm2,nodm1] = godmh
                force_field.params_to_indices[(4,i+1,3)] = (86, (nodm1,nodm2))

    # valency angle parameters
    line = f.readline().strip()
    num_val_params = int(line.split()[0])
    for i in range(num_val_params):
        line = f.readline().strip()
        split_line = line.split()
        ind1 = int(split_line[0])
        ind2 = int(split_line[1])
        ind3 = int(split_line[2])

        th0 = float(split_line[3])
        vka = float(split_line[4])
        vka3 = float(split_line[5])
        vka8 = float(split_line[6])
        vkac = float(split_line[7])
        vkap = float(split_line[8])
        vval2 = float(split_line[9])

        ind1 = ind1 - 1 #index starts from 0
        ind2 = ind2 - 1 #index starts from 0
        ind3 = ind3 - 1 #index starts from 0
        if ind1 in MY_ATOM_INDICES and ind2 in MY_ATOM_INDICES and ind3 in MY_ATOM_INDICES:
            force_field.th0[ind1,ind2,ind3] = th0
            force_field.vka[ind1,ind2,ind3] = vka
            force_field.vka3[ind1,ind2,ind3] = vka3
            force_field.vka8[ind1,ind2,ind3] = vka8
            force_field.vkac[ind1,ind2,ind3] = vkac
            force_field.vkap[ind1,ind2,ind3] = vkap
            force_field.vval2[ind1,ind2,ind3] = vval2
            body_3_indices_dst[0].append(ind3)
            body_3_indices_dst[1].append(ind2)
            body_3_indices_dst[2].append(ind1)

            body_3_indices_src[0].append(ind1)
            body_3_indices_src[1].append(ind2)
            body_3_indices_src[2].append(ind3)

            force_field.params_to_indices[(5,i+1, 1)] = (39, (ind1, ind2, ind3))
            force_field.params_to_indices[(5,i+1, 2)] = (40, (ind1, ind2, ind3))
            force_field.params_to_indices[(5,i+1, 3)] = (42, (ind1, ind2, ind3))
            force_field.params_to_indices[(5,i+1, 4)] = (43, (ind1, ind2, ind3))
            force_field.params_to_indices[(5,i+1, 5)] = (38, (ind1, ind2, ind3))
            force_field.params_to_indices[(5,i+1, 6)] = (41, (ind1, ind2, ind3))
            force_field.params_to_indices[(5,i+1, 7)] = (36, (ind1, ind2, ind3))

            '''
            force_field.th0[ind3,ind2,ind1] = th0
            force_field.vka[ind3,ind2,ind1] = vka
            force_field.vka3[ind3,ind2,ind1] = vka3
            force_field.vka8[ind3,ind2,ind1] = vka8
            force_field.vkac[ind3,ind2,ind1] = vkac
            force_field.vkap[ind3,ind2,ind1] = vkap
            force_field.vval2[ind3,ind2,ind1] = vval2
            '''

            if abs(vka) > 0.001:
                force_field.valency_params_mask[ind1,ind2,ind3] = 1.0
                #force_field.valency_params_mask[ind2,ind1,ind3] = 1.0
                #force_field.valency_params_mask[ind3,ind1,ind2] = 1.0

                #force_field.valency_params_mask[ind1,ind3,ind2] = 1.0
                #force_field.valency_params_mask[ind2,ind3,ind1] = 1.0
                force_field.valency_params_mask[ind3,ind2,ind1] = 1.0


    # torsion parameters
    line = f.readline().strip()
    num_tors_params = int(line.split()[0])
    torsion_param_sets = set()
    for tors in range(num_tors_params):
        line = f.readline().strip()
        split_line = line.split()
        ind1 = int(split_line[0])
        ind2 = int(split_line[1])
        ind3 = int(split_line[2])
        ind4 = int(split_line[3])

        v1 = float(split_line[4])
        v2 = float(split_line[5])
        v3 = float(split_line[6])
        v4 = float(split_line[7])
        vconj = float(split_line[8])
        #v2bo = float(split_line[9])
        #v3bo = float(split_line[10])

        ind1 = ind1 - 1 #index starts from 0
        ind2 = ind2 - 1 #index starts from 0
        ind3 = ind3 - 1 #index starts from 0
        ind4 = ind4 - 1 #index starts from 0


        # TODO: handle 0 indices in the param. file later
        if (ind1 > -1 and ind4 > -1 and
            ind1 in MY_ATOM_INDICES and ind2 in MY_ATOM_INDICES and
            ind3 in MY_ATOM_INDICES and ind4 in MY_ATOM_INDICES):

            force_field.v1[ind1,ind2,ind3,ind4] = v1
            force_field.v2[ind1,ind2,ind3,ind4] = v2
            force_field.v3[ind1,ind2,ind3,ind4] = v3
            force_field.v4[ind1,ind2,ind3,ind4] = v4
            force_field.vconj[ind1,ind2,ind3,ind4] = vconj
            force_field.torsion_params_mask[ind1,ind2,ind3,ind4] = 1

            force_field.params_to_indices[(6,tors+1, 1)] = (66, (ind1, ind2, ind3, ind4))
            force_field.params_to_indices[(6,tors+1, 2)] = (67, (ind1, ind2, ind3, ind4))
            force_field.params_to_indices[(6,tors+1, 3)] = (68, (ind1, ind2, ind3, ind4))
            force_field.params_to_indices[(6,tors+1, 4)] = (69, (ind1, ind2, ind3, ind4))
            force_field.params_to_indices[(6,tors+1, 5)] = (70, (ind1, ind2, ind3, ind4))

            '''
            force_field.v1[ind4,ind3,ind2,ind1] = v1
            force_field.v2[ind4,ind3,ind2,ind1] = v2
            force_field.v3[ind4,ind3,ind2,ind1] = v3
            force_field.v4[ind4,ind3,ind2,ind1] = v4
            force_field.vconj[ind4,ind3,ind2,ind1] = vconj
            '''
            body_4_indices_dst[0].append(ind4)
            body_4_indices_dst[1].append(ind3)
            body_4_indices_dst[2].append(ind2)
            body_4_indices_dst[3].append(ind1)

            body_4_indices_src[0].append(ind1)
            body_4_indices_src[1].append(ind2)
            body_4_indices_src[2].append(ind3)
            body_4_indices_src[3].append(ind4)
            torsion_param_sets.add((ind1,ind2,ind3,ind4))
            torsion_param_sets.add((ind4,ind3,ind2,ind1))

            force_field.torsion_params_mask[ind4,ind3,ind2,ind1] = 1

        elif (ind1 == -1 and ind4 == -1):
            # Last index is reserved for this part
            sel_ind = force_field.total_num_atom_types - 1
            force_field.params_to_indices[(6,tors+1, 1)] = (66, (sel_ind, ind2, ind3, sel_ind))
            force_field.params_to_indices[(6,tors+1, 2)] = (67, (sel_ind, ind2, ind3, sel_ind))
            force_field.params_to_indices[(6,tors+1, 3)] = (68, (sel_ind, ind2, ind3, sel_ind))
            force_field.params_to_indices[(6,tors+1, 4)] = (69, (sel_ind, ind2, ind3, sel_ind))
            force_field.params_to_indices[(6,tors+1, 5)] = (70, (sel_ind, ind2, ind3, sel_ind))

            for i in range(num_atom_types):
                for j in range(num_atom_types):
                    if (i,ind2,ind3,j) not in torsion_param_sets:

                        body_4_indices_src[0].append(sel_ind)
                        body_4_indices_src[1].append(ind2)
                        body_4_indices_src[2].append(ind3)
                        body_4_indices_src[3].append(sel_ind)

                        body_4_indices_dst[0].append(i)
                        body_4_indices_dst[1].append(ind2)
                        body_4_indices_dst[2].append(ind3)
                        body_4_indices_dst[3].append(j)

                        body_4_indices_src[0].append(sel_ind)
                        body_4_indices_src[1].append(ind2)
                        body_4_indices_src[2].append(ind3)
                        body_4_indices_src[3].append(sel_ind)

                        body_4_indices_dst[0].append(j)
                        body_4_indices_dst[1].append(ind3)
                        body_4_indices_dst[2].append(ind2)
                        body_4_indices_dst[3].append(i)

                        force_field.v1[sel_ind,ind2,ind3,sel_ind] = v1
                        force_field.v2[sel_ind,ind2,ind3,sel_ind] = v2
                        force_field.v3[sel_ind,ind2,ind3,sel_ind] = v3
                        force_field.v4[sel_ind,ind2,ind3,sel_ind] = v4
                        force_field.vconj[sel_ind,ind2,ind3,sel_ind] = vconj
                        force_field.torsion_params_mask[i,ind2,ind3,j] = 1
                        force_field.torsion_params_mask[j,ind3,ind2,i] = 1

    # hbond parameters
    line = f.readline().strip()
    num_hbond_params = int(line.split()[0])
    for i in range(num_hbond_params):
        line = f.readline().strip()
        split_line = line.split()

        ind1 = int(split_line[0]) - 1
        ind2 = int(split_line[1]) - 1
        ind3 = int(split_line[2]) -1

        rhb = float(split_line[3])
        dehb = float(split_line[4])
        vhb1 = float(split_line[5])
        vhb2 = float(split_line[6])

        force_field.rhb[ind1,ind2,ind3] = rhb
        force_field.dehb[ind1,ind2,ind3] = dehb
        force_field.vhb1[ind1,ind2,ind3] = vhb1
        force_field.vhb2[ind1,ind2,ind3] = vhb2
        force_field.hbond_params_mask[ind1,ind2,ind3] = 1

        force_field.params_to_indices[(7,i+1, 1)] = (87, (ind1,ind2,ind3))
        force_field.params_to_indices[(7,i+1, 2)] = (88, (ind1,ind2,ind3))
        force_field.params_to_indices[(7,i+1, 3)] = (89, (ind1,ind2,ind3))
        force_field.params_to_indices[(7,i+1, 4)] = (90, (ind1,ind2,ind3))


    f.close()

    for i in range(3):
        body_3_indices_src[i] = onp.array(body_3_indices_src[i],dtype=onp.int32)
        body_3_indices_dst[i] = onp.array(body_3_indices_dst[i],dtype=onp.int32)

    for i in range(4):
        body_4_indices_src[i] = onp.array(body_4_indices_src[i],dtype=onp.int32)
        body_4_indices_dst[i] = onp.array(body_4_indices_dst[i],dtype=onp.int32)       

    force_field.body_3_indices_src = tuple(body_3_indices_src)
    force_field.body_3_indices_dst = tuple(body_3_indices_dst)
    force_field.body_4_indices_src = tuple(body_4_indices_src)
    force_field.body_4_indices_dst = tuple(body_4_indices_dst)
    return force_field

def parse_geo_file(geo_file):
    import copy
    if not os.path.exists(geo_file):
        print("Path {} does not exist!".format(geo_file))
        return []
    list_systems = []
    f = open(geo_file,'r')
    run_str = ''
    name = ''
    atoms_positions = []
    bond_restraints = []
    angle_restraints = []
    torsion_restraints = []
    molcharge_items = []
    atom_names = []
    system_str = ''
    box = []
    box_angles = []
    is_periodic = False
    do_minimization = True
    max_it = 99999
    for line in f:
        if len(line.strip()) > 2:
            system_str = system_str + line
        line = line.split('#', 1)[0]
        if line.strip().startswith('#') or len(line) < 1:
            continue
        if line.startswith('END'):
            #add the filler atoms
            num_atoms = len(atom_names)
            new_pos = [10000,10000,10000]
            incr = [1000,1000,1000]
            # 1 filler atom needed to use for padded inter.
            for i in range(1):
                #TODO: masking things out is better than this way
                atom_names.append('FILLER')
                new_pos[0] = new_pos[0] + incr[0]
                new_pos[1] = new_pos[1] + incr[1]
                new_pos[2] = new_pos[2] + incr[2]
                atoms_positions.append(copy.deepcopy(new_pos))

            # currently only total charge for all of the atom is supported, no partial charges
            if len(molcharge_items) > 1 or (len(molcharge_items) == 1 and (molcharge_items[0][1] - molcharge_items[0][0] + 1) < num_atoms):
                print("[ERROR] error in {}, MOLCHARGE is only supported for the total system charge!".format(name))
                sys.exit()

            total_charge = 0

            if len(molcharge_items) == 1:
                total_charge = molcharge_items[0][2]

            new_system = Structure(name, num_atoms, onp.array(atoms_positions),[], atom_names,total_charge,is_periodic,do_minimization,max_it, onp.array(box),onp.array(box_angles),onp.array(bond_restraints),onp.array(angle_restraints),onp.array(torsion_restraints))
            new_system.bgf_file = system_str
            list_systems.append(new_system)
            atoms_positions = []
            atom_names = []
            bond_restraints = []
            angle_restraints = []
            torsion_restraints = []
            molcharge_items = []
            system_str = ''
            box = []
            box_angles = []
            is_periodic = False
            do_minimization = True
            max_it = 99999
        else:
            if line.startswith('DESCRP'):
                name = line.strip().split()[1]
            elif line.startswith('CRYSTX'):
                line = line.strip().split()
                x = float(line[1])
                y = float(line[2])
                z = float(line[3])
                x_ang = float(line[4])
                y_ang = float(line[5])
                z_ang = float(line[6])
                box = [x,y,z]
                box_angles = [x_ang,y_ang,z_ang]
                is_periodic = True
            elif line.startswith('RUTYPE'):

                if line.find('SINGLE') > -1:
                    do_minimization = False
                    max_it = 0 # means full minimization

                elif line.find('NORMAL RUN') > -1:
                    do_minimization = True
                    max_it = 99999 # means full minimization
                #for now assume all of them are the same
                #TODO: fix this
                elif line.find('MAXIT') > -1:
                    max_it = int(line.strip().split()[-1])

                    if max_it < 5:
                        max_it = 0
                        do_minimization = False

            elif line.startswith('MOLCHARGE'):
                #Ex. MOLCHARGE   1  30  1.00
                split_line = line.split()[1:]
                at1, at2 = split_line[:2]
                total_charge = float(split_line[2])
                molcharge_items.append([int(at1)-1,int(at2)-1,total_charge])

            elif line.startswith('BOND RESTRAINT'):
                split_line = line.split()[2:]
                at1,at2 = split_line[:2]
                dist = split_line[2]
                force1,force2 = split_line[3:5]
                d_dist = split_line[5]
                bond_restraints.append([int(at1)-1,int(at2)-1,float(force1),float(force2),float(dist),float(d_dist), 1])
            elif line.startswith('ANGLE RESTRAINT'):
                split_line = line.split()[2:]
                at1,at2,at3 = split_line[:3]
                angle = split_line[3]
                force1,force2 = split_line[4:6]
                d_angle = split_line[6]
                angle_restraints.append([int(at1)-1,int(at2)-1,float(at3)-1,float(force1),float(force2),float(angle),float(d_angle),1])
            elif line.startswith('TORSION RESTRAINT'):
                split_line = line.split()[2:]
                at1,at2,at3,at4 = split_line[:4]
                torsion = split_line[4]
                force1,force2 = split_line[5:7]
                d_torsion = split_line[7]
                torsion_restraints.append([int(at1)-1,int(at2)-1,int(at3)-1,int(at4)-1,float(force1),float(force2),float(torsion),float(d_torsion),1])
            elif line.startswith('HETATM'):
                line = line.strip().split()
                atom_index = int(line[1])
                atom_name = line[2]
                x = float(line[3])
                y = float(line[4])
                z = float(line[5])
                atom_pos = [x,y,z]
                atoms_positions.append(atom_pos)
                atom_names.append(atom_name)

    f.close()

    return list_systems

def parse_modified_params(params_file, ignore_sensitivity=1):

    # section indices sensitivity low_end high_end !comments
    if not os.path.exists(params_file):
        return
    params = []
    f = open(params_file,'r')

    for line in f:
        # remove comments
        line = line.split('!')[0]
        line = line.split('#')[0]
        split_line = line.strip().split()
        if len(split_line) < 6:
            continue
        section = int(split_line[0])

        index1 = int(split_line[1])
        index2 = int(split_line[2])
        sensitivity = float(split_line[3])
        low_end = float(split_line[4])
        high_end = float(split_line[5])
        if ignore_sensitivity:
            sensitivity = 1
        if low_end > high_end:
            temp = low_end
            low_end = high_end
            high_end = temp
        item = (section,index1,index2,sensitivity, low_end, high_end)
        params.append(item)

    return params

def map_params(params, index_map):
    new_params = []
    for p in params:
        key = (p[0],p[1],p[2])
        value = index_map[key]
        new_item = (value, p[3],p[4],p[5])
        new_params.append(new_item)
    return new_params
