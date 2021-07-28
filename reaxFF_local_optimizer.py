#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:13:36 2020

@author: cagri
"""

from reaxFF_potential import jax_calculate_total_energy_vmap
from reaxFF_potential import jax_calculate_total_energy_for_minim_vmap,safe_sqrt
from force_field import preprocess_force_field,dgrrdn,rdndgr,TYPE
from helper import parse_and_save_force_field
from simulation_system import SimulationSystem
import numpy as onp
import jax.numpy as np
import jax

import pickle
import time
from jax.experimental import optimizers
import sys
import matplotlib.pyplot as plt
import re
import copy
from scipy.optimize import minimize

DEVICE_NAME = 'gpu'
# global param, will be filled while reading "params" file


def calculate_dist_and_angles(list_positions,list_orth_matrices,list_all_shift_combs,
							  list_all_body_2_list,list_all_body_2_map,
							  list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,
							  list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,
							  list_all_hbond_list,list_all_hbond_shift,list_all_hbond_mask):
	list_all_dist_mat = [jax.vmap(SimulationSystem.create_distance_matrices)(list_positions[i],list_orth_matrices[i],list_all_shift_combs[i])  for i in range(len(list_positions))]
	list_all_body_2_distances = [jax.vmap(SimulationSystem.calculate_2_body_distances)(list_positions[i],list_orth_matrices[i],list_all_body_2_list[i],list_all_body_2_map[i]) for i in range(len(list_positions))]
	list_all_body_3_angles = [jax.vmap(SimulationSystem.calculate_3_body_angles)(list_positions[i],list_orth_matrices[i],list_all_body_2_list[i],list_all_body_3_list[i],list_all_body_3_map[i],list_all_body_3_shift[i]) for i in range(len(list_positions))]
	list_all_body_4_angles = [jax.vmap(SimulationSystem.calculate_body_4_angles_new)(list_positions[i],list_orth_matrices[i],list_all_body_4_list[i],list_all_body_4_map[i],list_all_body_4_shift[i]) for i in range(len(list_positions))]
	list_all_angles_and_dist = [jax.vmap(SimulationSystem.calculate_global_hbond_angles_and_dist)(list_positions[i],list_orth_matrices[i],list_all_hbond_list[i],list_all_hbond_shift[i],list_all_hbond_mask[i]) for i in range(len(list_positions))]

	return [list_all_dist_mat,list_all_body_2_distances, list_all_body_3_angles, list_all_body_4_angles, list_all_angles_and_dist]

def learning_rate_search(LR_list,selected_params, grads, flattened_force_field,flattened_non_dif_params,
					 energy_items, charge_items,
					 list_all_type,list_all_mask,
					 list_all_body_2_neigh_list,
					 list_all_dist_mat,
					 list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
					 list_all_body_3_list,list_all_body_3_map,list_all_body_3_angles,
					 list_all_body_4_list,list_all_body_4_map,list_all_body_4_angles):

	loss_func = jax.jit(jax_loss_vmap_new_test,backend=DEVICE_NAME)

	best_LR_ind = 0
	best_loss = 9999999999999999

	for i,LR in enumerate(LR_list):
		params = calculate_params_from_grad(selected_params, grads, LR)
		loss = loss_func(params, flattened_force_field,flattened_non_dif_params,
					 energy_items, charge_items,
					 list_all_type,list_all_mask,
					 list_all_body_2_neigh_list,
					 list_all_dist_mat,
					 list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
					 list_all_body_3_list,list_all_body_3_map,list_all_body_3_angles,
					 list_all_body_4_list,list_all_body_4_map,list_all_body_4_angles)

		if best_loss > loss:
			best_loss = loss
			best_LR_ind = i

	return LR_list[best_LR_ind]

def hessian(f):
    return jax.jacfwd(jax.jacrev(f))

def update_positions(orig_pos,cur_positions,change_vec):
	# DONT LET ATOMS MOVE MORE THAN 0.01
	change_vec = np.clip(change_vec, -0.01, +0.01)
	new_pos = cur_positions - change_vec

	diff = orig_pos - new_pos
	diff = np.clip(diff, -0.3, 0.3)
	new_pos = orig_pos - diff
	# make sure we are in the box
	#TODO change this part later
	return new_pos

def update_list_positions_SG_w_momentum(orig_list_pos, list_cur_pos, list_m, beta, list_grads, list_do_minim,RMSG_list, multip, indiv_LR):
	new_positions = []
	for i in range(len(orig_list_pos)):
		clip = 1.00
		list_grads[i] = np.clip(list_grads[i], -clip, clip)
		list_grads[i] = np.where(np.abs(list_grads[i]) < 0.1, 0.0, list_grads[i])
		list_grads[i] = list_grads[i] * (list_do_minim[i] * multip * indiv_LR[i]).reshape(-1,1,1)
		list_m[i] = list_grads[i] + beta * list_m[i]  # First  moment estimate

		pos = update_positions(orig_list_pos[i],list_cur_pos[i],list_grads[i])
		new_positions.append(pos)
	return new_positions

def update_list_positions(orig_list_pos, list_cur_pos, list_grads, list_do_minim,RMSG_list, multip,indiv_LR):
	new_positions = []
	for i in range(len(orig_list_pos)):
		#clip = np.where(RMSG_list[i] > 10, 1.0, 1.0)
		clip = 10.0
		list_grads[i] = np.clip(list_grads[i], -clip, clip)
		list_grads[i] = np.where(np.abs(list_grads[i]) < 0.1, 0.0, list_grads[i])
		#maxx = np.max(np.abs(list_grads[i]))
		#list_grads[i] = np.where(maxx > 1.0, list_grads[i] / maxx, list_grads[i])
		list_grads[i] = list_grads[i] * (list_do_minim[i] * multip * indiv_LR[i]).reshape(-1,1,1)

		pos = update_positions(orig_list_pos[i],list_cur_pos[i],list_grads[i])
		new_positions.append(pos)
	return new_positions


def update_grads(list_grads,list_real_atom_count,end_RMSG, list_num_minim_steps):
	#RMSG condition np.where(np.sqrt(np.mean(g**2, axis=(1,2))).reshape(-1,1,1)>1.0, 1.0, 0.0)
	RMSG_list = [np.sqrt(np.sum(g**2, axis=(1,2)) / (cnt.reshape(-1) * 3)).reshape(-1,1,1) for cnt,g in zip(list_real_atom_count,list_grads) ]
	return  [g * np.where(s > 0,1,0).reshape(-1,1,1) * np.where(RMSG >end_RMSG, 1.0, 0.0) for g,s,RMSG in zip(list_grads,list_num_minim_steps,RMSG_list) ], RMSG_list


def select_energy_minim(list_do_minim):
	index_lists = []
	for minim_list in list_do_minim:
		index_list = np.argwhere(minim_list == True).reshape(-1)
		index_lists.append(index_list)
	return index_lists

def get_minim_lists(minim_index_lists,list_do_minim, list_num_minim_steps,
					 list_real_atom_counts,
					 orig_list_all_pos,list_all_pos, list_all_shift_combs,list_orth_matrices,
					 list_all_type,list_all_mask,list_all_total_charge,
				 list_all_body_2_neigh_list,list_all_body_2_list,list_all_body_2_map,
				 list_all_body_2_trip_mask,list_all_body_3_list,
				 list_all_body_3_map,list_all_body_3_shift,
				 list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,
				  list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,
				 list_bond_rest,list_angle_rest,list_torsion_rest):

	list_do_minim_sub = []
	list_num_minim_steps_sub = []
	list_real_atom_counts_sub = []
	orig_list_all_pos_sub = []
	list_all_pos_sub = []
	list_all_shift_combs_sub = []
	list_orth_matrices_sub = []
	list_all_type_sub = []
	list_all_mask_sub = []
	list_all_total_charge_sub = []
	list_all_body_2_neigh_list_sub = []
	list_all_body_2_list_sub = []
	list_all_body_2_map_sub = []
	list_all_body_2_trip_mask_sub = []
	list_all_body_3_list_sub = []
	list_all_body_3_map_sub = []
	list_all_body_3_shift_sub = []
	list_all_body_4_list_sub = []
	list_all_body_4_map_sub = []
	list_all_body_4_shift_sub = []
	list_all_hbond_list_sub = []
	list_all_hbond_mask_sub = []
	list_all_hbond_shift_sub = []
	list_bond_rest_sub = []
	list_angle_rest_sub = []
	list_torsion_rest_sub = []

	for i,indices in enumerate(minim_index_lists):
		# skip zero sized clusters
		if len(indices) != 0:
			list_do_minim_sub.append(list_do_minim[i][indices])
			list_num_minim_steps_sub.append(list_num_minim_steps[i][indices])
			list_real_atom_counts_sub.append(list_real_atom_counts[i][indices])
			orig_list_all_pos_sub.append(orig_list_all_pos[i][indices,:,:])
			list_all_pos_sub.append(list_all_pos[i][indices,:,:])
			list_all_shift_combs_sub.append(list_all_shift_combs[i][indices,:,:])
			list_orth_matrices_sub.append(list_orth_matrices[i][indices,:,:])
			list_all_type_sub.append(list_all_type[i][indices,:])
			list_all_mask_sub.append(list_all_mask[i][indices,:])
			list_all_total_charge_sub.append(list_all_total_charge[i][indices])
			list_all_body_2_neigh_list_sub.append(list_all_body_2_neigh_list[i][indices,:,:,:])
			list_all_body_2_list_sub.append(list_all_body_2_list[i][indices,:,:])
			list_all_body_2_map_sub.append(list_all_body_2_map[i][indices,:])
			list_all_body_2_trip_mask_sub.append(list_all_body_2_trip_mask[i][indices,:])
			list_all_body_3_list_sub.append(list_all_body_3_list[i][indices,:,:])
			list_all_body_3_map_sub.append(list_all_body_3_map[i][indices,:])
			list_all_body_3_shift_sub.append(list_all_body_3_shift[i][indices,:,:])
			list_all_body_4_list_sub.append(list_all_body_4_list[i][indices,:,:])
			list_all_body_4_map_sub.append(list_all_body_4_map[i][indices,:])
			list_all_body_4_shift_sub.append(list_all_body_4_shift[i][indices,:,:])
			list_all_hbond_list_sub.append(list_all_hbond_list[i][indices,:,:])
			list_all_hbond_mask_sub.append(list_all_hbond_mask[i][indices,:])
			list_all_hbond_shift_sub.append(list_all_hbond_shift[i][indices,:,:])
			list_bond_rest_sub.append(list_bond_rest[i][indices,:,:])
			list_angle_rest_sub.append(list_angle_rest[i][indices,:,:])
			list_torsion_rest_sub.append(list_torsion_rest[i][indices,:,:])


	return 	[list_do_minim_sub,
	list_num_minim_steps_sub,
	list_real_atom_counts_sub,
	orig_list_all_pos_sub,
	list_all_pos_sub,
	list_all_shift_combs_sub,
	list_orth_matrices_sub,
	list_all_type_sub,
	list_all_mask_sub,
	list_all_total_charge_sub,
	list_all_body_2_neigh_list_sub,
	list_all_body_2_list_sub,
	list_all_body_2_map_sub,
	list_all_body_2_trip_mask_sub,
	list_all_body_3_list_sub,
	list_all_body_3_map_sub,
	list_all_body_3_shift_sub,
	list_all_body_4_list_sub,
	list_all_body_4_map_sub,
	list_all_body_4_shift_sub,
	list_all_hbond_list_sub,
	list_all_hbond_mask_sub,
	list_all_hbond_shift_sub,
	list_bond_rest_sub,
	list_angle_rest_sub,
	list_torsion_rest_sub]

def replace_pos_with_subs(minim_index_lists, list_all_pos, list_all_pos_sub):
	ctr=0
	for i in range(len(list_all_pos)):
		if len(minim_index_lists[i]) != 0:
			list_all_pos[i] = jax.ops.index_update(list_all_pos[i], minim_index_lists[i], list_all_pos_sub[ctr])
			ctr = ctr + 1

	return list_all_pos

def energy_minim_with_subs(list_all_pos,minim_index_lists,flattened_force_field,flattened_non_dif_params,subs, grad_and_loss_func, energy_minim_count, energy_minim_init_LR,energy_minim_multip_LR,end_RMSG):

	[list_do_minim_sub,
	list_num_minim_steps_sub,
	list_real_atom_counts_sub,
	orig_list_all_pos_sub,
	list_all_pos_sub,
	list_all_shift_combs_sub,
	list_orth_matrices_sub,
	list_all_type_sub,
	list_all_mask_sub,
	list_all_total_charge_sub,
	list_all_body_2_neigh_list_sub,
	list_all_body_2_list_sub,
	list_all_body_2_map_sub,
	list_all_body_2_trip_mask_sub,
	list_all_body_3_list_sub,
	list_all_body_3_map_sub,
	list_all_body_3_shift_sub,
	list_all_body_4_list_sub,
	list_all_body_4_map_sub,
	list_all_body_4_shift_sub,
	list_all_hbond_list_sub,
	list_all_hbond_mask_sub,
	list_all_hbond_shift_sub,
	list_bond_rest_sub,
	list_angle_rest_sub,
	list_torsion_rest_sub] = subs

	list_positions_sub,loss_vals,min_loss,minn_loss_vals,list_RMSG = energy_minimizer(grad_and_loss_func,energy_minim_count,
															  energy_minim_init_LR,energy_minim_multip_LR,list_do_minim_sub,list_num_minim_steps_sub,end_RMSG,
															  flattened_force_field,flattened_non_dif_params,
															  list_real_atom_counts_sub,
															 orig_list_all_pos_sub,
															 list_all_pos_sub, list_all_shift_combs_sub,list_orth_matrices_sub,
															 list_all_type_sub,list_all_mask_sub,list_all_total_charge_sub,
															 list_all_body_2_neigh_list_sub,list_all_body_2_list_sub,list_all_body_2_map_sub,
															 list_all_body_2_trip_mask_sub,list_all_body_3_list_sub,
															 list_all_body_3_map_sub,list_all_body_3_shift_sub,
															 list_all_body_4_list_sub,list_all_body_4_map_sub,list_all_body_4_shift_sub,
															 list_all_hbond_list_sub,list_all_hbond_mask_sub,list_all_hbond_shift_sub,
															 list_bond_rest_sub,list_angle_rest_sub,list_torsion_rest_sub)

	list_all_pos = jax.jit(replace_pos_with_subs, static_argnums=(0,), backend=DEVICE_NAME)(minim_index_lists, list_all_pos, list_positions_sub)

	return list_all_pos,loss_vals,min_loss,minn_loss_vals,list_RMSG

def change_indiv_LR(indiv_LR, cur_vals, prev_vals, decr, incr):

	for i in range(len(indiv_LR)):
		indiv_LR[i] = np.where(cur_vals[i] < prev_vals[i], indiv_LR[i] * incr, indiv_LR[i] * decr)
	return indiv_LR

def compare_and_update_pos(prev_pos, prev_energy, cur_pos, cur_energy, change_masks):
	for i in range(len(prev_energy)):

		cur_pos[i] = np.where(prev_energy[i].reshape(-1,1,1) < cur_energy[i].reshape(-1,1,1), prev_pos[i], cur_pos[i])
		change_masks[i] = np.where(prev_energy[i].reshape(-1,1,1) < cur_energy[i].reshape(-1,1,1), 0, 1)
		cur_energy[i] = np.where(prev_energy[i] < cur_energy[i], prev_energy[i], cur_energy[i])

	return cur_pos, cur_energy, change_masks

def energy_minimizer(grad_and_loss_func, count, init_LR,multip_LR,list_do_minim, list_num_minim_steps,end_RMSG,
					 flattened_force_field,flattened_non_dif_params,
					 list_real_atom_counts,
					 orig_list_all_pos,list_all_pos, list_all_shift_combs,list_orth_matrices, list_all_type,list_all_mask,
					 list_all_total_charge,
				 list_all_body_2_neigh_list,list_all_body_2_list,list_all_body_2_map,
				 list_all_body_2_trip_mask,list_all_body_3_list,
				 list_all_body_3_map,list_all_body_3_shift,
				 list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,
				  list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,
				 list_bond_rest,list_angle_rest,list_torsion_rest):

	# use conjugate gradient
	#minim_index_lists = select_energy_minim(list_do_minim)
	loss_vals = []
	#min_pos = copy.deepcopy(list_all_pos)
	change_masks = [np.ones_like(l, dtype=np.int32).reshape(-1,1,1) for l in list_real_atom_counts]

	LR = init_LR
	cur_all_los_vals = 0.0
	cur_total_loss = 0.0
	#prev_all_loss_vals = [np.ones_like(l, dtype=np.float32) * -999999999.0 for l in list_do_minim]
	indiv_LR = [np.ones_like(l, dtype=np.float32) for l in list_do_minim]
	decr = np.float32(0.75)
	incr = np.float32(1.1)
	#start2 = time.time()
	prev_pos = 0
	prev_energies = 0
	for iter_c in range(count):





		list_grads,cur_all_los_vals = energy_minim_gradients(grad_and_loss_func,flattened_force_field,flattened_non_dif_params,
																	   list_all_pos, list_all_shift_combs,
																	   list_orth_matrices, list_all_type,list_all_mask,list_all_total_charge,
														 list_all_body_2_neigh_list,list_all_body_2_list,list_all_body_2_map,
														 list_all_body_2_trip_mask,list_all_body_3_list,
														 list_all_body_3_map,list_all_body_3_shift,
														 list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,
														 list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,
														 list_bond_rest,list_angle_rest,list_torsion_rest)

		if iter_c > 0:
			indiv_LR = jax.jit(change_indiv_LR,backend=DEVICE_NAME)(indiv_LR, cur_all_los_vals, prev_energies, decr, incr)
			#list_all_pos, cur_all_los_vals, change_masks = jax.jit(compare_and_update_pos,backend=DEVICE_NAME)(prev_pos, prev_energies, list_all_pos, cur_all_los_vals, change_masks)


		cur_total_loss = sum([np.sum(loss_vals) for loss_vals in cur_all_los_vals])
			#print(prev[2][:20])
			#print(cur_all_los_vals[2][:20])
		#list_grads,RMSG_list = jax.jit(update_grads,backend=DEVICE_NAME)(list_grads,list_real_atom_counts_sub,end_RMSG, list_num_minim_steps_sub)
		list_grads,RMSG_list = jax.jit(update_grads,backend=DEVICE_NAME)(list_grads,list_real_atom_counts,end_RMSG, list_num_minim_steps)

		prev_pos = list_all_pos
		prev_energies = cur_all_los_vals

		if iter_c > 0:
			list_grads = [grads * mask for (grads, mask) in zip(list_grads, change_masks)]

		list_all_pos = jax.jit(update_list_positions,backend=DEVICE_NAME)(orig_list_all_pos, list_all_pos, list_grads, list_do_minim,RMSG_list, LR, indiv_LR)


		loss_vals.append(cur_total_loss)
		LR = LR * multip_LR
		list_num_minim_steps = [l - 1 for l in list_num_minim_steps]

	#end2 = time.time()
	#print(end2-start2)
	'''
	for i,indices in enumerate(minim_index_lists):
		if len(indices) != 0:
			list_all_pos[i] = list_all_pos_sub[i]
	'''
	return list_all_pos, loss_vals, cur_total_loss,cur_all_los_vals,RMSG_list

def energy_minim_gradients(grad_and_loss_func,flattened_force_field,flattened_non_dif_params,list_positions,
				 list_all_shift_combs,list_orth_matrices, list_all_type,list_all_mask,list_all_total_charge,
				 list_all_body_2_neigh_list,list_all_body_2_list,list_all_body_2_map,
				 list_all_body_2_trip_mask,list_all_body_3_list,
				 list_all_body_3_map,list_all_body_3_shift,
				 list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,
				 list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,
				 list_bond_rest,list_angle_rest,list_torsion_rest):
	cur_loss = 0
	list_grads = []
	all_los_vals = []
	for i in range(len(list_all_type)):
		loss_val, grads = grad_and_loss_func(list_positions[i],flattened_force_field,flattened_non_dif_params,list_all_shift_combs[i],
										 list_orth_matrices[i],
										 list_all_type[i],list_all_mask[i],list_all_total_charge[i],
										 list_all_body_2_neigh_list[i],list_all_body_2_list[i],list_all_body_2_map[i],
										 list_all_body_2_trip_mask[i],list_all_body_3_list[i],
										 list_all_body_3_map[i],list_all_body_3_shift[i],
										 list_all_body_4_list[i],list_all_body_4_map[i],list_all_body_4_shift[i],
										 list_all_hbond_list[i],list_all_hbond_mask[i],list_all_hbond_shift[i],
										 list_bond_rest[i],list_angle_rest[i],list_torsion_rest[i]
										 )

		list_grads.append(grads)
		all_los_vals.append(loss_val)
		#list_copy_all_pos[i] = jax.jit(update_positions, backend=DEVICE_NAME)(list_orig_all_pos[i],list_copy_all_pos[i],grads,LR)
	return list_grads,all_los_vals

def post_process_gradients(grads, batch_size):
	grads = np.nan_to_num(grads)

	grads = grads / batch_size


	return grads

def use_selected_parameters(params,param_indices, flattened_force_field):
	for i, ind in enumerate(param_indices):
		flattened_force_field[ind[0]] = jax.ops.index_update(flattened_force_field[ind[0]], ind[1], params[i])
	return flattened_force_field

def jax_loss_vmap_new_test(selected_params, param_indices, flattened_force_field, flattened_non_dif_params,
			   structured_training_data,
			   list_all_positions,
			   list_all_type,list_all_mask,
			   list_all_total_charge,
			   list_all_shift_combs,
			   list_orth_matrices,
			   list_all_body_2_neigh_list,
			   list_all_dist_mat,
			   list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
			   list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,list_all_body_3_angles,
			   list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,list_all_body_4_angles,
			   list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,list_all_angles_and_dist,
			   list_bond_rest,list_angle_rest,list_torsion_rest,
			   list_do_minim,orig_list_pos,return_indiv_error=False):

	flattened_force_field = use_selected_parameters(selected_params,param_indices, flattened_force_field)
	flattened_force_field = preprocess_force_field(flattened_force_field,flattened_non_dif_params)

	return jax_loss_vmap(flattened_force_field,flattened_non_dif_params,
			   structured_training_data,
			   list_all_positions,
			   list_all_type,list_all_mask,
			   list_all_total_charge,
			   list_all_shift_combs,
			   list_orth_matrices,
			   list_all_body_2_neigh_list,
			   list_all_dist_mat,
			   list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
			   list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,list_all_body_3_angles,
			   list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,list_all_body_4_angles,
			   list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,list_all_angles_and_dist,
			   list_bond_rest,list_angle_rest,list_torsion_rest,return_indiv_error=return_indiv_error)


def jax_loss_vmap(flattened_force_field,flattened_non_dif_params,
			   structured_training_data,
			   list_all_positions,
			   list_all_type,list_all_mask,
			   list_all_total_charge,
			   list_all_shift_combs,
			   list_orth_matrices,
			   list_all_body_2_neigh_list,
			   list_all_dist_mat,
			   list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
			   list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,list_all_body_3_angles,
			   list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,list_all_body_4_angles,
			   list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,list_all_angles_and_dist,
			   list_bond_rest,list_angle_rest,list_torsion_rest,return_indiv_error):
	all_errors = dict()
	total_error = 0
	energy_items_flag = 'ENERGY' in structured_training_data
	charge_items_flag = 'CHARGE' in structured_training_data
	geo_items_flag = 'GEOMETRY-2' in structured_training_data or 'GEOMETRY-3' in structured_training_data or 'GEOMETRY-4' in structured_training_data
	force_items_flag = 'FORCE-ATOM' in structured_training_data or 'FORCE-RMSG' in structured_training_data or 'RMSG-NEW' in structured_training_data

	list_counts = [len(l) for l in list_all_type]
	# max atom count
	total_num_systems = sum(list_counts)
	pot_func = jax.vmap(jax_calculate_total_energy_vmap,in_axes=(None,None,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))
	pot_func_pmap = jax.pmap(jax_calculate_total_energy_vmap,in_axes=(None,None,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),backend=DEVICE_NAME)
	all_pots = np.zeros(total_num_systems,dtype=TYPE)
	atom_counts = np.array([l.shape[1] for l in list_all_type])
	max_atom_count = np.max(atom_counts)
	if charge_items_flag:
		all_charges = np.zeros(shape=(total_num_systems,max_atom_count),dtype=TYPE)
	if geo_items_flag:
		all_positions = np.zeros(shape=(total_num_systems,max_atom_count,3),dtype=TYPE)
	if force_items_flag:
		all_forces = np.zeros(shape=(total_num_systems,max_atom_count,3),dtype=TYPE)
		force_func = jax.vmap(jax.grad(jax_calculate_total_energy_for_minim_vmap),
													   in_axes=(0,None,None,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))
		force_func_pmap = jax.pmap(jax.grad(jax_calculate_total_energy_for_minim_vmap),
													   in_axes=(0,None,None,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),backend=DEVICE_NAME)
	start = 0
	end = 0

	for i in range(len(list_all_type)):

		start = end
		end = start + list_counts[i]

		pots,charges = pot_func(flattened_force_field,flattened_non_dif_params,
									  list_all_type[i],list_all_mask[i],
									  list_all_total_charge[i],
									  list_all_body_2_neigh_list[i],
									  list_all_dist_mat[i],
									  list_all_body_2_list[i],list_all_body_2_map[i],list_all_body_2_trip_mask[i],list_all_body_2_distances[i],
									  list_all_body_3_list[i],list_all_body_3_map[i],list_all_body_3_angles[i],
									  list_all_body_4_list[i],list_all_body_4_map[i],list_all_body_4_angles[i],
									  list_all_hbond_list[i],list_all_hbond_mask[i],list_all_angles_and_dist[i])

		all_pots = jax.ops.index_update(all_pots, jax.ops.index[start:end], pots)
		if charge_items_flag:
			all_charges = jax.ops.index_update(all_charges, jax.ops.index[start:end, :atom_counts[i]], charges)
		if geo_items_flag:
			all_positions = jax.ops.index_update(all_positions, jax.ops.index[start:end, :atom_counts[i], :], list_all_positions[i])
		if force_items_flag:
			forces = force_func(list_all_positions[i],flattened_force_field,flattened_non_dif_params,
							 list_all_shift_combs[i],list_orth_matrices[i], list_all_type[i],list_all_mask[i],
							 list_all_total_charge[i],
							 list_all_body_2_neigh_list[i],list_all_body_2_list[i],list_all_body_2_map[i],
							 list_all_body_2_trip_mask[i],list_all_body_3_list[i],
							 list_all_body_3_map[i],list_all_body_3_shift[i],
							 list_all_body_4_list[i],list_all_body_4_map[i],list_all_body_4_shift[i],
							 list_all_hbond_list[i],list_all_hbond_mask[i],list_all_hbond_shift[i],
							 list_bond_rest[i],list_angle_rest[i],list_torsion_rest[i]
							 )
			all_forces = jax.ops.index_update(all_forces, jax.ops.index[start:end, :atom_counts[i], :], forces)

	if energy_items_flag:
		energy_sys_list_of_lists, energy_multip_list_of_lists, energy_all_weights, energy_all_energy_vals = structured_training_data['ENERGY']
		energy_preds = np.sum(all_pots[energy_sys_list_of_lists] * energy_multip_list_of_lists,axis=1)
		energy_error = np.sum(((energy_all_energy_vals - energy_preds) / energy_all_weights) ** 2)
		if return_indiv_error:
			all_errors['ENERGY'] = [energy_preds, energy_all_energy_vals, energy_all_weights, ((energy_all_energy_vals - energy_preds) / energy_all_weights) ** 2]
	else:
		energy_error = 0
	total_error = total_error  + energy_error
	#print('energy_error',energy_error)

	if charge_items_flag:
		chg_sys_index_list, chg_atom_index_list, chg_all_weights, chg_all_charge_vals = structured_training_data['CHARGE']
		charge_preds = all_charges[chg_sys_index_list,chg_atom_index_list]
		charge_error = np.sum(((chg_all_charge_vals - charge_preds) / chg_all_weights) ** 2)
		total_error = total_error  + charge_error
		if return_indiv_error:
			all_errors['CHARGE'] = [charge_preds, chg_all_charge_vals, chg_all_weights, ((chg_all_charge_vals - charge_preds) / chg_all_weights) ** 2]
		#print('charge_error', charge_error)
	#2-body
	if 'GEOMETRY-2' in structured_training_data:
		geo2_sys_index_list, geo2_atom_index_list, geo2_all_weights, geo2_all_target_vals = structured_training_data['GEOMETRY-2']
		pos1s = all_positions[geo2_sys_index_list, geo2_atom_index_list[:,0]]
		pos2s = all_positions[geo2_sys_index_list, geo2_atom_index_list[:,1]]
		calc_dist = jax.vmap(SimulationSystem.calculate_2_body_distance)(pos1s,pos2s)
		geo2_error = np.sum(((geo2_all_target_vals - calc_dist) / geo2_all_weights) ** 2)
		total_error = total_error  + geo2_error
		if return_indiv_error:
			all_errors['GEOMETRY-2'] = [calc_dist, geo2_all_target_vals, geo2_all_weights, ((geo2_all_target_vals - calc_dist) / geo2_all_weights) ** 2]
		#print('geo2_error',geo2_error)
	#3-body
	if 'GEOMETRY-3' in structured_training_data:
		geo3_sys_index_list, geo3_atom_index_list, geo3_all_weights, geo3_all_target_vals = structured_training_data['GEOMETRY-3']
		pos1s = all_positions[geo3_sys_index_list, geo3_atom_index_list[:,0]]
		pos2s = all_positions[geo3_sys_index_list, geo3_atom_index_list[:,1]]
		pos3s = all_positions[geo3_sys_index_list, geo3_atom_index_list[:,2]]
		geo3_all_target_vals = geo3_all_target_vals #* dgrrdn # degree to radian
		calc_ang = jax.vmap(SimulationSystem.calculate_valence_angle)(pos1s,pos2s,pos3s) * rdndgr
		# to have periodicity, Ex. diff between 170 and -170 is 20 degree.
		calc_ang = np.where(calc_ang < 0.0, calc_ang+360.0, calc_ang)
		geo3_all_target_vals = np.where(geo3_all_target_vals < 0.0, geo3_all_target_vals+360.0, geo3_all_target_vals)

		geo3_error = np.sum(((geo3_all_target_vals - calc_ang) / geo3_all_weights) ** 2)
		total_error = total_error  + geo3_error
		if return_indiv_error:
			all_errors['GEOMETRY-3'] = [calc_ang, geo3_all_target_vals, geo3_all_weights, ((geo3_all_target_vals - calc_ang) / geo3_all_weights) ** 2]
		#print('geo3_error',geo3_error)
	#4-body
	if 'GEOMETRY-4' in structured_training_data:
		geo4_sys_index_list, geo4_atom_index_list, geo4_all_weights, geo4_all_target_vals = structured_training_data['GEOMETRY-4']
		pos1s = all_positions[geo4_sys_index_list, geo4_atom_index_list[:,0]]
		pos2s = all_positions[geo4_sys_index_list, geo4_atom_index_list[:,1]]
		pos3s = all_positions[geo4_sys_index_list, geo4_atom_index_list[:,2]]
		pos4s = all_positions[geo4_sys_index_list, geo4_atom_index_list[:,3]]
		geo4_all_target_vals = geo4_all_target_vals #* dgrrdn# degree to radian
		calc_ang = jax.vmap(SimulationSystem.calculate_body_4_angle_single)(pos1s,pos2s,pos3s,pos4s).reshape(-1)
		calc_ang = calc_ang * rdndgr
		#print(geo4_all_target_vals)
		#print(calc_ang)
		#print('**********')
		#calc_ang = np.where(calc_ang < 0.0, calc_ang+360, calc_ang)
		#geo4_all_target_vals = np.where(geo4_all_target_vals < 0.0, geo4_all_target_vals+360, geo4_all_target_vals)
		# fix later
		geo4_all_target_vals = np.abs(geo4_all_target_vals)
		geo4_error = np.sum(((geo4_all_target_vals - calc_ang) / geo4_all_weights) ** 2)
		total_error = total_error  + geo4_error
		if return_indiv_error:
			all_errors['GEOMETRY-4'] = [calc_ang, geo4_all_target_vals, geo4_all_weights, ((geo4_all_target_vals - calc_ang) / geo4_all_weights) ** 2]
		#print(geo4_all_target_vals)
		#print(calc_ang)
		#print('geo4_error',geo4_error)
	if 'FORCE-ATOM' in structured_training_data:
		force_sys_index_list, force_all_atom_indices, force_all_weights, force_list_target_vals = structured_training_data['FORCE-ATOM']
		calc_forces = all_forces[:, :, :] *-1 #TODO: needed dont know why???
		#remove later

		#force_list_target_vals = force_sys_index_list
		#print(force_list_target_vals[0,0])
		#diff = force_list_target_vals.flatten() - calc_forces.flatten()
		#calc_forces_abs = np.abs(calc_forces)
		#diff = force_list_target_vals - calc_forces_abs.flatten()
		#diff = 450.0 - calc_forces_abs
		#diff = np.where(calc_forces_abs > 450.0, diff, 0.0)
		###############################
		calc_forces = all_forces[force_sys_index_list, force_all_atom_indices, :] *-1 #TODO: needed dont know why???
		force_error = np.sum(((force_list_target_vals - calc_forces) / force_all_weights.reshape(-1,1)) ** 2)
		#force_all_weights = np.where(np.abs(force_list_target_vals)==450.0, 5.0, 100.0).flatten()
		#force_all_weights = np.where(np.abs(force_list_target_vals).flatten()<40.0, 5.0, force_all_weights).flatten()
		#force_error = np.sum(((diff) / force_all_weights) ** 2) # force_all_weights.reshape(-1,1)
		total_error = total_error  + force_error
		if return_indiv_error:
			all_errors['FORCE-ATOM'] = [calc_forces, force_list_target_vals, force_all_weights, ((force_list_target_vals - calc_forces) / force_all_weights.reshape(-1,1)) ** 2]
		#(calc_forces[:20])
	if 'FORCE-RMSG' in structured_training_data:
		force_sys_index_list, force_all_weights, all_target = structured_training_data['FORCE-RMSG']
		calc_rmsg = safe_sqrt(np.mean(all_forces[force_sys_index_list, :, :]**2,axis=(1,2))).reshape(-1)
		force_all_weights = force_all_weights
		new_rmsg_error = np.sum((calc_rmsg / force_all_weights)**2)
		#print(calc_rmsg)
		total_error = total_error  + new_rmsg_error
		#print(new_rmsg_error)
		if return_indiv_error:
			all_errors['FORCE-RMSG'] = [calc_rmsg, all_target, force_all_weights, (calc_rmsg / force_all_weights)**2]

	if not return_indiv_error:
		return total_error
	else:
		return total_error, all_errors

def calculate_params_from_grad(selected_params, grads, learning_rate):
	new_vals = selected_params - grads * learning_rate
	return new_vals


def update_parameters(old_params,new_params, bounds):
	#diff = high_limit - low_limit
	#max_change = diff * 5/100
	#truncated_params = np.clip(new_params,old_params-max_change,old_params+max_change)

	truncated_params = np.clip(new_params,bounds[:,0],bounds[:,1]) #jax.vmap(controlled_update)(new_params ,low_limit,high_limit)
	# dont let params to change more than 5% of the range

	'''
	for i in range(len(PARAM_INDICES)):
		p = PARAM_INDICES[i]
		ind1 = p[0]
		ind2 = p[1]
		new_val =  new_params[i]
		#print(new_val, p)
		new_val = Simulator.controlled_update(new_val, params_list[i][2], params_list[i][3])

		truncated_params = jax.ops.index_update(truncated_params, i, new_val)
		#flattened_force_field[ind][param_indices] = new_val
		#print(ind, param_indices, new_val)
	'''
	return truncated_params



def controlled_update(new_val, low_limit, high_limit):
	#noise = onp.random.random() * diff/10000
	#noise = noise - 0.5 * diff/10000
	#new_val = new_val + onp.random.normal(0, 0.0005, 1)[0]
	'''
	if new_val < low_limit:
		new_val = low_limit
		#new_val = low_limit
	elif new_val > high_limit:
		new_val = high_limit
		#new_val = high_limit
	'''
	#new_val = np.clip(new_val,low_limit,high_limit)
	# dont let value to be 0
	#close_zero = np.where(high_limit > 0, 0.001, -0.001)
	#new_val = np.where(new_val == 0, close_zero, new_val)
	return np.clip(new_val,low_limit,high_limit)

def create_random_order(size):
    numbers = onp.arange(size)
    onp.random.shuffle(numbers)
    return numbers


def step(i, opt_state, opt_update, grads, new_LR):
    return jax.jit(opt_update,backend=DEVICE_NAME)([i,new_LR], grads, opt_state)


def find_max_dist_changes(list_new_dist, list_old_dist):
	max_changes = onp.zeros(len(list_new_dist))

	for i in range(len(list_new_dist)):
		l_new = list_new_dist[i]
		l_old = list_old_dist[i]
		diff = l_new - l_old
		cand_max_change = np.max(np.abs(diff))

		if cand_max_change > max_changes[i]:
			max_changes[i] = cand_max_change

	return max_changes



def add_noise_to_params(params, bounds, scale=0.001):
	noise = onp.random.uniform(low = -1.0, high = 1.0,size=len(params)) # between 0 and 1
	new_params = params + (noise * scale) * (bounds[:,1] - bounds[:,0])
	new_params = np.clip(new_params,a_min=bounds[:,0],a_max=bounds[:,1])

	return new_params

def find_best_index(num_iters,bounds,selected_params, grads,max_ind, args, loss_func):
	LR = 1e-2
	init_loss = loss_func(selected_params, *args)
	prev_loss = init_loss
	orig_selected_params = copy.deepcopy(selected_params)
	for ii in range(num_iters):
		selected_params[max_ind] = orig_selected_params[max_ind] - grads[max_ind] * LR
		selected_params[max_ind] = onp.clip(selected_params[max_ind], bounds[max_ind][0], bounds[max_ind][1])
		loss = loss_func(selected_params, *args)

		if loss < prev_loss:
			return LR
		else:
			LR = LR * 0.1
	return LR



def minimize_coordinate_descent(num_iters,bounds, selected_params, args, grad_func, loss_func):
	line_search_cnt = 20
	selected_params = onp.array(selected_params)
	for ii in range(num_iters):
		grads = grad_func(selected_params, *args)
		grads = onp.array(grads)
		#max_ind = onp.argmax(onp.abs(grads))
		max_ind = onp.random.randint(low=0, high=len(grads))
		orig_selected_params = copy.deepcopy(selected_params)
		LR = find_best_index(line_search_cnt,bounds,selected_params, grads,max_ind, args, loss_func)
		selected_params[max_ind] = orig_selected_params[max_ind] - grads[max_ind] * LR
		selected_params[max_ind] = onp.clip(selected_params[max_ind], bounds[max_ind][0], bounds[max_ind][1])
		loss = loss_func(selected_params, *args)
		print("iter {}, loss: {}, LR: {}".format( ii, loss, LR))
		print(max_ind,selected_params[max_ind], grads[max_ind])
	return selected_params

def train_FF(orig_loss,loss_func,grad_func,minim_index_lists,subs,energy_minim_loss_and_grad_function,energy_minim_count,
			   energy_minim_init_LR,energy_minim_multip_LR,list_do_minim,list_num_minim_steps,end_RMSG,
			   selected_params,param_indices,bounds,flattened_force_field,flattened_non_dif_params,
			   min_weight, structured_training_data, params_list, epoch_count,
			   list_real_atom_counts,
			   list_positions_init,
			   list_all_shift_combs,
			   list_orth_matrices,
			   list_all_type,list_all_mask,
			   list_all_total_charge,
			   list_all_body_2_neigh_list,
			   list_all_dist_mat,
			   list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
			   list_all_body_3_list,list_all_body_3_map,list_all_body_3_angles,list_all_body_3_shift,
			   list_all_body_4_list,list_all_body_4_map,list_all_body_4_angles,list_all_body_4_shift,
			   list_all_hbond_list,list_all_hbond_mask,list_all_angles_and_dist,list_all_hbond_shift,
			   list_bond_rest,list_angle_rest,list_torsion_rest,inner_minim=0,minim_start_init=True,optimizer='L-BFGS-B', optim_options=dict()):


	print_info=False
	all_loss_values = []
	all_params = []
	prev_loss = 99999999999
	global_min = 99999999999
	global_min_params = np.array(copy.deepcopy(selected_params))

	def new_loss(selected_params,param_indices,flattened_force_field,flattened_non_dif_params,
								 structured_training_data,
								 list_positions,
								 list_all_type,list_all_mask,
								 list_all_total_charge,
							 	 list_all_shift_combs,
								 list_orth_matrices,
								 list_all_body_2_neigh_list,
								 list_all_dist_mat,
								 list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
								 list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,list_all_body_3_angles,
								 list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,list_all_body_4_angles,
								 list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,list_all_angles_and_dist,
								 list_bond_rest,list_angle_rest,list_torsion_rest,
								 energy_minim_loss_and_grad_function,
								 energy_minim_count,energy_minim_init_LR,
								 energy_minim_multip_LR,list_do_minim,list_num_minim_steps):
		if inner_minim == 1:
			flattened_force_field = jax.jit(use_selected_parameters,backend=DEVICE_NAME, static_argnums=(1))(selected_params,param_indices, flattened_force_field)
			flattened_force_field = jax.jit(preprocess_force_field,backend=DEVICE_NAME)(flattened_force_field, flattened_non_dif_params)

			energy_minim_count2 = 20
			energy_minim_init_LR2 = np.float32(1E-4)
			energy_minim_multip_LR2 = np.float32(1.0)

			list_positions,loss_vals,min_loss,minn_loss_vals,list_RMSG = energy_minimizer(energy_minim_loss_and_grad_function,energy_minim_count2,
							  energy_minim_init_LR2,energy_minim_multip_LR2,list_do_minim,list_num_minim_steps,end_RMSG,
							  flattened_force_field,flattened_non_dif_params,
							  list_real_atom_counts,
							 orig_list_pos,
							 list_positions, list_all_shift_combs,list_orth_matrices, list_all_type,list_all_mask,list_all_total_charge,
							 list_all_body_2_neigh_list,list_all_body_2_list,list_all_body_2_map,
							 list_all_body_2_trip_mask,list_all_body_3_list,
							 list_all_body_3_map,list_all_body_3_shift,
							 list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,
							 list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,
							 list_bond_rest,list_angle_rest,list_torsion_rest)

			[list_all_dist_mat,list_all_body_2_distances, list_all_body_3_angles, list_all_body_4_angles, list_all_angles_and_dist] = jax.jit(calculate_dist_and_angles, backend=DEVICE_NAME)(list_positions,list_orth_matrices,list_all_shift_combs,
																															  list_all_body_2_list,list_all_body_2_map,
																															  list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,
																															  list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,
																															  list_all_hbond_list,list_all_hbond_shift,list_all_hbond_mask)




		loss,grads = loss_func(selected_params,param_indices,flattened_force_field,flattened_non_dif_params,
								 structured_training_data,
								 list_positions,
								 list_all_type,list_all_mask,
								 list_all_total_charge,
							 	 list_all_shift_combs,
								 list_orth_matrices,
								 list_all_body_2_neigh_list,
								 list_all_dist_mat,
								 list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
								 list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,list_all_body_3_angles,
								 list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,list_all_body_4_angles,
								 list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,list_all_angles_and_dist,
								 list_bond_rest,list_angle_rest,list_torsion_rest,
								 list_do_minim,orig_list_pos,False)

		#is_in_bound = onp.all(selected_params >= bounds[:,0]) and onp.all(selected_params <= bounds[:,1])

		#if is_in_bound == False:
		#	loss = 9999999999

		return onp.asarray(loss,dtype=onp.float64), onp.asarray(np.nan_to_num(grads),dtype=onp.float64)

	#new_grad = lambda *x:onp.asarray(np.nan_to_num(grad_func(*x)),dtype=onp.float64)
	#new_loss = lambda *x:onp.asarray(loss_func(*x),dtype=onp.float64)
	minim_flag = sum([np.sum(l) for l in list_do_minim]) != 0 and energy_minim_count > 0
	bounds = []
	for p in params_list:
	    bounds.append((p[2],p[3]))
	bounds = onp.array(bounds)

	#list_positions = copy.deepcopy(list_positions_init)
	#list_positions = [np.array(p) for p in list_positions]
	total_f_ev = 0
	total_grad_ev = 0
	prev_loss = 99999999999
	current_loss = 99999999999
	orig_list_pos = []
	f_ev_list = []
	g_ev_list = []
	all_f_optim = []
	ep = 0
	args = (param_indices,flattened_force_field,flattened_non_dif_params,
								 structured_training_data,
								 list_positions_init,
								 list_all_type,list_all_mask,
								 list_all_total_charge,
							 	 list_all_shift_combs,
								 list_orth_matrices,
								 list_all_body_2_neigh_list,
								 list_all_dist_mat,
								 list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
								 list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,list_all_body_3_angles,
								 list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,list_all_body_4_angles,
								 list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,list_all_angles_and_dist,
								 list_bond_rest,list_angle_rest,list_torsion_rest,
								 energy_minim_loss_and_grad_function,
								 energy_minim_count,energy_minim_init_LR,
								 energy_minim_multip_LR,list_do_minim,list_num_minim_steps)

	#def callback(x):
	#	all_f_optim.append(ep, new_loss(x, *args)[0])

	#to scale the weights
#print('before training')
	list_positions = list_positions_init
	orig_list_pos = copy.deepcopy(list_positions_init)
	orig_list_pos = [np.array(p) for p in orig_list_pos]
	for i in range(epoch_count+1):


		epoch_start = time.time()
		#list_positions_init_mod = [l + onp.random.normal(scale=0.01,	size=l.shape) for l in list_positions_init]
		if minim_flag:
			flattened_force_field = jax.jit(use_selected_parameters,backend=DEVICE_NAME, static_argnums=(1))(selected_params,param_indices, flattened_force_field)
			flattened_force_field = jax.jit(preprocess_force_field,backend=DEVICE_NAME)(flattened_force_field, flattened_non_dif_params)
			if minim_start_init == False:
				list_positions_init = list_positions
			minim_start =time.time()
			list_positions,loss_vals,min_loss,minn_loss_vals,list_RMSG = energy_minim_with_subs(list_positions_init,minim_index_lists,
																						flattened_force_field,flattened_non_dif_params,subs,
																						energy_minim_loss_and_grad_function, energy_minim_count,
																						energy_minim_init_LR,energy_minim_multip_LR,end_RMSG)

			[list_all_dist_mat,list_all_body_2_distances, list_all_body_3_angles, list_all_body_4_angles, list_all_angles_and_dist] = jax.jit(calculate_dist_and_angles, backend=DEVICE_NAME)(list_positions,list_orth_matrices,list_all_shift_combs,
																														  list_all_body_2_list,list_all_body_2_map,
																														  list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,
																														  list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,
																														  list_all_hbond_list,list_all_hbond_shift,list_all_hbond_mask)
			minim_end = time.time()
			print("minim. took {}".format(minim_end-minim_start))
			print('Energy minim loss: {}'.format(min_loss))
			count = sum([onp.sum((l.reshape(-1)>end_RMSG) * minim_steps.reshape(-1) > 50) for l,minim_steps in zip(list_RMSG,subs[1])])
			print('RMSG > {:.1} count:{}'.format(float(end_RMSG),count))
			count = sum([onp.sum((l.reshape(-1)>2.5) * (minim_steps.reshape(-1) > 50)) for l,minim_steps in zip(list_RMSG,subs[1])])
			print('RMSG > {} count:{}'.format(2.5,count))
			count = sum([onp.sum((l.reshape(-1)>5.0) * (minim_steps.reshape(-1) > 50)) for l,minim_steps in zip(list_RMSG,subs[1])])
			print('RMSG > {} count:{}'.format(5.0,count))
			'''
			for i in range(len(list_RMSG)):
				for j in range(len(list_RMSG[i])):
					if list_RMSG[i][j] > end_RMSG and list_do_minim[i][j]:
						print(i,j,list_RMSG[i][j])
			'''
		else:
			list_positions = list_positions_init
		prev_loss = current_loss
		current_loss = orig_loss(selected_params,param_indices,flattened_force_field,flattened_non_dif_params,
								 structured_training_data,
								 list_positions,
								 list_all_type,list_all_mask,
								 list_all_total_charge,
							 	 list_all_shift_combs,
								 list_orth_matrices,
								 list_all_body_2_neigh_list,
								 list_all_dist_mat,
								 list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
								 list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,list_all_body_3_angles,
								 list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,list_all_body_4_angles,
								 list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,list_all_angles_and_dist,
								 list_bond_rest,list_angle_rest,list_torsion_rest,
								 list_do_minim,orig_list_pos,False)

		all_loss_values.append(current_loss)
		all_params.append(selected_params)
		print("current_loss", current_loss)
		if current_loss < global_min:
			global_min = current_loss
			global_min_params = np.array(copy.deepcopy(selected_params))
			print("global_min", global_min)

		if abs(current_loss-prev_loss) / current_loss < 0.001: #or current_loss > global_min * 1.5:
			selected_params = add_noise_to_params(global_min_params, bounds, scale=0.01)
			print('noise added')
			continue
		ep = i
		args = (param_indices,flattened_force_field,flattened_non_dif_params,
									 structured_training_data,
									 list_positions,
									 list_all_type,list_all_mask,
									 list_all_total_charge,
								 	 list_all_shift_combs,
									 list_orth_matrices,
									 list_all_body_2_neigh_list,
									 list_all_dist_mat,
									 list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
									 list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,list_all_body_3_angles,
									 list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,list_all_body_4_angles,
									 list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,list_all_angles_and_dist,
									 list_bond_rest,list_angle_rest,list_torsion_rest,
									 energy_minim_loss_and_grad_function,
									 energy_minim_count,energy_minim_init_LR,
									 energy_minim_multip_LR,list_do_minim,list_num_minim_steps)

		if True and i < epoch_count:

			min_state = minimize(new_loss, selected_params,jac=True,
							args=(param_indices,flattened_force_field,flattened_non_dif_params,
								 structured_training_data,
								 list_positions,
								 list_all_type,list_all_mask,
								 list_all_total_charge,
							 	 list_all_shift_combs,
								 list_orth_matrices,
								 list_all_body_2_neigh_list,
								 list_all_dist_mat,
								 list_all_body_2_list,list_all_body_2_map,list_all_body_2_trip_mask,list_all_body_2_distances,
								 list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,list_all_body_3_angles,
								 list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,list_all_body_4_angles,
								 list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,list_all_angles_and_dist,
								 list_bond_rest,list_angle_rest,list_torsion_rest,
								 energy_minim_loss_and_grad_function,
								 energy_minim_count,energy_minim_init_LR,
								 energy_minim_multip_LR,list_do_minim,list_num_minim_steps),
							method=optimizer,bounds=bounds,options=optim_options) #dict(maxiter=1000, disp=True,iprint = 1,maxls=40,maxcor=100))
			print("funv.ev {}, fun: {}".format(min_state.nfev,min_state.fun))
			f_ev_list.append(min_state.nfev)
			#g_ev_list.append(min_state.njev)

			total_f_ev = total_f_ev + min_state.nfev
			#total_grad_ev = total_grad_ev +  min_state.njev

			selected_params = np.array(min_state.x)

		epoch_end = time.time()
		print("Epoch-{} took {} sec".format(i,epoch_end-epoch_start))


	flattened_force_field = jax.jit(use_selected_parameters,backend=DEVICE_NAME, static_argnums=(1))(global_min_params,param_indices, flattened_force_field)
	print("total funv.ev {}".format(total_f_ev))


	return flattened_force_field,global_min_params,global_min,all_params,all_loss_values,f_ev_list,g_ev_list #,all_f_optim
