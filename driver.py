#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:33:40 2021

@author: cagri
"""
# again, this only works on startup!
from jax.config import config
import jax.profiler
#jax.profiler.start_server(9999)
#config.update("jax_enable_x64", True)
#config.update("jax_debug_nans", False)
#config.update("jax_log_compiles", 1)
import  os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
num_threads = str(os.cpu_count())
os.environ['MKL_NUM_THREADS']=num_threads
os.environ['OPENBLAS_NUM_THREADS']=num_threads

os.environ["NUM_INTER_THREADS"]="1"
os.environ["NUM_INTRA_THREADS"]=num_threads

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=true "
                           "intra_op_parallelism_threads={}".format(num_threads))

from force_field import ForceField,random_init_force_field,preprocess_force_field,TYPE
from simulation_system import SimulationSystem
from helper import parse_modified_params,map_params
from helper import structure_training_data,parse_geo_file,read_train_set
from helper import parse_force_field,parse_and_save_force_field,find_all_cutoffs,set_flattened_force_field
from reaxFF_local_optimizer import train_FF,energy_minimizer,jax_loss_vmap_new_test
from reaxFF_potential import jax_calculate_total_energy_for_minim_vmap,calculate_total_energy_single
from reaxFF_potential import calculate_total_energy_multi
from reaxFF_local_optimizer import use_selected_parameters,add_noise_to_params,select_energy_minim,get_minim_lists
import jax

#from jax.lib import xla_bridge
#print(jax.__version__)
#for multicore
import jax.numpy as np
import numpy as onp
import pickle
import time
from jax.experimental import optimizers
import sys
import matplotlib.pyplot as plt
import re
import copy
from helper import align_system_inter_lists,cluster_systems_for_aligning,pool_handler_for_inter_list_generation,process_and_cluster_geos
from multiprocessing import Pool

import argparse

DEVICE_NAME = 'gpu'

if __name__ == '__main__':
    # create parser for command-line arguments
    parser = argparse.ArgumentParser(description='JAX-ReaxFF driver',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # default inputs: inital force field, parameters, geo and trainset files
    parser.add_argument('--init_FF', metavar='filename',
        type=str,
        default="ffield",
        help='Initial force field file')
    parser.add_argument('--params', metavar='filename',
        type=str,
        default="params",
        help='Parameters file')
    parser.add_argument('--geo', metavar='filename',
        type=str,
        default="geo",
        help='Geometry file')
    parser.add_argument('--train_file', metavar='filename',
        type=str,
        default="trainset.in",
        help='Training set file')
    # optimization related parameters
    parser.add_argument('--opt_method', metavar='method',
        choices=['L-BFGS-B', 'SLSQP'],
        type=str,
        default='L-BFGS-B',
        help='Optimization method')
    parser.add_argument('--num_trials', metavar='number',
        type=int,
        choices=range(1, 1000),
        default=1,
        help='Number of trials')
    parser.add_argument('--num_steps', metavar='number',
        type=int,
        choices=range(1, 1000),
        default=20,
        help='Number of optimization steps per trial')
    parser.add_argument('--init_FF_type', metavar='number',
        choices=['random', 'educated'],
        default='random',
        help='''How to start the trials from the given initial force field.
        "random": Sample selected parameter from uniform distribution between given ranges.
        "educated": Sample the selected parameters from a narrow uniform distribution centered at the given value.''')
    # energy minimization related parameters
    parser.add_argument('--num_e_minim_steps', metavar='number',
        type=int,
        choices=range(1, 5000),
        default=200,
        help='Number of energy minimization steps')
    parser.add_argument('--e_minim_LR', metavar='init_LR',
        type=float,
        default=1e-4,
        help='Initial learning rate for energy minimization')
    parser.add_argument('--end_RMSG', metavar='end_RMSG',
        type=float,
        default=1.0,
        help='Stopping condition for E. minimization')
    # output related options
    parser.add_argument('--out_folder', metavar='folder',
        type=str,
        default="outputs",
        help='Folder to store the output files')

    #parse arguments
    args = parser.parse_args()

    # read the initial force field
    force_field = parse_force_field(args.init_FF)
    force_field.init_params_for_filler_atom_type()
    force_field.flatten()

    # FF preprocessing
    force_field.flattened_force_field = preprocess_force_field(force_field.flattened_force_field, force_field.non_dif_params)

    # print INFO
    print("[INFO] Force field field is read")
    ###########################################################################
    #read the paramemters to be optimized
    params_list_orig = parse_modified_params(args.params, ignore_sensitivity=0)
    params_list = map_params(params_list_orig, force_field.params_to_indices)

    # preprocess params
    param_indices=[]
    for par in params_list:
    	param_indices.append(par[0])

    bounds = []
    for p in params_list:
        bounds.append((p[2],p[3]))
    bounds = onp.array(bounds)
    # print INFO
    print("[INFO] Parameter file is read, there are {} parameters to be optimized!".format(len(param_indices)))
    ###########################################################################
    # read the geo file
    systems = parse_geo_file(args.geo)

    do_minim_count = 0
    for s in systems:
    	do_minim_count += s.num_min_steps > 2
    	#print(s.name,s.num_min_steps)

    # print INFO
    print("[INFO] Geometry file is read, there are {} geometries and {} require energy minimization!".format(len(systems), do_minim_count))


    (ordered_systems,[list_all_type,
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
                     list_all_angles_and_dist]) = process_and_cluster_geos(systems,force_field,param_indices,bounds)
    ###########################################################################

    all_training_items = read_train_set(args.train_file)
    print("[INFO] trainset file is read, there are {} items".format(len(all_training_items)))

    structured_training_data = structure_training_data(ordered_systems, all_training_items)
    ###########################################################################

    # put force field in device memory
    for i in range(len(force_field.flattened_force_field)):
    	force_field.flattened_force_field[i] = jax.device_put(force_field.flattened_force_field[i])
    for i in range(len(force_field.non_dif_params)):
    	force_field.non_dif_params[i] = jax.device_put(force_field.non_dif_params[i])

    flattened_force_field = force_field.flattened_force_field
    flattened_non_dif_params = force_field.non_dif_params


    selected_params = [flattened_force_field[par[0][0]][par[0][1]] for par in params_list]
    selected_params = np.array(selected_params, dtype=TYPE)
    selected_params_init = selected_params



    orig_loss = jax.jit(jax_loss_vmap_new_test,static_argnums=(1,3),backend=DEVICE_NAME)
    loss_func = jax.jit(jax_loss_vmap_new_test,static_argnums=(1,3),backend=DEVICE_NAME)
    grad_func = jax.jit(jax.grad(jax_loss_vmap_new_test),static_argnums=(1,3),backend=DEVICE_NAME)
    loss_and_grad = jax.jit(jax.value_and_grad(jax_loss_vmap_new_test),static_argnums=(1,3),backend=DEVICE_NAME)


    grad_and_loss_func = energy_minim_loss_and_grad_function = jax.jit(jax.vmap(jax.value_and_grad(jax_calculate_total_energy_for_minim_vmap),
    													   in_axes=(0,None,None,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
    													   static_argnums=(4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22),backend=DEVICE_NAME)

    orig_loss2 = jax_loss_vmap_new_test

    def new_g(*x):
    	grads = grad_func(*x)
    	grads = grads / 1e2 #works for silica
    	return grads

    def new_loss_and_grad(*x):
    	val,grads = loss_and_grad(*x)
    	grads = grads / 1e2 #works for silica
    	return val, grads

    grad_func_w_noise = lambda *x:np.clip(grad_func(*x),-1e8,1e8)
    orig_list_all_pos = copy.deepcopy(list_all_pos)
    orig_list_all_pos = [np.array(a) for a in orig_list_all_pos]
    minim_index_lists = select_energy_minim(list_do_minim)

    subs = jax.jit(get_minim_lists, static_argnums=(0,), backend=DEVICE_NAME)(minim_index_lists,list_do_minim, list_num_minim_steps,
    										 list_real_atom_counts,
    										 orig_list_all_pos,list_all_pos, list_all_shift_combs,list_orth_matrices,
    										 list_all_type,list_all_mask,
    										 list_all_total_charge,
    									 list_all_body_2_neigh_list,list_all_body_2_list,list_all_body_2_map,
    									 list_all_body_2_trip_mask,list_all_body_3_list,
    									 list_all_body_3_map,list_all_body_3_shift,
    									 list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,
    									  list_all_hbond_list,list_all_hbond_mask,list_all_hbond_shift,
    									 list_bond_rest,list_angle_rest,list_torsion_rest)

    energy_minim_count = args.num_e_minim_steps
    energy_minim_init_LR = np.float32(args.e_minim_LR)
    energy_minim_multip_LR = np.float32(1.0**(1.0/(energy_minim_count+1e-10)))
    end_RMSG = np.float32(args.end_RMSG)
    population_size = args.num_trials
    min_weight = 1.0
    results_list = []

    opt_method = args.opt_method
    # Options for LBFGS
    optim_options =dict(maxiter=100,maxls=20,maxfev=1000,maxcor=20, disp=False)
    for i in range(population_size):
        if args.init_FF_type == 'random':
            selected_params = onp.random.uniform(low=bounds[:,0],high=bounds[:,1])
            selected_params = np.array(selected_params, dtype=TYPE)
        else:
            selected_params = add_noise_to_params(selected_params_init,bounds,scale=0.1)
        s=time.time()
        flattened_force_field,global_min_params,global_min,all_params,all_loss_values,f_ev_list,g_ev_list = train_FF(orig_loss,loss_and_grad,grad_func,
    													 minim_index_lists,subs,energy_minim_loss_and_grad_function,energy_minim_count,
    												   energy_minim_init_LR,energy_minim_multip_LR,list_do_minim,list_num_minim_steps,end_RMSG,
    												   selected_params,param_indices,bounds, flattened_force_field,flattened_non_dif_params,
    												   min_weight, structured_training_data, params_list,20,
    												   list_real_atom_counts,
    												   list_all_pos,
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
    												   list_bond_rest,list_angle_rest,list_torsion_rest,inner_minim=0,minim_start_init=True,optimizer=opt_method,optim_options=optim_options)
        e=time.time()
        result = {"time":e-s, "value": global_min, "params": global_min_params}
        results_list.append(result)
