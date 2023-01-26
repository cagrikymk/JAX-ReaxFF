#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver code to run the trainer

Author: Mehmet Cagri Kaymak
"""
import jax.profiler
import  os
from jaxreaxff.forcefield import preprocess_force_field, TYPE
from jaxreaxff.reaxffpotential import calculate_total_energy_for_minim
from jaxreaxff.optimizer import use_selected_parameters,add_noise_to_params
from jaxreaxff.optimizer import select_energy_minim,get_minim_lists
from jaxreaxff.optimizer import train_FF,loss_w_sel_params
from jaxreaxff.helper import process_and_cluster_geos
from jaxreaxff.helper import parse_modified_params,map_params,filter_geo_items
from jaxreaxff.helper import structure_training_data,parse_geo_file,read_train_set
from jaxreaxff.helper import parse_force_field,parse_and_save_force_field
from jaxreaxff.helper import produce_error_report
import jax
from jaxreaxff.optimizer import energy_minim_with_subs,calculate_dist_and_angles
import jax.numpy as np
import numpy as onp
import time
import copy
import argparse, textwrap
from jaxreaxff.smartformatter import SmartFormatter
from jaxreaxff.myjit import my_jit



def main():
    # create parser for command-line arguments
    parser = argparse.ArgumentParser(description='JAX-ReaxFF driver',
                                     formatter_class=SmartFormatter)
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
        help='Optimization method - "L-BFGS-B" or "SLSQP"')
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
    parser.add_argument('--rest_search_start', metavar='number',
        type=int,
        choices=range(-1, 1001),
        default=-1,
        help='R|Restrict the search space after epoch > rest_search_start.\n' +
        '-1 means to restricted search')
    parser.add_argument('--init_FF_type', metavar='init_type',
        choices=['random', 'educated', 'fixed'],
        default='fixed',
        help='R|How to start the trials from the given initial force field.\n' +
        '"random": Sample the parameters from uniform distribution between given ranges.\n'
        '"educated": Sample the parameters from a narrow uniform distribution centered at given values.\n'
        '"fixed": Start from the parameters given in "init_FF" file')
    parser.add_argument('--random_sample_count', metavar='number',
        type=int,
        default=0,
        help='R|Number of samples for the random parameter search before the gradient step.\n' +
        'Only applicable to the "random" initial start, ignored otherwise.')
    # energy minimization related parameters
    parser.add_argument('--num_e_minim_steps', metavar='number',
        type=int,
        choices=range(0, 5000),
        default=0,
        help='Number of energy minimization steps')
    parser.add_argument('--e_minim_LR', metavar='init_LR',
        type=float,
        default=5e-4,
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
    parser.add_argument('--save_opt', metavar='option',
        choices=['all', 'best'],
        default="best",
        help='R|"all" or "best"\n' +
        '"all": save all of the trained force fields\n' +
        '"best": save only the best force field')
    parser.add_argument('--cutoff2', metavar='cutoff',
        type=float,
        default=0.001,
        help='BO-cutoff for valency angles and torsion angles')
    parser.add_argument('--max_num_clusters', metavar='max # clusters',
        type=int,
        default=10,
        choices=range(1, 16),
        help='R|Max number of clusters that can be used\n' +
             'High number of clusters lowers the memory cost\n' +
             'However, it increases compilation time,especially for cpus')
    parser.add_argument('--num_threads', metavar='# threads',
        type=int,
        default=-1,
        help='R|Number of threads to use to preprocess the data\n' +
             '-1: # threads = # available cpu cores * 2')
    parser.add_argument('--backend', metavar='backend',
        type=str,
        choices=['gpu', 'cpu', 'tpu'],
        default='gpu',
        help='Backend for JAX')

    parser.add_argument('--seed', metavar='seed',
        type=int,
        default=0,
        help='Seed value')
                  
    #parse arguments
    args = parser.parse_args()
    device_name = args.backend.lower()

    print("Selected backend for JAX:",device_name.upper())
    default_backend = jax.default_backend().lower()

    if device_name == 'gpu' and default_backend == 'cpu':
        print("[WARNING] selected backend({}) is not available!".format(device_name.upper()))
        print("[WARNING] Falling back to CPU")
        print("To use the GPU version, jaxlib with CUDA support needs to installed!")
        device_name = default_backend
        
    if args.random_sample_count < 0:
        print("[WARNING] random_sample_count cannot be less than 0, setting it to 0 to disable the random search!")
        args.random_sample_count = 0 

    # advanced options
    advanced_opts = {"perc_err_change_thr":0.01,       # if change in error is less than this threshold, add noise
                     "perc_noise_when_stuck":0.01,     # noise percantage (wrt param range) to add when stuck
                     "perc_width_rest_search":0.05,    # width of the restricted parameter search after iteration > rest_search_start
                     "rest_search_start":args.rest_search_start, #estrict the search space after epoch > rest_search_start, ignore if -1
                     "backend":device_name
                     }

    onp.random.seed(args.seed)

    # read the initial force field
    force_field = parse_force_field(args.init_FF,cutoff2 = args.cutoff2)
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
    param_indices = tuple(param_indices)

    bounds = []
    for p in params_list:
        bounds.append((p[2],p[3]))
    bounds = onp.array(bounds)
    # print INFO
    print("[INFO] Parameter file is read, there are {} parameters to be optimized!".format(len(param_indices)))
    ###########################################################################

    all_training_items,all_training_items_str = read_train_set(args.train_file)
    total_num_items = sum([len(all_training_items[key]) for key in all_training_items])
    print("[INFO] trainset file is read, there are {} items".format(total_num_items))
    for key in all_training_items:
        print("{}:{}".format(key, len(all_training_items[key])))

    ###########################################################################
    # read the geo file
    systems = parse_geo_file(args.geo)

    do_minim_count = 0
    for s in systems:
        do_minim_count += s.num_min_steps > 1
        #print(s.name,s.num_min_steps)

    # print INFO
    print("[INFO] Geometry file is read, there are {} geometries and {} require energy minimization!".format(len(systems), do_minim_count))

    systems = filter_geo_items(systems, all_training_items)
    do_minim_count = 0
    for s in systems:
        do_minim_count += s.num_min_steps > 1
    print("After removing geometries that are not used in the trainset file:")
    print("[INFO] Geometry file is read, there are {} geometries and {} require energy minimization!".format(len(systems), do_minim_count))
    ###########################################################################
    num_threads = os.cpu_count() * 2 # twice the number of availabe cores
    if args.num_threads > 0:
        num_threads = args.num_threads

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
                     list_all_angles_and_dist]) = process_and_cluster_geos(systems,force_field,
                                                                           param_indices,
                                                                           bounds,
                                                                           max_num_clusters=args.max_num_clusters,
                                                                           num_threads=num_threads)
    ###########################################################################


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



    loss_func = my_jit(loss_w_sel_params,static_argnums=(1,33),
                        static_list_of_array_argnums=(6,),backend=advanced_opts['backend'])
    grad_func = my_jit(jax.grad(loss_w_sel_params),static_argnums=(1,33),
                        static_list_of_array_argnums=(6,),backend=advanced_opts['backend'])
    loss_and_grad = my_jit(jax.value_and_grad(loss_w_sel_params),static_argnums=(1,33),
                            static_list_of_array_argnums=(6,),backend=advanced_opts['backend'])

    energy_minim_loss_and_grad_function = jax.jit(jax.vmap(jax.value_and_grad(calculate_total_energy_for_minim),
                                                           in_axes=(0,None,None,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)),
                                                 backend=device_name)

    my_args =  (param_indices, flattened_force_field, flattened_non_dif_params,
               structured_training_data,
               list_all_pos,
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
               list_do_minim)

    def new_grad(*x):
        grads = grad_func(*x)
        return grads

    def new_loss_and_grad(*x):
        val,grads = loss_and_grad(*x)
        return onp.asarray(val,dtype=onp.float64), onp.asarray(np.nan_to_num(grads),dtype=onp.float64)

    def new_loss_func(params, args):
        loss = loss_func(params, *args)
        return onp.float64(loss)

    # copy the original atom positions
    orig_list_all_pos = copy.deepcopy(list_all_pos)
    orig_list_all_pos = [np.array(a) for a in orig_list_all_pos]
    # indices that require energy minimization
    minim_index_lists = select_energy_minim(list_do_minim)

    subsets_with_en_minim = my_jit(get_minim_lists, static_list_of_array_argnums=(0,),
                                    backend=advanced_opts['backend'])(minim_index_lists,list_do_minim, list_num_minim_steps,
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
    results_list = []
    best_FF = None
    best_fitness = 10**20
    opt_method = args.opt_method
    num_steps = args.num_steps
    # remove later
    # Options for LBFGS
    if opt_method == "L-BFGS-B":
        optim_options =dict(maxiter=100,maxls=20,maxcor=20, disp=False)
    else:
        optim_options =dict(maxiter=100, disp=False)
    for i in range(population_size):
        print('*' * 40)
        print("Trial-{} is starting...".format(i+1))
        if args.init_FF_type == 'random':
            min_loss = float('inf')
            min_params = None
            for _ in range(args.random_sample_count):
                selected_params = onp.random.uniform(low=bounds[:,0],high=bounds[:,1])
                selected_params = np.array(selected_params, dtype=TYPE)
                loss = new_loss_func(selected_params, my_args)
                if loss < min_loss or onp.isnan(min_loss) == True:
                    min_loss = loss
                    min_params = selected_params
            if min_params != None and onp.isnan(min_loss) == False:
                selected_params = min_params
                print("Loss after random search (w/o energy minim.): ", min_loss)
            else:
                selected_params = onp.random.uniform(low=bounds[:,0],high=bounds[:,1])
                selected_params = np.array(selected_params, dtype=TYPE)
        elif args.init_FF_type == 'educated':
            selected_params = add_noise_to_params(selected_params_init,bounds,scale=0.1)
        else: # if init_FF_type == 'fixed'
            selected_params = np.array(selected_params_init)

        s=time.time()
        flattened_force_field,global_min_params,global_min,all_params,all_loss_values,f_ev_list,g_ev_list = train_FF(loss_func,new_loss_and_grad,grad_func,
                                                       minim_index_lists,subsets_with_en_minim,energy_minim_loss_and_grad_function,energy_minim_count,
                                                       energy_minim_init_LR,energy_minim_multip_LR,list_do_minim,list_num_minim_steps,end_RMSG,
                                                       selected_params,param_indices,bounds, flattened_force_field,flattened_non_dif_params,
                                                       structured_training_data, params_list,num_steps,
                                                       advanced_opts,
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
                                                       list_bond_rest,list_angle_rest,list_torsion_rest,
                                                       inner_minim=0,minim_start_init=True,
                                                       optimizer=opt_method,optim_options=optim_options)

        e=time.time()
        result = {"time":e-s, "value": global_min, "params": global_min_params}
        results_list.append(result)

        if best_fitness > global_min or best_FF == None:
            best_fitness = global_min
            best_FF = result

        print("Trial-{} ended, error value: {:.2f}".format(i+1, global_min))

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    if args.save_opt == "all":
        for i,res in enumerate(results_list):
            params = res['params']
            current_loss = res['value']
            flattened_force_field = jax.jit(use_selected_parameters,backend=advanced_opts['backend'], static_argnums=(1))(params,param_indices, flattened_force_field)
            force_field.flattened_force_field = flattened_force_field
            force_field.unflatten()

            flattened_force_field = jax.jit(preprocess_force_field,backend=advanced_opts['backend'])(flattened_force_field, flattened_non_dif_params)

            minim_flag = sum([np.sum(l) for l in list_do_minim]) != 0 and energy_minim_count > 0

            if minim_flag:

                list_positions,loss_vals,min_loss,minn_loss_vals,list_RMSG = energy_minim_with_subs(orig_list_all_pos,minim_index_lists,
                                                                                            flattened_force_field,flattened_non_dif_params,subsets_with_en_minim,
                                                                                            energy_minim_loss_and_grad_function, energy_minim_count,
                                                                                            energy_minim_init_LR,energy_minim_multip_LR,end_RMSG,
                                                                                            advanced_opts)




                [list_all_dist_mat,list_all_body_2_distances,
                list_all_body_3_angles,
                list_all_body_4_angles,
                list_all_angles_and_dist] = jax.jit(calculate_dist_and_angles, backend=advanced_opts['backend'])(list_positions,list_orth_matrices,list_all_shift_combs,
                                                                                                  list_all_body_2_list,list_all_body_2_map,
                                                                                                  list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,
                                                                                                  list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,
                                                                                                  list_all_hbond_list,list_all_hbond_shift,list_all_hbond_mask)
            else:
                list_positions = orig_list_all_pos

            current_loss,indiv_error = loss_func(params,param_indices,flattened_force_field,flattened_non_dif_params,
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
                                     list_do_minim,True)

            new_name = "{}/new_FF_{}_{:.2f}".format(args.out_folder,i+1,current_loss)
            parse_and_save_force_field(args.init_FF, new_name, force_field)

            report_name = "{}/report_{}_{:.2f}".format(args.out_folder,i+1,current_loss)

            produce_error_report(report_name, all_training_items,all_training_items_str, indiv_error)
    else:
        params = best_FF['params']
        current_loss = best_FF['value']
        flattened_force_field = jax.jit(use_selected_parameters,backend=advanced_opts['backend'], static_argnums=(1))(params,param_indices, flattened_force_field)
        force_field.flattened_force_field = flattened_force_field
        force_field.unflatten()

        minim_flag = sum([np.sum(l) for l in list_do_minim]) != 0 and energy_minim_count > 0
        flattened_force_field = jax.jit(preprocess_force_field,backend=advanced_opts['backend'])(flattened_force_field, flattened_non_dif_params)

        if minim_flag:

            list_positions,loss_vals,min_loss,minn_loss_vals,list_RMSG = energy_minim_with_subs(orig_list_all_pos,minim_index_lists,
                                                                                        flattened_force_field,flattened_non_dif_params,subsets_with_en_minim,
                                                                                        energy_minim_loss_and_grad_function, energy_minim_count,
                                                                                        energy_minim_init_LR,energy_minim_multip_LR,end_RMSG,
                                                                                        advanced_opts)



            [list_all_dist_mat,list_all_body_2_distances,
            list_all_body_3_angles,
            list_all_body_4_angles,
            list_all_angles_and_dist] = jax.jit(calculate_dist_and_angles, backend=advanced_opts['backend'])(list_positions,list_orth_matrices,list_all_shift_combs,
                                                                                              list_all_body_2_list,list_all_body_2_map,
                                                                                              list_all_body_3_list,list_all_body_3_map,list_all_body_3_shift,
                                                                                              list_all_body_4_list,list_all_body_4_map,list_all_body_4_shift,
                                                                                              list_all_hbond_list,list_all_hbond_shift,list_all_hbond_mask)
        else:
            list_positions = orig_list_all_pos

        current_loss,indiv_error = loss_func(params,param_indices,flattened_force_field,flattened_non_dif_params,
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
                                 list_do_minim,True)

        new_name = "{}/best_FF_{:.2f}".format(args.out_folder,current_loss)
        parse_and_save_force_field(args.init_FF, new_name, force_field)

        report_name = "{}/best_report_{:.2f}".format(args.out_folder,current_loss)
        produce_error_report(report_name, all_training_items,all_training_items_str, indiv_error)


if __name__ == "__main__":
    main()
