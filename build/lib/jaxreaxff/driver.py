#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver code to run the trainer

Author: Mehmet Cagri Kaymak
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
import jax
jax.config.update("jax_enable_x64", True)
import jax.profiler
import jax.numpy as jnp
import numpy as onp
import time
import argparse
from .smartformatter import SmartFormatter
from frozendict import frozendict
from jax_md.reaxff.reaxff_energy import calculate_reaxff_energy
from jax_md.reaxff.reaxff_forcefield import ForceField
from jax_md.reaxff.reaxff_helper import read_force_field
from jax_md import dataclasses
from jaxreaxff.optimizer import (calculate_loss, 
                                 calculate_energy_and_charges_w_rest, 
                                 add_noise_to_params, random_parameter_search, 
                                 train_FF, energy_minimize, update_inter_sizes) 
from jaxreaxff.helper import set_params, get_params, produce_error_report
from jaxreaxff.interactions import (reaxff_interaction_list_generator, 
                                    calculate_dist_and_angles, 
                                    DYNAMIC_INTERACTION_KEYS)
from jaxreaxff.structure import align_structures
from jaxreaxff.helper import (move_dataclass, process_and_cluster_geos, 
                              create_structure_map, read_parameter_file, 
                              map_params, read_geo_file, read_train_set, 
                              filter_data, structure_training_data,
                              parse_and_save_force_field)
import math
from functools import partial
from jaxreaxff.helper import build_float_range_checker

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
  parser.add_argument('--use_valid', metavar='boolean',
      type=bool,
      default=False,
      help='Flag indicating whether to use validation data (True/False)')
  parser.add_argument('--valid_file', metavar='filename',
      type=str,
      default="validset.in",
      help='Validation set file (same format as trainset.in)')
  parser.add_argument('--valid_geo_file', metavar='filename',
      type=str,
      default="valid_geo",
      help='Geo file for the validation data')
  # optimization related parameters
  parser.add_argument('--opt_method', metavar='method',
      choices=['L-BFGS-B', 'SLSQP'],
      type=str,
      default='L-BFGS-B',
      help='Optimization method - "L-BFGS-B" or "SLSQP"')
  parser.add_argument('--num_trials', metavar='number',
      type=int,
      default=1,
      help='R|Number of trials (Population size).\n' +
      'If set to <= 0, provided force field will be evaluated w/o any training (init_FF).')
  parser.add_argument('--num_steps', metavar='number',
      type=int,
      default=5,
      help='Number of optimization steps per trial')
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
      help='R|Before the optimization starts, uniforms sample the paramater space.\n' +
      'Select the best sample to start the training with, only works with "random" inital start.\n' +
      'if set to 0, no random search step will be skipped. ')
  # energy minimization related parameters
  parser.add_argument('--num_e_minim_steps', metavar='number',
      type=int,
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
  parser.add_argument('--bonded_cutoff', metavar='cutoff',
      type=float,
      default=5.0,
      help='Cutoff distance for bonded interactions (in Angstrom).')
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
  parser.add_argument('--perc_noise_when_stuck', metavar='percentage',
      type=build_float_range_checker(0.0, 0.1),
      default=0.04,
      help='R|Percentage of the noise that will be added to the parameters\n' +
           'when the optimizer is stuck.\n' +
           'param_noise_i = (param_min_i, param_max_i) * perc_noise_when_stuck\n' +
           'Allowed range: [0.0, 0.1]')
  parser.add_argument('--seed', metavar='seed',
      type=int,
      default=0,
      help='Seed value')
  
  #parse arguments
  args = parser.parse_args()
  # TODO: remove
  args.save_opt = "all"
  default_backend = jax.default_backend().lower()
  
  if default_backend == 'cpu':
    print("[WARNING] Falling back to CPU")
    print("To use the GPU version, jaxlib with CUDA support needs to installed!")
  
  # advanced options
  advanced_opts = {"perc_err_change_thr":0.01,                         # if change in error is less than this threshold, add noise
                   "perc_noise_when_stuck":args.perc_noise_when_stuck, # noise percantage (wrt param range) to add when stuck
                   "perc_width_rest_search":0.15,                      # width of the restricted parameter search after iteration > rest_search_start
                   }
  
  onp.random.seed(args.seed)
  TYPE = jnp.float64
  # read the initial force field
  force_field = read_force_field(args.init_FF, cutoff2 = args.cutoff2, dtype=TYPE)
  force_field = ForceField.fill_off_diag(force_field)
  force_field = ForceField.fill_symm(force_field)
  
  # print INFO
  print("[INFO] Force field field is read")
  ###########################################################################
  #read the paramemters to be optimized
  params_list_orig = read_parameter_file(args.params, ignore_sensitivity=0)
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
  print(f"[INFO] Parameter file is read, there are {len(param_indices)} parameters to be optimized!")
  ###########################################################################
  
  
  # read the geo file
  systems = read_geo_file(args.geo, force_field.name_to_index, 10.0)
  
  training_data = read_train_set(args.train_file)
  # default value for the valid. data
  validation_data = None
  systems_tr, training_data = filter_data(systems, training_data)
  # read and process the validation data if used
  if args.use_valid:
    print("[INFO] Validation data is provided!")
    systems_valid = read_geo_file(args.valid_geo_file, force_field.name_to_index, 10.0)
    validation_data = read_train_set(args.valid_file)
    systems_valid, validation_data = filter_data(systems_valid, validation_data)
    # combine training and validation data together (geo files)
    used_geo_names = set([s.name for s in systems_tr])
    systems = systems_tr
    for sys in systems_valid:
      if sys.name not in used_geo_names:
        systems.append(sys)
  else:
     systems = systems_tr
       
  geo_name_to_index, geo_index_to_name = create_structure_map(systems)
  training_data = structure_training_data(training_data, geo_name_to_index)
  if args.use_valid:
     validation_data = structure_training_data(validation_data, geo_name_to_index)
  # replace names with indices
  for i,s in enumerate(systems):
      s = dataclasses.replace(s, name = geo_name_to_index[s.name])
      systems[i] = s
  
  
  ###########################################################################
  num_threads = os.cpu_count()    
  [globally_sorted_indices, 
   all_cut_indices, 
   center_sizes] = process_and_cluster_geos(systems, force_field,
                                            max_num_clusters=args.max_num_clusters, 
                                            num_threads=num_threads, 
                                            chunksize=4,
                                            close_cutoff=args.bonded_cutoff, far_cutoff=10.0)
  for i in range(len(center_sizes)):
      for k in center_sizes[i].keys():
          if k in DYNAMIC_INTERACTION_KEYS:
            multip = 1.5
            # give extra buffer room if we need to e. minim
            if (k in ['filter3_size', 'filter4_size'] 
                and args.num_e_minim_steps > 0):
              multip = 2.0
            center_sizes[i][k] = math.ceil(multip * center_sizes[i][k])
          if center_sizes[i][k] == 0:
              center_sizes[i][k] = 1
  
  aligned_data = []
  for i in range(len(center_sizes)):
      zz = align_structures([systems[i] for i in all_cut_indices[i]], center_sizes[i], TYPE)
      zz = move_dataclass(zz, jnp)
      aligned_data.append(zz)
  
  force_field = move_dataclass(force_field, jnp)
  
  batched_allocate = reaxff_interaction_list_generator(force_field,
                                                       close_cutoff = args.bonded_cutoff,
                                                       far_cutoff = 10.0,
                                                       use_hbond=True)
  
  allocate_func = jax.jit(batched_allocate,static_argnums=(3,))
  center_sizes = [frozendict(c) for c in center_sizes]   
  
  list_positions = [s.positions for s in aligned_data]

  get_params_jit = jax.jit(get_params,static_argnums=(1,))
  set_params_jit = jax.jit(set_params,static_argnums=(1,))
  
  force_f = jax.jit(jax.vmap(jax.value_and_grad(calculate_energy_and_charges_w_rest,
                                            has_aux=True),
                         in_axes=(0,0,0, None)))
  
  minimize_kwargs = {"allocate_func":allocate_func, "force_func":force_f,
                     "init_LR":args.e_minim_LR, "minim_steps":args.num_e_minim_steps
                     , "target_RMSG":args.end_RMSG}
  minim_func = partial(energy_minimize, **minimize_kwargs)
  
  
  loss_and_grad_func = jax.jit(jax.value_and_grad(calculate_loss), 
                               static_argnames=('return_indiv_error',))
  loss_func = jax.jit(calculate_loss, static_argnames=('return_indiv_error',))
  
  
  def new_loss_and_grad_func(params, param_indices,
                             force_field, training_data,
                             list_positions, aligned_data, center_sizes):
    params = jnp.array(params)
    force_field = set_params_jit(force_field, param_indices, params)
    all_inters = [allocate_func(list_positions[i], aligned_data[i], 
                                force_field, center_sizes[i])[0] 
                  for i in range(len(center_sizes))]
    loss, grads_ff = loss_and_grad_func(force_field,
                                        list_positions,
                                        aligned_data,
                                        all_inters,
                                        training_data)
  
    grads = get_params_jit(grads_ff, param_indices)
    loss = onp.asarray(loss,dtype=onp.float64)
    grads = onp.asarray(grads,dtype=onp.float64)
  
    return loss, grads
  
  def new_loss_func(params, param_indices,
                    force_field, training_data,
                    list_positions, aligned_data, center_sizes,
                    return_indiv_error = False):
    params = jnp.array(params)
    force_field = set_params_jit(force_field, param_indices, params)
    all_inters = [allocate_func(list_positions[i], aligned_data[i], 
                                force_field, center_sizes[i])[0] 
                  for i in range(len(center_sizes))]
    results = loss_func(force_field,
                    list_positions,
                    aligned_data,
                    all_inters,
                    training_data,
                    return_indiv_error)
    if return_indiv_error:
      loss, indiv_errors = results
    else:
      loss = results
    loss = onp.asarray(loss, dtype=onp.float64)
    if return_indiv_error:
      return loss, indiv_errors
    return loss
  
  init_params = get_params(force_field, param_indices)
  init_params = onp.array(init_params)
  
  
  population_size = args.num_trials
  random_sample_count = args.random_sample_count
  results_list = []
  best_params = None
  best_fitness = float("inf")
  opt_method = args.opt_method
  num_steps = args.num_steps
  e_minim_flag = sum([jnp.sum(data.energy_minimize) for data in aligned_data]) > 0
  e_minim_flag = e_minim_flag & (args.num_e_minim_steps > 0)
  if opt_method == "L-BFGS-B":
      optim_options =dict(maxiter=100,maxls=20,maxcor=20, disp=False)
  else:
      optim_options =dict(maxiter=100, disp=False)
  
  for i in range(population_size):
    print('*' * 40)
    print("Trial-{} is starting...".format(i+1))
    start = time.time()
    if args.init_FF_type == 'random':
      min_params = random_parameter_search(bounds, random_sample_count,
                                  param_indices, force_field, training_data,
                                  list_positions, aligned_data, center_sizes,
                                  new_loss_func)
      selected_params = min_params
    elif args.init_FF_type == 'educated':
      selected_params = add_noise_to_params(init_params, bounds, scale=0.1)
    else: # fixed
      selected_params = jnp.array(init_params)
  
    [global_min_params,
     global_min,
     center_sizes] = train_FF(selected_params, param_indices, bounds, force_field,
                           aligned_data, center_sizes, training_data,
                           validation_data,
                           num_steps, e_minim_flag, opt_method, optim_options,
                           advanced_opts,
                           new_loss_and_grad_func, minim_func, allocate_func)
    end = time.time()
  
    result = {"time":end-start, "value": global_min, 
              "params": global_min_params,
              "unique_id":i+1}
    results_list.append(result)
  
    if best_fitness > global_min or best_params == None:
      best_fitness = global_min
      best_params = global_min_params
  
    print("Trial-{} ended, loss value: {:.2f}".format(i+1, global_min))
    print("Lowest loss so far        : {:.2f}".format(best_fitness))
  
  
  if not os.path.exists(args.out_folder):
    os.makedirs(args.out_folder)
  
  if args.save_opt == "all":
    results_to_save = results_list
  else:
    results_to_save = [{'params':best_params, 'value':best_fitness, 
                        "unique_id":"best"}]
  if population_size <= 0:
     print("[INFO] The population size <= 0, the initial force field is being evaluated...")
     results_to_save = [{'params':jnp.array(init_params), 'value':float('inf'), 
                        "unique_id":"init_ff"}]   
  
  for ii,res in enumerate(results_to_save):
    params = res['params']
    current_loss = res['value']
    unique_id = res['unique_id']
    force_field = set_params_jit(force_field, param_indices, params)
    if e_minim_flag:
      minim_start = time.time()
      [list_positions, cur_total_energy,
      center_sizes, cur_RMSG_vals] = minim_func(aligned_data,
                                                center_sizes,
                                                force_field)
      minim_end = time.time()
    else:
      # extend the interaction list sizes if needed
      for i in range(len(aligned_data)):
        sub_nbr = allocate_func(list_positions[i], aligned_data[i],
                                   force_field, center_sizes[i])[0]
        if jnp.any(sub_nbr.did_buffer_overflow):
          center_sizes[i] = update_inter_sizes(list_positions[i],
                                                   aligned_data[i],
                                                   force_field,
                                                   center_sizes[i],
                                                   multip=1.5)
  
    loss, indiv_errors = new_loss_func(params, param_indices,
                                      force_field, training_data,
                                      list_positions, aligned_data,
                                      center_sizes,
                                      True)
    for k in indiv_errors.keys():
      # move data to regular numpy arrays
      for i,sub_val in enumerate(indiv_errors[k]):
        indiv_errors[k][i] = onp.array(sub_val)
    loss = float(loss)
    loss_str = str(round(loss))
    new_name = "{}/new_FF_{}_{}".format(args.out_folder,unique_id,loss_str)
    new_force_field = move_dataclass(force_field, onp)
    parse_and_save_force_field(args.init_FF, new_name, new_force_field)
  
    report_name = "{}/report_{}_{}.txt".format(args.out_folder,unique_id,loss_str)
    produce_error_report(report_name, training_data, indiv_errors, geo_index_to_name)
  
    # produce the report for the validation data if available
    if args.use_valid:
      [valid_loss, 
       valid_indiv_errors] = new_loss_func(params, param_indices,
                                        force_field, validation_data,
                                        list_positions, aligned_data,
                                        center_sizes,
                                        True)
      for k in valid_indiv_errors.keys():
        # move data to regular numpy arrays
        for i,sub_val in enumerate(valid_indiv_errors[k]):
          valid_indiv_errors[k][i] = onp.array(sub_val)
      valid_loss = float(valid_loss)
      valid_loss_str = str(round(valid_loss))
      report_name = "{}/valid_report_{}_{}.txt".format(args.out_folder,unique_id,valid_loss_str)
      produce_error_report(report_name, validation_data, valid_indiv_errors, geo_index_to_name)       
         
if __name__ == "__main__":
  main()
