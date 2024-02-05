import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
import jax
jax.config.update("jax_enable_x64", True)
import jax.profiler
import jax.numpy as jnp
import numpy as onp
import argparse
from frozendict import frozendict
from jax_md.reaxff.reaxff_forcefield import ForceField
from jax_md.reaxff.reaxff_helper import read_force_field
from jax_md import dataclasses
from jaxreaxff.optimizer import calculate_energy_and_charges, update_inter_sizes
from jaxreaxff.helper import set_params, get_params
from jaxreaxff.interactions import (reaxff_interaction_list_generator,
                                    DYNAMIC_INTERACTION_KEYS)
from jaxreaxff.structure import align_and_batch_structures
from jaxreaxff.helper import (move_dataclass,
                              read_parameter_file,
                              map_params,
                              parse_and_save_force_field,
                              count_inter_list_sizes,
                              create_structure_map)
from jaxreaxff.helper_v2 import calculate_shift, loss_function, read_data, calculate_full_error

import math
from functools import partial
import optax
from tqdm import tqdm
import copy

def  main():
  # create parser for command-line arguments
  parser = argparse.ArgumentParser(description='JAX-ReaxFF driver')
  # default inputs: inital force field, parameters, geo and trainset files
  parser.add_argument('--init_FF', metavar='filename',
      type=str,
      default="ffield",
      help='Initial force field file')
  parser.add_argument('--params', metavar='filename',
      type=str,
      default="params",
      help='Parameters file')
  parser.add_argument('--data_file', metavar='filename',
      type=str,
      default="QM9.pickle",
      help='Pickled dataset')
  parser.add_argument('--num_epoch', metavar='number',
      type=int,
      default=100,
      help='Number of epoch')
  parser.add_argument('--batch_size', metavar='number',
      type=int,
      default=512,
      help='Batch size')
  parser.add_argument('--use_forces', metavar='boolean',
      type=bool,
      default=False,
      help='Flag to indicate use forces in the loss function') 
  parser.add_argument('--use_charges', metavar='boolean',
      type=bool,
      default=False,
      help='Flag to indicate use charges in the loss function') 
  parser.add_argument('--force_w', metavar='weight',
      type=float,
      default=0.1,
      help='Force weight in the loss function')
  parser.add_argument('--charge_w', metavar='weight',
      type=float,
      default=0.1,
      help='Charge weight in the loss function')
  parser.add_argument('--init_LR', metavar='init_LR',
      type=float,
      default=1e-3,
      help='Initial learning rate')
  # output related options
  parser.add_argument('--out_folder', metavar='folder',
      type=str,
      default="outputs",
      help='Folder to store the output files')
  parser.add_argument('--cutoff2', metavar='cutoff',
      type=float,
      default=0.001,
      help='BO-cutoff for valency angles and torsion angles')
  parser.add_argument('--seed', metavar='seed',
      type=int,
      default=0,
      help='Seed value')
  #parse arguments
  args = parser.parse_args()
  print("Arguments:", args)
  default_backend = jax.default_backend().lower()
  
  if default_backend == 'cpu':
      print("[WARNING] Falling back to CPU")
      print("To use the GPU version, jaxlib with CUDA support needs to installed!")
  
  
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
  for i in range(len(force_field.self_energies)):
    params_list.append((("self_energies", (i,)), 0.1, -500.0, 500.0))
  
  params_list.append((("shift", (0,)), 0.1, -500.0, 500.0))  
  
  # preprocess params
  param_indices=[]
  for par in params_list:
      param_indices.append(par[0])
  param_indices = tuple(param_indices)
  
  bounds = []
  for p in params_list:
      bounds.append((p[2],p[3]))
  bounds = jnp.array(bounds)
  # print INFO
  print(f"[INFO] Parameter file is read, there are {len(param_indices)} parameters to be optimized!")
  
  data = read_data(args.data_file, force_field)
  b_inds = onp.arange(len(data))
  onp.random.shuffle(b_inds)
  #b_inds = b_inds[:100]
  data = [data[i] for i in b_inds]
  print(f"[INFO] Dataset file is read, there are {len(data)} strucures!")
  
  geo_name_to_index, geo_index_to_name = create_structure_map(data)
  
  # replace names with indices
  for i,s in enumerate(data):
      s = dataclasses.replace(s, name = geo_name_to_index[s.name])
      data[i] = s
    
  thread_count = os.cpu_count()
  force_field = move_dataclass(force_field, onp)
  size_dicts = count_inter_list_sizes(data, force_field, num_threads=thread_count, chunksize=8)
  max_sizes = size_dicts[0]
  multip = 1.5
  for k in DYNAMIC_INTERACTION_KEYS:
    for s in size_dicts:
      # assign some buffer room
      s[k] = math.ceil(s[k] * multip)
  max_sizes = size_dicts[0]
  for k in max_sizes.keys():
    for s in size_dicts:
      max_sizes[k] = max(max_sizes[k], s[k])
  max_sizes = frozendict(max_sizes)
  print("[INFO] Interaction list sizes:")
  for item in max_sizes.items():
      print(item)
  
  force_field = move_dataclass(force_field, jnp)
  
  batch_size = args.batch_size
  data = align_and_batch_structures(data, max_sizes, batch_size=batch_size, dtype=TYPE)
  total_size = len(data)
  train_size = int(total_size * 0.8)
  train_data = data[:train_size]
  test_data = data[train_size:]
  
  
  batched_allocate = reaxff_interaction_list_generator(force_field,
                                        close_cutoff = 5.0,
                                        far_cutoff = 10.0,
                                        use_hbond=True)
  
  allocate_f = jax.jit(batched_allocate,static_argnums=(3,))
  get_params_jit = jax.jit(get_params,static_argnums=(1,))
  set_params_jit = jax.jit(set_params,static_argnums=(1,))
  energy_f = jax.jit(jax.vmap(calculate_energy_and_charges, (0,0,0,None)))
  energy_force_f = jax.jit(jax.vmap(jax.value_and_grad(calculate_energy_and_charges), (0,0,0,None)))
  new_loss_f = partial(loss_function, use_forces=args.use_forces, force_w=args.force_w, 
                                        use_charges=args.use_charges, charge_w=args.charge_w)
  grad_f = jax.jit(jax.grad(new_loss_f))
  
  
  total_steps = args.num_epoch * len(train_data)
  decay_scheduler = optax.piecewise_interpolate_schedule(interpolate_type="linear",
                                                         init_value=args.init_LR,
                                                         boundaries_and_scales={int(total_steps*0.25):1.0,
                                                                               int(total_steps*0.5):0.5,
                                                                               int(total_steps*0.75):0.3})
  
  force_field = calculate_shift(force_field, data, max_sizes, allocate_f, energy_f)
  selected_params = get_params_jit(force_field, param_indices)                                                                                                 
  
  optimizer = optax.chain(
        #optax.clip(10.0),
        optax.adam(decay_scheduler))
  
  opt_state = optimizer.init(selected_params)
  
  MSE, MAE, en_preds, en_targets = calculate_full_error(force_field, test_data, max_sizes, 
                           allocate_f, energy_force_f, energy_f,
                           args.use_forces, args.use_charges)
  all_MAE_te = []
  all_MSE_te = []  
  print(f"Test MAE: {round(MAE,2)}, MSE: {round(MSE,2)}", flush=True)
  all_MAE_te.append(MAE)
  all_MSE_te.append(MSE)
  
  best_FF = force_field
  lowest_err = MAE
  for ind in range(args.num_epoch):
    b_inds = onp.arange(len(train_data))
    onp.random.shuffle(b_inds)
    train_data = [train_data[i] for i in b_inds]
    
    for X in tqdm(train_data[:]):
      force_field_new = set_params_jit(force_field, param_indices, selected_params)
      nbr_lists, new_counts = allocate_f(X.positions, X, force_field_new, max_sizes)
      # extend the interaction list sizes if needed
      if jnp.any(nbr_lists.did_buffer_overflow):
        print("Interaction list overflow during training!")
        new_max_sizes = update_inter_sizes(X.positions,
                                           X,
                                           force_field_new,
                                           max_sizes,
                                           multip=multip)
        
        print("name: old size -> new size")
        for k in new_max_sizes.keys():
          if max_sizes[k] != new_max_sizes[k]:
            print(f"{k}: {max_sizes[k]}->{new_max_sizes[k]}")
        max_sizes = new_max_sizes
          
      grad_ff = grad_f(force_field_new, X, nbr_lists)
      grads = get_params_jit(grad_ff, param_indices)
      
      updates, opt_state = jax.jit(optimizer.update)(grads,
                                                 opt_state,
                                                 selected_params)
      selected_params = jax.jit(optax.apply_updates)(selected_params, updates)
      selected_params = jnp.clip(selected_params, bounds[:,0], bounds[:,1])
      
    if ind % 10 == 0:
      MSE, MAE, en_preds, en_targets = calculate_full_error(force_field_new, test_data, max_sizes, 
                         allocate_f, energy_force_f, energy_f,
                         args.use_forces, args.use_charges)
      print(f"Test MAE: {round(MAE,2)}, MSE: {round(MSE,2)}", flush=True)
      if MAE < lowest_err:
        best_FF = copy.deepcopy(force_field_new)
        lowest_err = MAE
      
      name = f"{args.out_folder}/ffield_{ind+1}"
      print(best_FF.self_energies)
      print(best_FF.shift)
      best_FF = move_dataclass(best_FF, onp)
      parse_and_save_force_field(args.init_FF, name, best_FF)
   
if __name__ == "__main__":
  main()
      

