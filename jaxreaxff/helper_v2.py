from jax_md.reaxff.reaxff_forcefield import ForceField
from jax_md.dataclasses import replace
import jax.numpy as jnp
import numpy as onp
import pickle
from jaxreaxff.structure import (Structure, BondRestraint, 
                                 AngleRestraint, TorsionRestraint)
from jaxreaxff.optimizer import calculate_energy_and_charges
import jax


def read_data(data_path, force_field):
  with open(data_path, 'rb') as fptr:
    data = pickle.load(fptr)
    structures = data['structures']
    self_energies = data['self_energies']
    systems = []
    orth_matrix = onp.eye(3,dtype=onp.float32) * 999
    shift_arr = onp.array([0,0,0],dtype=onp.int32).reshape(-1,3)
    total_charge = 0.0
    target_ch = None
    target_f = None
    for row in structures:
      N = len(row['species'])
      try:
        species = [force_field.name_to_index[t] for t in row['species']]
        my_self_e = sum([self_energies[t] for t in row['species']])
      except:
        continue
      if 'atomic_charges' in row:
        target_ch = row['atomic_charges']
      else:
        target_ch = None
      if 'total_charge' in row:
        total_charge = row['total_charge']
      else:
        total_charge = 0.0
      species = onp.array(species,dtype=onp.int32)
      species_str = onp.array(row['species'])
      atomic_nums = onp.zeros_like(species)
      atomic_nums = onp.where(species_str == "C", 6, atomic_nums)
      atomic_nums = onp.where(species_str == "O", 8, atomic_nums)
      atomic_nums = onp.where(species_str == "H", 1, atomic_nums)
      atomic_nums = onp.where(species_str == "N", 7, atomic_nums)

      target_e = row['energy'] - my_self_e
      if 'forces' in row:
        target_f = row['forces']
      else:
        target_f = None
      new_system = Structure(name="",
                             atom_count=N,
                             atom_types=species,
                             atomic_nums=atomic_nums,
                             positions=row['coordinates'],
                             orth_matrix=orth_matrix,
                             total_charge=total_charge,
                             energy_minimize=False,
                             energy_minim_steps=0,
                             periodic_image_shifts=shift_arr,
                             bond_restraints=None,
                             angle_restraints=None,
                             torsion_restraints=None,
                             target_e=target_e,
                             target_f=target_f,
                             target_ch=target_ch)
      systems.append(new_system)
    return systems

def calculate_indiv_energy(force_field, systems, max_sizes, allocate_f, energy_f):
  all_preds = []
  all_targets = []
  for X1 in systems:
    nbr_lists1, new_counts1 = allocate_f(X1.positions, X1, force_field, max_sizes)
    (energy1, charges1) = energy_f(X1.positions, X1, nbr_lists1, force_field)
    all_preds.append(onp.array(energy1).flatten())
    all_targets.append(onp.array(X1.target_e).flatten())
  all_preds = onp.hstack(all_preds)
  all_targets = onp.hstack(all_targets)

  return all_preds, all_targets

def calculate_shift(force_field, data, max_sizes, allocate_f, energy_f):
  force_field = ForceField.fill_off_diag(force_field)
  force_field = ForceField.fill_symm(force_field)
  force_field = replace(force_field, 
                        self_energies=force_field.self_energies * 0,
                        shift=force_field.shift * 0)
  TYPE = force_field.self_energies.dtype

  all_preds, all_targets = calculate_indiv_energy(force_field, data, 
                                                  max_sizes, allocate_f, energy_f)
  MAE = onp.mean(onp.abs(all_preds - all_targets))
  MSE = onp.mean(onp.abs(all_preds - all_targets)**2)
  print("Before the shift:")
  print(f"MAE: {round(MAE,2)}, MSE: {round(MSE,2)}", flush=True)
  print(force_field.self_energies, flush=True)
  print(force_field.shift, flush=True)
  # collect atom type matrix
  avaiable_types = onp.arange(len(force_field.self_energies))
  avaiable_types = sorted(avaiable_types)
  num_types = len(avaiable_types)
  final_result = []
  for X in data:
    result = onp.zeros((len(X.atom_types), num_types + 1))
    for i, t in enumerate(avaiable_types):
      count = jnp.sum(X.atom_types == t, axis=1)
      result[:, i] = onp.array(count)
    result[:,-1] = 1.0
    final_result.append(result)

  A, targ, pred = onp.vstack(final_result), onp.array(all_targets).flatten(), onp.array(all_preds).flatten()
  diff = targ - pred
  res = onp.linalg.lstsq(A, diff, rcond=None)
  new_self = jnp.array(res[0])
  new_self_full = onp.zeros(num_types, dtype=TYPE)
  new_self_full[:num_types] = new_self[:num_types]
  new_self_full = jnp.array(new_self_full)
  shift = new_self[-1:].astype(TYPE)
  force_field = replace(force_field,
                        self_energies=new_self_full,
                        shift=shift)
  print(force_field.self_energies, flush=True)
  print(force_field.shift, flush=True)
  all_preds, all_targets = calculate_indiv_energy(force_field, data,
                                                  max_sizes, allocate_f, energy_f)
  MAE = onp.mean(onp.abs(all_preds - all_targets))
  MSE = onp.mean(onp.abs(all_preds - all_targets)**2)
  print("After the shift:")
  print(f"MAE: {round(MAE,2)}, MSE: {round(MSE,2)}", flush=True)
  return force_field


def calculate_full_error(force_field, data, max_sizes, 
                         allocate_f, energy_force_f, energy_f,
                         use_forces, use_charges):
  count = 0
  en_preds = []
  en_targets = []
  force_targets = []
  force_preds = []
  charge_targets = []
  charge_preds = []
  total_atom_count = 0
  for X in data:
    Y = X.target_e
    nbr_lists, new_counts = allocate_f(X.positions, X, force_field, max_sizes)
    total_atom_count += int(jnp.sum(X.atom_count))
    if use_forces:
        (energy, charges), forces = energy_force_f(X.positions, X, nbr_lists, force_field)
        for i in range(len(Y)):
          target_f = X.target_f[i, :X.atom_count[i]]
          force_targets.append(onp.array(target_f).reshape(-1,3))
          pred_f = forces[i, :X.atom_count[i]]
          force_preds.append(onp.array(pred_f).reshape(-1,3))
    else:
        (energy, charges) = energy_f(X.positions, X, nbr_lists, force_field)
    count += len(Y)
    en_preds.append(onp.array(energy).flatten())
    en_targets.append(onp.array(Y).flatten())
    
    if use_charges:
      charge_preds.append(onp.array(charges).flatten())
      charge_targets.append(onp.array(X.target_ch).flatten())
      
      
  en_preds = onp.hstack(en_preds)
  en_targets = onp.hstack(en_targets)

  if use_forces:
    force_targets = onp.vstack(force_targets)
    force_preds = onp.vstack(force_preds)   
    force_err = onp.abs(force_targets - force_preds)

    MAFE = onp.mean(force_err)
    MSFE = onp.mean(force_err**2)
    MAFE = onp.round(MAFE, 3)
    MSFE = onp.round(MSFE, 3)
    print(f"Avg per atom force err MAE: {MAFE}, MSE: {MSFE}")
  if use_charges:
    charge_targets = onp.hstack(charge_targets)
    charge_preds = onp.hstack(charge_preds)
    ch_err = onp.abs(charge_targets - charge_preds)

    MACE = onp.mean(ch_err)
    MSCE = onp.mean(ch_err**2)
    MACE = onp.round(MACE, 3)
    MSCE = onp.round(MSCE, 3)
    print(f"Avg per atom charge err MAE: {MACE}, MSE: {MSCE}")
  
  MAE = onp.mean(onp.abs(en_preds - en_targets))
  MSE = onp.mean(onp.abs(en_preds - en_targets)**2)
  
  return MSE, MAE, en_preds, en_targets

def loss_function(force_field, structure, nbr_lists, 
                  use_forces=False, force_w=1.0, 
                  use_charges=False, charge_w=1.0):
  force_field = ForceField.fill_off_diag(force_field)
  force_field = ForceField.fill_symm(force_field)
  force_loss = 0
  charge_loss = 0
  if use_forces:
      atom_mask = structure.atom_types >= 0
      (energy_vals, charges), forces = jax.vmap(jax.value_and_grad(calculate_energy_and_charges, has_aux=True),
                                     (0,0,0,None))(structure.positions, structure, nbr_lists, force_field)
      force_err = (forces - structure.target_f) ** 2
      force_err = force_err * atom_mask[:,:, jnp.newaxis]
      force_loss = jnp.sum(force_err/(structure.atom_count.reshape(-1,1,1) * 3)) * force_w
  else:
      energy_vals, charges = jax.vmap(calculate_energy_and_charges,
                             (0,0,0,None))(structure.positions, structure, nbr_lists, force_field)
  if use_charges:
    charge_err = (charges - structure.target_ch) ** 2  
    charge_err = charge_err * atom_mask
    charge_loss = jnp.sum(charge_err/structure.atom_count.reshape(-1,1)) * charge_w
    
  target_energies = structure.target_e
  energy_err = (target_energies.flatten() - energy_vals.flatten())**2
  energy_loss = jnp.sum(energy_err / jnp.sqrt(structure.atom_count))

  return energy_loss + force_loss + charge_loss
