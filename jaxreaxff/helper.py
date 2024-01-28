"""
Contains helper functions for I/O and training

Author: Mehmet Cagri Kaymak
"""

import  os
import jax.numpy as jnp
import numpy as onp
import time
import sys
from multiprocessing import get_context
from tabulate import tabulate
import math
from jaxreaxff.clustering import modified_kmeans
from jaxreaxff.trainingdata import ChargeItem, EnergyItem, DistItem, AngleItem 
from jaxreaxff.trainingdata import TorsionItem, ForceItem, RMSGItem, TrainingData
from jaxreaxff.structure import Structure, BondRestraint, AngleRestraint, TorsionRestraint
from jaxreaxff.inter_list_counter import pool_handler_for_inter_list_count
from jax_md import dataclasses
from jax_md.reaxff.reaxff_forcefield import ForceField
# Since we shouldnt access the private API (jaxlib), create a dummy jax array
# and get the type information from the array.
#from jaxlib.xla_extension import ArrayImpl as JaxArrayType
JaxArrayType = type(jnp.zeros(1))

def get_params(force_field, params_list):
  '''
  Get the selected parameters from the force field
  '''
  res = []
  for param in params_list:
    name = param[0]
    index = param[1]
    x = getattr(force_field, name)[index]
    res.append(x)
  return jnp.array(res)

def set_params(force_field, params_list, params):
  '''
  Replace the selected parameters in the force field
  '''
  attr_dict = dict()
  for i,param in enumerate(params_list):
    name = param[0]
    index = param[1]
    x = getattr(force_field, name)
    attr_dict[name] = x
  for i,param in enumerate(params_list):
    name = param[0]
    index = param[1]
    attr_dict[name] = attr_dict[name].at[index].set(params[i])
  new_ff = dataclasses.replace(force_field, **attr_dict)
  new_ff = ForceField.fill_off_diag(new_ff)
  new_ff = ForceField.fill_symm(new_ff)
  return new_ff

def split_dataclass(data):
  '''
  From a dataclass with batched atrributes, seperate each sample
  and create a list of samples
  '''
  result = []
  field_names = [field.name for field in dataclasses.fields(data)]
  val = getattr(data, field_names[0])
  size = len(val)
  for i in range(size):
    sub = filter_dataclass(data, i)
    result.append(sub)
  return result

def filter_dataclass(data, filter_map):
  '''
  Apply a given filter to a dataclass with batched atrributes
  '''
  sel_dict = {}
  field_names = [field.name for field in dataclasses.fields(data)]
  d_class = data.__class__
  for attr in field_names:
    val = getattr(data, attr)
    if type(val) in [JaxArrayType, onp.ndarray]:
      sel_dict[attr] = val[filter_map]
    # recursive filtering since dataclass might contain other dataclasses
    # as an attribute
    #TODO: if there is self reference, this will cause stack overflow
    if dataclasses.is_dataclass(val):
      sel_dict[attr] = filter_dataclass(val, filter_map)

  return d_class(**sel_dict)

def move_dataclass(obj, target_numpy):
  '''
  Move a given dataclass object to target numpy (either onp or jnp)
  '''
  field_names = [field.name for field in dataclasses.fields(obj)]
  replace_dict = dict()
  for attr in field_names:
    val = getattr(obj, attr)
    if type(val) in [JaxArrayType, onp.ndarray]:
      replace_dict[attr] = target_numpy.array(val)
    #TODO: if there is self reference, this will cause stack overflow
    if dataclasses.is_dataclass(val):
      replace_dict[attr] = move_dataclass(val, target_numpy)
  new_obj = dataclasses.replace(obj, **replace_dict)
  return new_obj

def orthogonalization_matrix(box_lengths, angles_degr):
  '''
  Calculate a transformation matrix to be used in distance calculations
  with periodic boundary condition
  '''
  # to calculate shifted box coord: mat.dot(shift_array)
  # source: reaxff frotran, subroutine vlist
  a,b,c = box_lengths
  angles = onp.radians(angles_degr)
  sina, sinb, sing = onp.sin(angles)
  cosa, cosb, cosg = onp.cos(angles)
  cosphi = (cosg - cosa * cosb)/(sina * sinb)
  if cosphi >  1.0:
    cosphi = 1.0
  sinphi = onp.sqrt(1.0 - cosphi*cosphi)
  #tm11,tm21,tm31,tm22,tm32,tm33
  mat =  onp.array((
          (a * sinb * sinphi,       0.0,                0.0),
          (a * sinb * cosphi,      b * sina,           0.0),
          (a *  cosb,              b * cosa,            c)),
          dtype=jnp.float32)

  if angles_degr[0] == 90.0 and angles_degr[1] == 90.0 and angles_degr[2] == 90.0:
    mat = onp.eye(3, dtype=jnp.float32)
    mat[0,0] = a
    mat[1,1] = b
    mat[2,2] = c
  return mat

def calculate_box_shifts(is_periodic, far_nbr_cutoff, orth_mat):
  '''
  Find all different boxes we can see given the cutoff
  '''
  if is_periodic == True:
    kx_limit = math.ceil(far_nbr_cutoff / orth_mat[0,0])
    ky_limit = math.ceil(far_nbr_cutoff / orth_mat[1,1])
    kz_limit = math.ceil(far_nbr_cutoff / orth_mat[2,2])

  else:
    kx_limit = 0
    ky_limit = 0
    kz_limit = 0
  kx_list = list(range(-kx_limit, kx_limit+1))
  ky_list = list(range(-ky_limit, ky_limit+1))
  kz_list = list(range(-kz_limit, kz_limit+1))

  all_shift_comb = [[0,0,0]] # this should be the first one
  for x in kx_list:
    for y in ky_list:
      for z in kz_list:
        # this is already added
        if x == 0 and y == 0 and z == 0:
          continue
        all_shift_comb.append((x,y,z))
  return onp.array(all_shift_comb)

def cluster_systems_for_aligning(size_dicts, num_cuts=5,
                                 max_iterations=100,
                                 rep_count=20, print_mode=True):
  '''
  Cluster the similar structures together for aligning them while
  minimizing padding
  '''
  #run the modified k-means algorithm to form the clusters
  [labels,
   min_centr,
   min_counts,
   min_cost] = modified_kmeans(size_dicts,
                               k=num_cuts,
                               max_iterations=max_iterations,
                               rep_count=rep_count,
                               print_mode=print_mode)

  all_cut_indices = [[] for i in range(num_cuts)]
  for i,s in enumerate(size_dicts):
    label = labels[i]
    all_cut_indices[label].append(i)

  return all_cut_indices, min_cost, min_centr


def count_inter_list_sizes(systems, force_field,
                           num_threads=1, pool=None, chunksize=32):
  '''
  Calculate the interaction list sizes for given list of structures
  '''
  force_field = move_dataclass(force_field, onp)
  start = time.time()
  # get_context("fork") needed for Mac arm processors
  if pool == None:
    my_pool = get_context("fork").Pool(num_threads)
  else:
    my_pool = pool
  size_dicts = pool_handler_for_inter_list_count(systems, force_field,
                                                 my_pool, chunksize)
  end = time.time()
  if pool == None:
    my_pool.terminate()
  print("Multithreaded interaction list counting took {:.2f} secs with {} threads".format(end-start,num_threads))
  return size_dicts


def process_and_cluster_geos(systems,force_field,max_num_clusters=10,
                             num_threads=1,chunksize=1,all_cut_indices=None):
  '''
  Calculate the interaction list sizes for given list of structures first
  then cluster the similar structures together
  '''  
  size_dicts = count_inter_list_sizes(systems, force_field, num_threads=num_threads, chunksize=chunksize)

  if all_cut_indices == None:
    all_costs_old = []
    prev = -1
    selected_n_cut = 0
    for n_cut in range(1,max_num_clusters+1):
      all_cut_indices, cost_total, center_sizes = cluster_systems_for_aligning(size_dicts,num_cuts=n_cut,max_iterations=1000,rep_count=1000,print_mode=False)
      #print("Cost with {} clusters: {}".format(n_cut, cost_total))
      all_costs_old.append(cost_total)
      if prev != -1 and cost_total > prev or (prev-cost_total) / prev < 0.15:
          selected_n_cut = n_cut - 1
          break
      prev = cost_total
    #sys.exit()
    if selected_n_cut == 0:
      selected_n_cut = max_num_clusters
    all_cut_indices, cost_total, center_sizes = cluster_systems_for_aligning(size_dicts,num_cuts=selected_n_cut,max_iterations=1000,rep_count=1000,print_mode=True)

  globally_sorted_indices = []
  for l in all_cut_indices:
    for ind in l:
      globally_sorted_indices.append(ind)
  return globally_sorted_indices, all_cut_indices, center_sizes


def build_energy_report_item(energy_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report row for energy item
  '''
  sys_inds = energy_item.sys_inds
  multips = energy_item.multip
  out_str = ""
  for i in range(len(sys_inds)):
    if multips[i] == 0.0:
      continue
    name = geo_index_to_name[sys_inds[i]]
    div = round(1.0/multips[i])
    sign_str = "+"
    if div < 0:
      sign_str = "-"
    out_str += f"{sign_str} {name}/{abs(div)} "
  out_str = out_str[:-1]
  out_str = "ENERGY: " + out_str
  row = [out_str, energy_item.weight, energy_item.target, pred, weighted_error]
  return row

def build_force_report_item(force_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report row for force item
  '''
  name = geo_index_to_name[force_item.sys_ind]
  out_str = f"{name} {force_item.a_ind + 1}"
  dirs = ["X", "Y", "Z"]
  rows = []
  for i in range(3):
    new_out_str = f"FORCE-{dirs[i]}: " + out_str
    row = [new_out_str, force_item.weight, float(force_item.target[i]),
     float(pred[i]), float(weighted_error[i])]
    rows.append(row)
  return rows

def build_charge_report_item(charge_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report row for charge item
  '''
  name = geo_index_to_name[charge_item.sys_ind]
  out_str = f"{name} {charge_item.a_ind + 1}"
  out_str = "CHARGE: " + out_str
  row = [out_str, charge_item.weight, charge_item.target, pred, weighted_error]
  return row

def build_distance_report_item(distance_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report row for distance item
  '''
  name = geo_index_to_name[distance_item.sys_ind]
  out_str = f"{name} {distance_item.a1_ind + 1} {distance_item.a2_ind + 1}"
  out_str = "DISTANCE: " + out_str
  row = [out_str, distance_item.weight, distance_item.target, pred, weighted_error]
  return row

def build_angle_report_item(angle_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report row for angle item
  '''
  name = geo_index_to_name[angle_item.sys_ind]
  out_str = f"{name} {angle_item.a1_ind + 1} {angle_item.a2_ind + 1} {angle_item.a3_ind + 1}"
  out_str = "ANGLE: " + out_str
  row = [out_str, angle_item.weight, angle_item.target, pred, weighted_error]
  return row

def build_torsion_report_item(torsion_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report for torsion items
  '''
  name = geo_index_to_name[torsion_item.sys_ind]
  out_str = f"{name} {torsion_item.a1_ind + 1} {torsion_item.a2_ind + 1} {torsion_item.a3_ind + 1} {torsion_item.a4_ind + 1}"
  out_str = "TORSION: " + out_str
  row = [out_str, torsion_item.weight, torsion_item.target, pred, weighted_error]
  return row

# Produces a report with item based error (similar to what the standalone code does)
def produce_error_report(filename, tranining_items, indiv_error, geo_index_to_name):
  '''
  Produce an error report, similar to how the standalone code does it
  '''
  fptr = open(filename, 'w')
  headers = ["Item Text", "Weight", "Target", "Prediction", "Weighted Error", "Cum. Sum."]
  data_to_print = []
  cumulative_err = 0.0
  functions = {"ENERGY":build_energy_report_item,
               "CHARGE":build_charge_report_item,
               "FORCE":build_force_report_item,
               "DISTANCE":build_distance_report_item,
               "ANGLE":build_angle_report_item,
               "TORSION":build_torsion_report_item}

  attributes = {"ENERGY":"energy_items",
               "CHARGE":"charge_items",
               "FORCE":"force_items",
               "DISTANCE":"dist_items",
               "ANGLE":"angle_items",
               "TORSION":"torsion_items"}

  for key, attr in attributes.items():
    if key not in indiv_error:
      continue
    sub_items = getattr(tranining_items, attr)
    sub_items = move_dataclass(sub_items, onp)
    sub_items = split_dataclass(sub_items)
    [preds, targets, weighted_errors] = indiv_error[key]
    for i, item in enumerate(sub_items): 
      row = functions[key](item, preds[i], weighted_errors[i], geo_index_to_name)
      if key == "FORCE":
        rows = row
        for j, row in enumerate(rows):
          cumulative_err += float(weighted_errors[i][j])
          row.append(cumulative_err)
          data_to_print.append(row)
      else:
        cumulative_err += weighted_errors[i]
        row.append(cumulative_err)
        data_to_print.append(row)


  table = tabulate(data_to_print, headers, floatfmt=".2f")
  print(table, file=fptr)
  fptr.close()


def preprocess_trainset_line(line):
  '''
  Proeprocess a given line from training set
  '''
  line = line.replace('/', ' / ')
  return line

def read_geo_file(geo_file, name_to_index_map, far_nbr_cutoff):
  '''
  Read the geometries from the provided geometry file
  '''
  if not os.path.exists(geo_file):
    print("Path {} does not exist!".format(geo_file))
    return []
  list_systems = []
  f = open(geo_file,'r')
  system_name = ''
  atoms_positions = []
  # add dummy restraints
  bond_restraints = [[-1,-1,0,0,0]]
  angle_restraints = [[-1,-1,-1,0,0,0]]
  torsion_restraints = [[-1,-1,-1,-1,0,0,0]]
  molcharge_items = []
  atom_names = []
  system_str = ''
  box = [999.0, 999.0, 999.0]
  box_angles = [90.0,90.0,90.0]
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
      num_atoms = len(atom_names)
      # currently only total charge for all of the atom is supported, no partial charges
      if (len(molcharge_items) > 1 
          or (len(molcharge_items) == 1 
              and (molcharge_items[0][1] - molcharge_items[0][0] + 1) < num_atoms)):
        print("[ERROR] error in {}, MOLCHARGE is only supported for the total system charge!".format(system_name))
        sys.exit()

      total_charge = 0
      if len(molcharge_items) == 1:
        total_charge = molcharge_items[0][2]
      atom_names = onp.array(atom_names)
      atomic_nums = onp.zeros(num_atoms, dtype=onp.int32)
      # The rest of the atomic numbers are not important
      atomic_nums = onp.where(atom_names=="C", 6, atomic_nums)
      atomic_nums = onp.where(atom_names=="O", 8, atomic_nums)
      atomic_nums = onp.where(atom_names=="H", 1, atomic_nums)
      reax_atom_types = [name_to_index_map[name] for name in atom_names]
      reax_atom_types = onp.array(reax_atom_types)
      atoms_positions = onp.array(atoms_positions)
      # box information
      orth_mat = orthogonalization_matrix(box, box_angles)
      all_shifts = calculate_box_shifts(is_periodic, far_nbr_cutoff, orth_mat)
      # restraints
      bond_restraints = onp.array(bond_restraints)
      angle_restraints = onp.array(angle_restraints)
      torsion_restraints = onp.array(torsion_restraints)

      new_bond_restraints = BondRestraint(ind1 = bond_restraints[:,0].astype(onp.int32),
                                          ind2 = bond_restraints[:,1].astype(onp.int32),
                                          force1 = bond_restraints[:,2].astype(onp.float32),
                                          force2 = bond_restraints[:,3].astype(onp.float32),
                                          target = bond_restraints[:,4].astype(onp.float32))

      new_angle_restraints = AngleRestraint(ind1 = angle_restraints[:,0].astype(onp.int32),
                                          ind2 = angle_restraints[:,1].astype(onp.int32),
                                          ind3 = angle_restraints[:,2].astype(onp.int32),
                                          force1 = angle_restraints[:,3].astype(onp.float32),
                                          force2 = angle_restraints[:,4].astype(onp.float32),
                                          target = angle_restraints[:,5].astype(onp.float32))

      new_torsion_restraints = TorsionRestraint(ind1 = torsion_restraints[:,0].astype(onp.int32),
                                          ind2 = torsion_restraints[:,1].astype(onp.int32),
                                          ind3 = torsion_restraints[:,2].astype(onp.int32),
                                          ind4 = torsion_restraints[:,3].astype(onp.int32),
                                          force1 = torsion_restraints[:,4].astype(onp.float32),
                                          force2 = torsion_restraints[:,5].astype(onp.float32),
                                          target = torsion_restraints[:,6].astype(onp.float32))
      # create the structure from the read data
      new_system = Structure(system_name, num_atoms,
                             reax_atom_types, atomic_nums,
                             atoms_positions, orth_mat, total_charge,
                             do_minimization, max_it, all_shifts,
                             new_bond_restraints, new_angle_restraints,
                             new_torsion_restraints)

      list_systems.append(new_system)
      atoms_positions = []
      atom_names = []
      # add dummy restraints
      bond_restraints = [[-1,-1,0,0,0]]
      angle_restraints = [[-1,-1,-1,0,0,0]]
      torsion_restraints = [[-1,-1,-1,-1,0,0,0]]
      molcharge_items = []
      system_str = ''
      box = [999.0, 999.0, 999.0]
      box_angles = [90.0,90.0,90.0]
      is_periodic = False
      do_minimization = True
      max_it = 99999
    else:
      if line.startswith('DESCRP'):
        system_name = line.strip().split()[1]
      # box info
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
      # whether we need energy minim.
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
      # charge info
      elif line.startswith('MOLCHARGE'):
        #Ex. MOLCHARGE   1  30  1.00
        split_line = line.split()[1:]
        at1, at2 = split_line[:2]
        total_charge = float(split_line[2])
        molcharge_items.append([int(at1)-1,int(at2)-1,total_charge])
      # restraint info
      elif line.startswith('BOND RESTRAINT'):
        split_line = line.split()[2:]
        at1,at2 = split_line[:2]
        dist = split_line[2]
        force1,force2 = split_line[3:5]
        d_dist = split_line[5]
        bond_restraints.append([int(at1)-1,int(at2)-1,float(force1),float(force2),float(dist)])
      elif line.startswith('ANGLE RESTRAINT'):
        split_line = line.split()[2:]
        at1,at2,at3 = split_line[:3]
        angle = split_line[3]
        force1,force2 = split_line[4:6]
        d_angle = split_line[6]
        angle_restraints.append([int(at1)-1,int(at2)-1,float(at3)-1,float(force1),float(force2),float(angle)])
      elif line.startswith('TORSION RESTRAINT'):
        split_line = line.split()[2:]
        at1,at2,at3,at4 = split_line[:4]
        torsion = split_line[4]
        force1,force2 = split_line[5:7]
        d_torsion = split_line[7]
        torsion_restraints.append([int(at1)-1,int(at2)-1,int(at3)-1,int(at4)-1,float(force1),float(force2),float(torsion)])
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

def create_structure_map(structures):
  '''
  Create name -> index and index->name maps
  '''
  name_to_index = {}
  index_to_name = {}
  for i in range(len(structures)):
    s = structures[i]
    name_to_index[s.name] = i
    index_to_name[i] = s.name
  return name_to_index, index_to_name


def read_parameter_file(params_file, ignore_sensitivity=1):
  '''
  Read the parameter file
  '''
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
  '''
  Map the read parameters to new type of indexing to select them from
  a given force field object
  '''
  new_params = []
  for p in params:
      key = (p[0],p[1],p[2])
      value = index_map[key]
      new_item = (value, p[3],p[4],p[5])
      new_params.append(new_item)
  return new_params


def read_train_set(train_in):
  '''
  Read the training set data
  '''
  f = open(train_in, 'r')
  training_items = {}
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
  energy_items = []
  charge_items = []

  for line in f:
    #print(line)
    line = line.strip()
    # ignore everything after #
    line = line.split('#', 1)[0]
    line = line.split('!', 1)[0]
    if len(line) == 0 or line.startswith("#"):
      continue
    # flags to use to detect corresponding regions
    elif line.startswith("ENERGY"):
      energy_flag = 1

    elif line.startswith("CHARGE"):
      charge_flag = 1

    elif line.startswith("GEOMETRY"):
      geo_flag = 1

    elif line.startswith('FORCES'):
      force_flag = 1

    elif line.startswith("ENDENERGY"):
      energy_flag = 0

    elif line.startswith("ENDCHARGE"):
      charge_flag = 0

    elif line.startswith("ENDGEOMETRY"):
      geo_flag = 0

    elif line.startswith("ENDFORCES"):
      force_flag = 0
    # energy items
    elif energy_flag == 1:
      line = preprocess_trainset_line(line)
      split_line = line.split()
      # w and energy + 4 items per ref. item
      num_ref_items = int((len(split_line) - 2) / 4) 

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
      energy_item = EnergyItem(name_list, multiplier_list, energy, w)

      energy_items.append(energy_item)
    # charge item
    elif charge_flag == 1:
      line = preprocess_trainset_line(line)
      split_line = line.split()
      name = split_line[0].strip()
      weight = float(split_line[1])
      index = int(split_line[2]) - 1
      charge = float(split_line[3])
      charge_item = ChargeItem(name, index, charge, weight)
      charge_items.append(charge_item)
    # geo item
    elif geo_flag == 1:
      line = preprocess_trainset_line(line)
      split_line = line.split()
      name = split_line[0].strip()
      weight = float(split_line[1])
      target = float(split_line[-1])
      # 2-body
      if len(split_line) == 5:
        index1 = int(split_line[2]) - 1
        index2 = int(split_line[3]) - 1
        dist_item = DistItem(name, index1, index2, target, weight)
        geo2_items.append(dist_item)

      # 3-body
      if len(split_line) == 6:
        index1 = int(split_line[2]) - 1
        index2 = int(split_line[3]) - 1
        index3 = int(split_line[4]) - 1
        angle_item = AngleItem(name, index1, index2, index3, target, weight)
        geo3_items.append(angle_item)
      # 4-body
      if len(split_line) == 7:
        index1 = int(split_line[2]) - 1
        index2 = int(split_line[3]) - 1
        index3 = int(split_line[4]) - 1
        index4 = int(split_line[5]) - 1
        torsion_item = TorsionItem(name, index1, index2, index3, index4, target, weight)
        geo4_items.append(torsion_item)
      #RMSG
      if len(split_line) == 3:
        rmsg_item = RMSGItem(name, target, weight)
        force_RMSG_items.append(rmsg_item)
    # force item
    elif force_flag == 1:
      line = preprocess_trainset_line(line)
      split_line = line.split()
      name = split_line[0].strip()
      weight = float(split_line[1])
      #force on indiv. atoms
      if len(split_line) == 6:
        index = int(split_line[2]) - 1
        f1 = float(split_line[3])
        f2 = float(split_line[4])
        f3 = float(split_line[5])
        force_item = ForceItem(name, index, [f1,f2,f3], weight)
        force_atom_items.append(force_item)

  if len(energy_items) > 0:
    training_items['energy_items'] = energy_items

  if len(charge_items) > 0:
    training_items["charge_items"] = charge_items

  if len(geo2_items) > 0:
    training_items["dist_items"] = geo2_items

  if len(geo3_items) > 0:
    training_items["angle_items"] = geo3_items

  if len(geo4_items) > 0:
    training_items["torsion_items"] = geo4_items

  if len(force_RMSG_items) > 0:
    training_items["RMSG_items"] = force_RMSG_items

  if len(force_atom_items) > 0:
    training_items["force_items"] = force_atom_items

  return training_items

def filter_data(systems, training_items):
  '''
  Filter out the unused items
  '''
  system_names = {s.name for s in systems}
  new_systems = []
  new_training_items = {}
  used_geo_names = set()
  for key in training_items.keys():
    new_training_items[key] = []
    for item in training_items[key]:
      if key == 'energy_items':
        names = item.sys_inds
      else:
        names = [item.sys_ind,]
      skip = False
      for name in names:
        if name not in system_names:
          skip = True
          break
      if skip == False:
        new_training_items[key].append(item)
        for name in names:
          used_geo_names.add(name)
  new_systems = [s for s in systems if s.name in used_geo_names]
  return new_systems, new_training_items


def structure_training_data(training_items, geo_name_to_index):
  '''
  Restructure the training data items to be used for training
  '''
  # replace names with indices
  for key in training_items.keys():
      for i, item in enumerate(training_items[key]):
          if key == 'energy_items':
              sys_inds = [geo_name_to_index[name] for name in item.sys_inds]
              item = dataclasses.replace(item, sys_inds = sys_inds)
          else:
              sys_ind = geo_name_to_index[item.sys_ind]
              item = dataclasses.replace(item, sys_ind = sys_ind)
          training_items[key][i] = item
  # Align the sizes for the energy items
  if 'energy_items' in training_items:
      energy_items = training_items['energy_items']
      max_sys_len = max([len(item.sys_inds) for item in energy_items])
      for i, item in enumerate(energy_items):
          sys_inds = item.sys_inds
          filler = [-1] * (max_sys_len - len(sys_inds))
          sys_inds.extend(filler)

          multip = item.multip
          filler = [0.0] * (max_sys_len - len(multip))
          multip.extend(filler)

          energy_items[i] = dataclasses.replace(energy_items[i],
                                                multip=multip,
                                                sys_inds=sys_inds)
      training_items['energy_items'] = energy_items

  new_items = {}
  for key in training_items.keys():
      items = training_items[key]
      if len(items) == 0:
          continue
      field_names = [field.name for field in dataclasses.fields(items[0])]
      collected_attr = {}
      for attr in field_names:
          attr_list = []
          for item in items:
              val = getattr(item, attr)
              attr_list.append(val)
          collected_attr[attr] = onp.array(attr_list)
      collected_obj = items[0].__class__(**collected_attr)
      new_items[key] = collected_obj
  return TrainingData(**new_items)

def parse_and_save_force_field(old_ff_file, new_ff_file,force_field):
  '''
  Save the force field to a file
  '''
  output = ""
  f = open(old_ff_file, 'r')
  line = f.readline()
  output = output + line
  header = line.strip()

  line = f.readline()
  output = output + line
  num_params = int(line.strip().split()[0])
  global_params = jnp.zeros(shape=(num_params,1), dtype=jnp.float64)
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
    line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.alf[i]) #alf
    line[3 + 9 * 1:3 + 9 * 2] = "{:9.4f}".format(ff.vop[i]) #vop
    line[3 + 9 * 2:3 + 9 * 3] = "{:9.4f}".format(ff.valf[i]) #valf
    line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.valp1[i]) #valp1
    line[3 + 9 * 5:3 + 9 * 6] = "{:9.4f}".format(ff.electronegativity[i])
    line[3 + 9 * 6:3 + 9 * 7] = "{:9.4f}".format(ff.idempotential[i])

    output = output + ''.join(line)
    # third line
    line = f.readline()
    line = list(line)
    line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.vnq[i]) #vnq - rob3
    line[3 + 9 * 1:3 + 9 * 2] = "{:9.4f}".format(ff.vlp1[i]) #vlp1
    line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.bo131[i]) #bo131
    line[3 + 9 * 4:3 + 9 * 5] = "{:9.4f}".format(ff.bo132[i]) #bo132
    line[3 + 9 * 5:3 + 9 * 6] = "{:9.4f}".format(ff.bo133[i]) #bo133

    output = output + ''.join(line)

    # fourth line
    line = f.readline()
    line = list(line)
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

    if i1 != -1 and i4 != -1:
      line[12 + 9 * 0:12 + 9 * 1] = "{:9.4f}".format(ff.v1[i1,i2,i3,i4])
      line[12 + 9 * 1:12 + 9 * 2] = "{:9.4f}".format(ff.v2[i1,i2,i3,i4])
      line[12 + 9 * 2:12 + 9 * 3] = "{:9.4f}".format(ff.v3[i1,i2,i3,i4])
      line[12 + 9 * 3:12 + 9 * 4] = "{:9.4f}".format(ff.v4[i1,i2,i3,i4])
      line[12 + 9 * 4:12 + 9 * 5] = "{:9.4f}".format(ff.vconj[i1,i2,i3,i4])

    if i1 == -1 and i4 == -1:
      sel_ind = force_field.num_atom_types - 1
      line[12 + 9 * 0:12 + 9 * 1] = "{:9.4f}".format(ff.v1[sel_ind,
                                                           i2,
                                                           i3,
                                                           sel_ind])
      line[12 + 9 * 1:12 + 9 * 2] = "{:9.4f}".format(ff.v2[sel_ind,
                                                           i2,
                                                           i3,
                                                           sel_ind])
      line[12 + 9 * 2:12 + 9 * 3] = "{:9.4f}".format(ff.v3[sel_ind,
                                                           i2,
                                                           i3,
                                                           sel_ind])
      line[12 + 9 * 3:12 + 9 * 4] = "{:9.4f}".format(ff.v4[sel_ind,
                                                           i2,
                                                           i3,
                                                           sel_ind])
      line[12 + 9 * 4:12 + 9 * 5] = "{:9.4f}".format(ff.vconj[sel_ind,
                                                              i2,
                                                              i3,
                                                              sel_ind])
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



