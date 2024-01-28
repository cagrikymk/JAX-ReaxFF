'''
Contains the dataclass definitions for Structure
It is used to store the information regarding geometries
'''
from jax_md import dataclasses, util
import numpy as onp
Array = util.Array

@dataclasses.dataclass
class BondRestraint(object):
    ind1: Array
    ind2: Array
    target: Array
    force1: Array
    force2: Array

@dataclasses.dataclass
class AngleRestraint(object):
    ind1: Array
    ind2: Array
    ind3: Array
    target: Array
    force1: Array
    force2: Array

@dataclasses.dataclass
class TorsionRestraint(object):
    ind1: Array
    ind2: Array
    ind3: Array
    ind4: Array
    target: Array
    force1: Array
    force2: Array

@dataclasses.dataclass
class Structure(object):
    name: Array
    atom_count: Array
    atom_types: Array
    atomic_nums: Array
    positions: Array
    orth_matrix: Array

    total_charge: Array
    energy_minimize: Array
    energy_minim_steps: Array
    periodic_image_shifts: Array

    bond_restraints: BondRestraint
    angle_restraints: AngleRestraint
    torsion_restraints: TorsionRestraint


def align_restraints(structures):
    full_size = len(structures)
    field_names = ['bond_restraints', 'angle_restraints', 'torsion_restraints']
    classes = [BondRestraint, AngleRestraint, TorsionRestraint]
    final_structures = []
    for i,field in enumerate(field_names):
        my_class = classes[i]
        max_size = 1
        for s in structures:
            val = getattr(s, field)
            max_size = max(max_size, len(val.target))

        attr_dict = {}
        class_fields = [field.name for field in dataclasses.fields(my_class)]
        for c_field in class_fields:
            sub_val = getattr(val, c_field)
            attr_dict[c_field] = onp.zeros((full_size,max_size), dtype=sub_val.dtype) - 1
            for j,s in enumerate(structures):
                val = getattr(s, field)
                sub_val = getattr(val, c_field)
                attr_dict[c_field][j, :len(sub_val)] = sub_val
        final_structures.append(my_class(**attr_dict))
    return final_structures

def align_structures(structures, max_sizes, dtype=onp.float32):
    full_size = len(structures)
    num_atoms = max_sizes['num_atoms']
    image_count = max_sizes['periodic_image_count']

    name = onp.zeros(shape=(full_size,), dtype=onp.int32)
    atom_count = onp.zeros(shape=(full_size,), dtype=onp.int32)
    atom_types = onp.zeros(shape=(full_size, num_atoms), dtype=onp.int32) -1 # -1 is for masking
    atomic_nums = onp.zeros(shape=(full_size, num_atoms), dtype=onp.int32)
    positions = onp.zeros(shape=(full_size, num_atoms, 3), dtype=dtype)
    orth_matrix = onp.zeros(shape=(full_size, 3, 3), dtype=dtype)

    total_charge = onp.zeros(shape=(full_size,), dtype=dtype)
    energy_minimize = onp.zeros(shape=(full_size,), dtype=onp.bool_)
    energy_minim_steps = onp.zeros(shape=(full_size,), dtype=onp.int32)

    periodic_image_shifts = onp.zeros(shape=(full_size, image_count, 3), dtype=onp.int32)
    # add 999 to ignore the padded items during distance calc.
    # all distances will be >> cutoffs when padded
    periodic_image_shifts = periodic_image_shifts + 999

    for i in range(full_size):
      s = structures[i]
      name[i] = s.name
      atom_count[i] = s.atom_count
      atom_types[i,:s.atom_count] = s.atom_types
      atomic_nums[i,:s.atom_count] = s.atomic_nums
      positions[i,:s.atom_count,:] = s.positions
      orth_matrix[i] = s.orth_matrix
      total_charge[i] = s.total_charge
      energy_minimize[i] = s.energy_minimize
      energy_minim_steps[i] = s.energy_minim_steps
      periodic_image_shifts[i, :len(s.periodic_image_shifts)] = s.periodic_image_shifts

    bond_rest, angle_res, tors_rest = align_restraints(structures)
    new_system = Structure(name=name,
                           atom_count=atom_count,
                           atom_types=atom_types,
                           atomic_nums=atomic_nums,
                           positions=positions,
                           orth_matrix=orth_matrix,
                           total_charge=total_charge,
                           energy_minimize=energy_minimize,
                           energy_minim_steps=energy_minim_steps,
                           periodic_image_shifts=periodic_image_shifts,
                           bond_restraints=bond_rest,
                           angle_restraints=angle_res,
                           torsion_restraints=tors_rest)
    return new_system
