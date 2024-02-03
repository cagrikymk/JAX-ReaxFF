import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
import jax
jax.config.update("jax_enable_x64", True)
import numpy as onp
import jax.numpy as jnp
from absl.testing import parameterized
from jax_md import dataclasses
from jax_md.reaxff.reaxff_forcefield import ForceField
from jax_md.reaxff.reaxff_helper import read_force_field
from jaxreaxff.helper import (move_dataclass,
                              process_and_cluster_geos,
                              create_structure_map,
                              read_geo_file,
                              count_inter_list_sizes)
from jaxreaxff.structure import align_structures
from jaxreaxff.interactions import reaxff_interaction_list_generator
import pickle
from frozendict import frozendict

from jax_md.reaxff.reaxff_energy import calculate_reaxff_energy
from jaxreaxff.interactions import calculate_dist_and_angles

ATOL = 1e-3
RTOL = 1e-4

def calculate_energy(positions, structure, nbr_lists, force_field):
  # handle off diag. and symm. in the force field
  force_field = ForceField.fill_off_diag(force_field)
  force_field = ForceField.fill_symm(force_field)

  dists_and_angles = calculate_dist_and_angles(positions,
                                               structure,
                                               nbr_lists)
  # set max_solver_iter to -1 to use direct solve (LU based)
  energy, charges =  calculate_reaxff_energy(structure.atom_types,
                              structure.atomic_nums,
                              nbr_lists,
                              *dists_and_angles,
                              force_field,
                              total_charge = structure.total_charge,
                              tol= 1e-06,
                              backprop_solve = False,
                              tors_2013 = False,
                              solver_model = "EEM",
                              max_solver_iter=-1)
  return energy

def assert_numpy_allclose(a, b, atol=None, rtol=None, err_msg=''):
  kw = {}
  if atol: kw['atol'] = atol
  if rtol: kw['rtol'] = rtol
  with onp.errstate(invalid='ignore'):
    onp.testing.assert_allclose(a, b, **kw, err_msg=err_msg)

TEST_DATA = []
def read_test_data(test_folder):
  for root, sub_dirs, _ in os.walk(test_folder):
    for dir in sub_dirs:
      test_name = dir
      ffield_path = f"{root}/{dir}/ffield"
      geo_path = f"{root}/{dir}/geo"
      results_path = f"{root}/{dir}/target_data.pickle"
      with open(results_path, 'rb') as f:
        results = pickle.load(f)
      item = {"name":test_name, "ffield_path":ffield_path,
              "geo_path":geo_path,
              "results":results}
      TEST_DATA.append(item)
# The file paths for force fields and geometries will be provided here
# as well as the target data to match.

read_test_data("tests/data")

def read_and_process_FF_file(filename, cutoff2 = 0.001, dtype=jnp.float64):
  force_field = read_force_field(filename,cutoff2 = cutoff2, dtype=dtype)
  force_field = ForceField.fill_off_diag(force_field)
  force_field = ForceField.fill_symm(force_field)

  return force_field

def read_and_process_geo_file(filename, force_field):
  systems = read_geo_file(filename, force_field.name_to_index, 10.0)
  geo_name_to_index, geo_index_to_name = create_structure_map(systems)
  # replace names with indices
  for i,s in enumerate(systems):
      s = dataclasses.replace(s, name = geo_name_to_index[s.name])
      systems[i] = s
  return systems, geo_index_to_name

class ReaxFFEnergyTest(parameterized.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.values = [0 for _ in TEST_DATA]

    cls.ffields = [read_and_process_FF_file(test["ffield_path"])
                    for test in TEST_DATA]
    geo_res = [read_and_process_geo_file(test["geo_path"], cls.ffields[i]) for i, test in enumerate(TEST_DATA)]
    cls.geos = [r[0] for r in geo_res]
    cls.geo_index_to_name = [r[1] for r in geo_res]

  @parameterized.parameters(
       [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
  def test_inter_count(self, i, name):
    my_ff = self.ffields[i]
    my_geo = self.geos[i]
    my_index_to_name = self.geo_index_to_name[i]
    data = TEST_DATA[i]['results']
    num_threads = os.cpu_count()
    size_dicts = count_inter_list_sizes(my_geo, my_ff, num_threads=num_threads, chunksize=4)
    for i in range(len(size_dicts)):
      geo_name = my_index_to_name[i]
      hbond_count = size_dicts[i]["hbond_size"]
      body3_count = size_dicts[i]["filter3_size"]
      body4_count = size_dicts[i]["filter4_size"]
      self.assertEqual(hbond_count, data[geo_name]['hbond count'], geo_name + ' hbond count')
      self.assertEqual(body3_count, data[geo_name]['3-body count'], geo_name + ' 3-body count')
      self.assertEqual(body4_count, data[geo_name]['4-body count'], geo_name + ' 4-body count')

  @parameterized.parameters(
       [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
  def test_energy_and_forces(self, i, name):
    my_ff = self.ffields[i]
    my_geo = self.geos[i]
    num_structures = len(my_geo)
    my_index_to_name = self.geo_index_to_name[i]
    data = TEST_DATA[i]['results']
    num_threads = os.cpu_count()
    globally_sorted_indices, all_cut_indices, center_sizes = process_and_cluster_geos(my_geo,my_ff, max_num_clusters=5, num_threads=num_threads)
    for i in range(len(center_sizes)):
      for k in center_sizes[i].keys():
        if center_sizes[i][k] == 0:
          center_sizes[i][k] = 1

    aligned_data = []
    for i in range(len(center_sizes)):
      zz = align_structures([my_geo[i] for i in all_cut_indices[i]], center_sizes[i], jnp.float64)
      zz = move_dataclass(zz, jnp)
      aligned_data.append(zz)
    my_ff = move_dataclass(my_ff, jnp)


    batched_allocate = reaxff_interaction_list_generator(my_ff,
                                          close_cutoff = 5.0,
                                          far_cutoff = 10.0,
                                          use_hbond=True)

    allocate_f = jax.jit(batched_allocate,static_argnums=(3,))
    center_sizes = [frozendict(c) for c in center_sizes]

    all_inters = [allocate_f(aligned_data[i].positions, aligned_data[i], my_ff, center_sizes[i])[0] for i in range(len(center_sizes))]
    my_calculate_energy = jax.vmap(jax.value_and_grad(calculate_energy), (0,0,0, None))
    all_energies = jnp.zeros(num_structures)
    all_forces = [None] * num_structures
    for i in range(len(center_sizes)):
      energy, forces = my_calculate_energy(aligned_data[i].positions, aligned_data[i], all_inters[i], my_ff)
      all_energies = all_energies.at[aligned_data[i].name].set(energy)
      for j in range(len(aligned_data[i].name)):
        ind = int(aligned_data[i].name[j])
        num_atoms = aligned_data[i].atom_count[j]
        all_forces[ind] = onp.array(forces[j][:num_atoms])
    all_energies = onp.array(all_energies)

    for i in range(num_structures):
      geo_name = my_index_to_name[i]
      assert_numpy_allclose(all_energies[i], data[geo_name]['energy'],
                             err_msg=f"{geo_name} - energy", atol=ATOL, rtol=RTOL)
      assert_numpy_allclose(all_forces[i], data[geo_name]['forces'],
                            err_msg=f"{geo_name} - force", atol=0.1, rtol=0.01)

