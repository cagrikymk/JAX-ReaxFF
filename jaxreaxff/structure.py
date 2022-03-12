#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains structure related logic:
Structure class and interaction list generation

Author: Mehmet Cagri Kaymak
"""
import numpy as onp
import jax.numpy as np
import jax
from jaxreaxff.forcefield import TYPE
import math

CLOSE_NEIGH_CUTOFF = 5.0 #A
BUFFER_DIST = 0.5
FAR_NEIGH_CUTOFF = 10.0#A

BODY_4_BOND_CUTOFF = 3.5
BODY_3_BOND_CUTOFF = 4.0
HBOND_CUTOFF = 7.5

BOND_RESTRAINTS_MAX_SIZE = 10
ANGLE_RESTRAINTS_MAX_SIZE = 10
TORSION_RESTRAINTS_MAX_SIZE = 10


# it fixes nan values issue, from: https://github.com/google/jax/issues/1052
def vectorized_cond(pred, true_fun, false_fun, operand):
    # true_fun and false_fun must act elementwise (i.e. be vectorized)
    #how to use: grad(lambda x: vectorized_cond(x > 0.5, lambda x: np.arctan2(x, x), lambda x: 0., x))(0.)
    true_op = np.where(pred, operand, 0)
    false_op = np.where(pred, 0, operand)
    return np.where(pred, true_fun(true_op), false_fun(false_op))


#from: https://github.com/google/jax/issues/1052
@jax.custom_jvp
def safe_sqrt(x):
    return np.sqrt(x)

# f_jvp :: (a, T a) -> (b, T b)
def safe_sqrt_jvp(primals, tangents):
    x, = primals
    t, = tangents
    return safe_sqrt(x), np.where(x==0,0.0, 0.5 * np.power(x, -0.5)) * t

safe_sqrt.defjvp(safe_sqrt_jvp)

#source:  reaxff frotran, subroutine distan
def orthogonalization_matrix(box_lengths, angles_degr):
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
            dtype=TYPE)

    if angles_degr[0] == 90.0 and angles_degr[1] == 90.0 and angles_degr[2] == 90.0:
        mat = onp.eye(3, dtype=TYPE)
        mat[0,0] = a
        mat[1,1] = b
        mat[2,2] = c
    #box_norms = onp.sqrt(onp.sum(onp.square(mat),axis=1))


    '''
    mat[:,0] = mat[:,0] / box_norms[0]
    mat[:,1] = mat[:,1] / box_norms[1]
    mat[:,2] = mat[:,2] / box_norms[2]
    '''
    return mat

def project_onto_orth_box(box_size,box_angles,positions):
    matrix = orthogonalization_matrix(box_size,box_angles)
    new_box_size = box_size.dot(matrix)
    new_positions = positions.dot(matrix)

    return new_box_size,new_positions


class Structure:
    def __init__(self,name, real_atom_count,positions,atom_types, atom_names,total_charge, is_periodic, do_minimization,num_min_steps, sim_box,box_angles, bond_restraints, angle_restraints, torsion_restraints):
        self.name = name
        self.atom_names = atom_names
        self.num_atoms = len(self.atom_names)
        self.real_atom_count = real_atom_count
        self.total_charge = total_charge
        self.atom_positions = positions # the size is set in read_geo function
        self.box_size = sim_box
        self.box_angles = box_angles
        if len(atom_types) == 0:
            self.atom_types = onp.zeros(shape=(self.num_atoms), dtype=onp.int32)
        else:
            self.atom_types = atom_types
        # mark the real and filler atoms
        self.atom_mask = onp.zeros(shape=(self.num_atoms), dtype=onp.bool)
        self.atom_mask[:real_atom_count] = 1
        self.do_minimization = do_minimization
        self.num_min_steps = num_min_steps
        #PERIODIC BOX PARAMS
        self.is_periodic = is_periodic
        # assign a large box
        #TODO: can a nonperiodic box have non-orth. axis?
        if self.is_periodic == False:
            self.box_size = onp.array([500.0,500.0,500.0])
            self.box_angles = onp.array([90.0,90.0,90.0])

        self.orth_matrix = orthogonalization_matrix(self.box_size,self.box_angles)

        if self.is_periodic == True:

            self.kx_limit = math.ceil(FAR_NEIGH_CUTOFF / self.orth_matrix[0,0])
            self.ky_limit = math.ceil(FAR_NEIGH_CUTOFF / self.orth_matrix[1,1])
            self.kz_limit = math.ceil(FAR_NEIGH_CUTOFF / self.orth_matrix[2,2])
            #TODO remove it
            #kx_limit = ky_limit = kz_limit = 0

        else:
            self.kx_limit = 0
            self.ky_limit = 0
            self.kz_limit = 0
        #TODO: currently not being used but later support this
        self.kx_list = list(range(-self.kx_limit, self.kx_limit+1))
        self.ky_list = list(range(-self.ky_limit, self.ky_limit+1))
        self.kz_list = list(range(-self.kz_limit, self.kz_limit+1))

        self.all_shift_comb = [[0,0,0]] # this should be the first one
        for x in self.kx_list:
            for y in self.ky_list:
                for z in self.kz_list:
                    # this is already added
                    if x == 0 and y == 0 and z == 0:
                        continue
                    self.all_shift_comb.append((x,y,z))
        self.all_shift_comb = onp.array(self.all_shift_comb)
        # 125 = 5^3
        #self.distance_matrices = self.create_distance_matrices(self.is_periodic,self.atom_positions,self.box_size)

        #NEW INTERACTION LIST PARAMETERS
        #local_body_2_neigh_list holds the neigh. index and bond index in global_body_2_inter_list and shifting factor

        #binary map which shows the direction of shifting
        # for a given row, if [0,1] -> for the first bond, the first atom is shifted.
        #                                in the second bond, the second atom is shifted

        #binary map which shows the direction of shifting
        # for a given row, if [0,1,1] -> for the first bond, the first atom is shifted.
        #                                in the second bond, the second atom is shifted
        #                                in the third bond, the second atom is shifted
        self.global_body_2_count = 0
        self.global_body_3_count = 0
        self.global_body_4_count = 0


        # Restraints
        # The last value is for masking (1 or 0)
        # at1,at2,force1,force2,dist,d_dist, 1
        self.bond_restraints = onp.zeros(shape=(BOND_RESTRAINTS_MAX_SIZE, 7))
        #at1,at2,at3,force1,force2,angle,d_angle,1
        self.angle_restraints = onp.zeros(shape=(BOND_RESTRAINTS_MAX_SIZE, 8))
        # at1,at2,at3,at4,force1,force2,torsion,d_torsion,1
        self.torsion_restraints = onp.zeros(shape=(BOND_RESTRAINTS_MAX_SIZE, 9))

        if (len(bond_restraints) > 0):
            self.bond_restraints[:len(bond_restraints),:] = bond_restraints
        if (len(angle_restraints) > 0):
            self.angle_restraints[:len(angle_restraints),:] = angle_restraints
        if (len(torsion_restraints) > 0):
            self.torsion_restraints[:len(torsion_restraints),:] = torsion_restraints

        # for the filler ones, assign the last atom as atom index
        self.bond_restraints[len(bond_restraints):,:2] = -1
        self.angle_restraints[len(angle_restraints):,:3] = -1
        self.torsion_restraints[len(torsion_restraints):,:4] = -1

        self.flattened_system = []
        self.bgf_file = ''


        #calculate some indicator for similarity (currently volume is good)

        self.volume = onp.prod(self.box_size)


    def create_bgf_file(self,run_type):
        #TODO: generate bgf file for a given run type
        pass

    def create_distance_matrices_single(shift_pos,tiled_atom_pos1, orth_matrix):

        tiled_atom_pos1_trans = tiled_atom_pos1.swapaxes(0,1)
        shifted_tiled_atom_pos1_trans = tiled_atom_pos1_trans + shift_pos
        diff = tiled_atom_pos1 - shifted_tiled_atom_pos1_trans
        distance_matrix = safe_sqrt(np.square(diff).sum(axis=2))
        return distance_matrix


    def create_distance_matrices(atom_positions, orth_matrix, all_shift_comb):
        num_atoms = len(atom_positions)
        atom_pos = atom_positions.reshape((num_atoms,1,3))
        shift_pos = np.dot(orth_matrix,all_shift_comb.transpose()).transpose() #np.dot(orth_matrix,shift)
        tiled_atom_pos1 = np.tile(atom_pos,(1,num_atoms,1))

        distance_matrices = jax.vmap(Structure.create_distance_matrices_single, in_axes=(0,None,None))(shift_pos,tiled_atom_pos1,orth_matrix)

        return distance_matrices

    # the above functions are slow before aligning the systems, use these ones initially
    def create_distance_matrices_single_onp(shift_pos,tiled_atom_pos1, orth_matrix):

        tiled_atom_pos1_trans = tiled_atom_pos1.swapaxes(0,1)
        shifted_tiled_atom_pos1_trans = tiled_atom_pos1_trans + shift_pos
        diff = tiled_atom_pos1 - shifted_tiled_atom_pos1_trans
        distance_matrix = onp.sqrt(onp.square(diff).sum(axis=2))
        return distance_matrix

    def create_distance_matrices_onp(atom_positions, orth_matrix, all_shift_comb):
        num_atoms = len(atom_positions)
        atom_pos = atom_positions.reshape((num_atoms,1,3))
        shift_pos = onp.dot(orth_matrix,all_shift_comb.transpose()).transpose() #np.dot(orth_matrix,shift)
        tiled_atom_pos1 = onp.tile(atom_pos,(1,num_atoms,1))

        #shift_pos = np.matmul(orth_matrix,all_shift_comb)
        distance_matrices = onp.stack([Structure.create_distance_matrices_single_onp(pos,tiled_atom_pos1,orth_matrix) for pos in shift_pos])
        return distance_matrices


    def create_local_neigh_list(num_atoms,real_atom_count,atom_types,distance_matrices,all_shift_comb,cutoff_dict,do_minim):
        '''
        NOTES from the fortran code:
        self indices are located at the end of the global verlet list so that they can be ignored for some calculations (why??)
        '''
        local_body_2_neigh_list = [[] for _ in range(num_atoms)]

        local_body_2_neigh_counts = [0] * num_atoms
        # move from device memory to RAM
        #distance_matrices = onp.array(self.distance_matrices)
        for ctr, [kx,ky,kz] in enumerate(all_shift_comb):
            for i in range(real_atom_count):
                for j in range(i,real_atom_count):
                    # use the precalculated distance
                    # the real box is at index 0
                    distance = distance_matrices[ctr][i][j]
                    type_i = atom_types[i]
                    type_j = atom_types[j]

                    if distance != 0 and distance <= cutoff_dict[(type_i,type_j)] + BUFFER_DIST * do_minim: # use the buffer region if minim. is needed
                        # if there is a bond between real-i and real-j. add this bond id to their inter list

                        # bond neigh. lists should be updated (even if one atom is im.)
                        #inter_id = -1 ---> unknown
                        local_body_2_neigh_list[i].append((j,-1,kx,ky,kz))
                        local_body_2_neigh_counts[i] += 1

                        # in case self image
                        if i != j:
                            local_body_2_neigh_list[j].append((i,-1,-kx,-ky,-kz))
                            local_body_2_neigh_counts[j] += 1



        # equalize the sizes
        max_count = max(local_body_2_neigh_counts)
        min_count = min(local_body_2_neigh_counts)
        # to not have empty lists
        if max_count == 0:
            max_count = 1
        filler_list = [(-1,-1,0,0,0)] * (max_count - min_count)

        for i in range(num_atoms):
            diff = max_count - local_body_2_neigh_counts[i]
            local_body_2_neigh_list[i].extend(filler_list[:diff])

        local_body_2_neigh_counts = onp.array(local_body_2_neigh_counts,dtype=onp.int32)
        local_body_2_neigh_list = onp.array(local_body_2_neigh_list,dtype=onp.int32)

        return local_body_2_neigh_list,local_body_2_neigh_counts


    def body_2_check(type1, type2, bond_params_mask):
        # make sure the param. exists
        if bond_params_mask[type1,type2] == 0 and bond_params_mask[type2,type1] == 0:
            return False
        return True

    def find_and_update_inter_id(local_body_2_neigh_list,local_body_2_neigh_counts,src,dst,shift1,inter_id):
        '''
        Find and update interaction id if dst is in the neigh. list of src
        make sure both atoms are real
        '''
        for neigh in range(local_body_2_neigh_counts[src]):
            ind = int(local_body_2_neigh_list[src][neigh][0])
            shift2 = local_body_2_neigh_list[src][neigh][2:5]

            if ind == dst:
                # sum since the order changes
                is_same = onp.sum(onp.abs(shift1 - shift2)) == 0
                if is_same:
                    local_body_2_neigh_list[src][neigh][1] = inter_id
                    return

    def calculate_2_body_distances_onp(atom_positions,orth_matrix, global_body_2_inter_list,global_body_2_inter_list_mask):
        # updated to support not orth. boxes
        shift = orth_matrix.dot(global_body_2_inter_list[:,4:7].transpose()).transpose()
        pos1 = atom_positions[global_body_2_inter_list[:,0]]
        pos2 = atom_positions[global_body_2_inter_list[:,2]] + shift
        # put 1000 for the masked positions no not get nan later
        distances = onp.sqrt(onp.sum(onp.power((pos1 - pos2),2),axis=1)) + (onp.bitwise_not(global_body_2_inter_list_mask) * 15.0)

        #global_body_2_inter_list[:,0] = distances
        return distances

    def calculate_2_body_distance(pos1, pos2):
        return safe_sqrt(np.sum(np.power((pos1 - pos2),2)))

    def calculate_2_body_distances(atom_positions,orth_matrix, global_body_2_inter_list,global_body_2_inter_list_mask):
        # updated to support not orth. boxes
        shift = orth_matrix.dot(global_body_2_inter_list[:,4:7].transpose()).transpose()
        pos1 = atom_positions[global_body_2_inter_list[:,0]]
        pos2 = atom_positions[global_body_2_inter_list[:,2]] + shift
        # put 1000 for the masked positions no not get nan later
        distances = safe_sqrt(np.sum(np.power((pos1 - pos2),2),axis=1)) + (np.bitwise_not(global_body_2_inter_list_mask) * 15.0)

        #global_body_2_inter_list[:,0] = distances
        return distances
    # use the force field to check if params exist for a given interaction
    # if not, dont add it to the list
    def create_global_body_2_inter_list(real_atom_count,atom_types,atom_names,atom_positions,orth_matrix,local_body_2_neigh_counts,local_body_2_neigh_list, bond_params_mask):
        inter_ctr = 0
        temp_global_body_2_inter_list = []
        temp_triple_bond_body_2_mask = []
        for i in range(real_atom_count):
            cnt = local_body_2_neigh_counts[i]
            #We dont have to calculate the distance her bc if it is in local neigh.lost, the distance is small enough to form a bond
            #if cnt > 0:
                #selected_indices = self.local_body_2_neigh_list[i,:cnt,0]
                #shifts =  self.local_body_2_neigh_list[i,:cnt,2:5]
                # updated to support not orth. boxes
                #sec_atom_pos = self.atom_positions[selected_indices] + self.orth_matrix.dot(shifts.transpose()).transpose()

                #diffs = self.atom_positions[i] - sec_atom_pos
                #distances = onp.sqrt(onp.square(diffs).sum(axis=1))

            for neigh in range(cnt):
                ind1 = i
                ind2 = local_body_2_neigh_list[i][neigh][0]
                shift = local_body_2_neigh_list[i][neigh][2:5]
                #is_real_ind2 = np.sum(np.abs(shift)) == 0

                # to not double count
                if ind2 >= ind1:
                    type1 = atom_types[ind1]
                    type2 = atom_types[ind2]

                    # shift one of the atoms based on the shifting factor
                    #new_pos = self.atom_positions[ind2] + shift * self.box_size
                    # atom 2 is real if there is no shifting


                    #diff = self.atom_positions[ind1] - new_pos

                    #distance = np.sqrt(np.sum(diff * diff))

                    #distance = distances[neigh]
                    # if the bond passes the filtering
                    if Structure.body_2_check(type1, type2, bond_params_mask):
                        local_body_2_neigh_list[ind1][neigh][1] = inter_ctr # store the bond id in the local neigh. list as well
                        #we should also update the neighborhood list of the other atom (if it is real)
                        #if is_real_ind2:
                        # from j->i, the sign if flipped for shift
                        Structure.find_and_update_inter_id(local_body_2_neigh_list,local_body_2_neigh_counts,ind2,ind1,-shift,inter_ctr)
                        # if there is triple bond, update the mask
                        # TODO: can this part be improved later?
                        trip_bond_cond = (atom_names[ind1] == 'C' and atom_names[ind2] == 'O') or (atom_names[ind2] == 'C' and atom_names[ind1] == 'O')
                        temp_triple_bond_body_2_mask.append(trip_bond_cond)

                        # add the shifting factor to support minimization later
                        #self.global_body_2_inter_list[inter_ctr,:] = [distance, ind1, type1, ind2, type2, shift[0], shift[1], shift[2]]
                        temp_global_body_2_inter_list.append((ind1, type1, ind2, type2, shift[0], shift[1], shift[2]))
                        inter_ctr = inter_ctr + 1

        # add one filler interaction in case there is 0
        temp_global_body_2_inter_list.append((-1, -1, -1, -1, 0,0,0))
        temp_triple_bond_body_2_mask.append(0)
        inter_ctr = inter_ctr + 1

        global_body_2_count = inter_ctr
        # update the mask
        global_body_2_inter_list_mask = onp.ones(shape=(global_body_2_count), dtype=onp.bool)
        global_body_2_inter_list_mask[-1] = 0
        #self.global_body_2_inter_list_mask[:self.global_body_2_count] = 1
        global_body_2_inter_list = onp.array(temp_global_body_2_inter_list,dtype=onp.int32)
        triple_bond_body_2_mask = onp.array(temp_triple_bond_body_2_mask)

        global_body_2_distances = Structure.calculate_2_body_distances_onp(atom_positions,orth_matrix,
                                                                                     global_body_2_inter_list,global_body_2_inter_list_mask)

        #self.global_body_2_inter_list[self.global_body_2_count:,0] = 9999
        #self.global_body_2_inter_list[self.global_body_2_count:,[1,2,3,4]] = -1

        return global_body_2_inter_list,global_body_2_inter_list_mask,triple_bond_body_2_mask,global_body_2_distances,global_body_2_count


    def body_3_check(angle, type1, type2, type3,valency_params_mask):

        if valency_params_mask[type1,type2,type3] == 0 and valency_params_mask[type3,type2,type1] == 0:
            return False

        return True

    def calculate_valence_angle(pos1,pos2,pos3):
        '''
        Assume bond between pos2 is the center one
        '''
        vec1 = (pos1 - pos2)
        vec2 = (pos3 - pos2)
        #norm1 = np.linalg.norm(vec1)
        #norm2 = np.linalg.norm(vec2)
        norm1 = safe_sqrt(np.sum(vec1 * vec1))
        norm2 = safe_sqrt(np.sum(vec2 * vec2))

        cos_angle = 0
        #vec1 = np.where(norm1==0, vec1, vec1/norm1)
        #vec2 = np.where(norm2==0, vec2, vec2/norm2)
        # to not get a nan value
        #vec1 = vectorized_cond(norm1 == 0.0, lambda x: vec1, lambda x: vec1/x, norm1)
        #vec2 = vectorized_cond(norm2 == 0.0, lambda x: vec2, lambda x: vec2/x, norm2)
        # if both vectors are the same, we can disc. them by adding different small numbers for stability
        vec1 = vec1 / (norm1 + 1e-10)
        vec2 = vec2 / (norm2 + 1e-6)
        dot_prod = np.dot(vec1, vec2)


        cos_angle = vectorized_cond(np.logical_or(dot_prod >= 1.0, dot_prod < -1.0), lambda x: 0., lambda x: np.arccos(x), dot_prod)
        return cos_angle


    def create_body_3_inter_list(is_periodic,real_atom_count,atom_types,atom_names,atom_positions,orth_matrix,
                              local_body_2_neigh_counts,local_body_2_neigh_list,
                              global_body_2_inter_list,global_body_2_distances,bo,
                              valency_params_mask,
                              cutoff2):

        bo_new = bo - cutoff2
        bo_new = onp.where(bo_new > 0.0, bo_new, 0.0)
        #TODO: later seperate angle calculation and counting
        temp_global_body_3_inter_list  = []
        temp_global_body_3_inter_shift_map = []
        #if k > i and j != i and i != k and j != k:
        inter_ctr = 0
        shifts =  local_body_2_neigh_list[:,:,2:5]
        selected_atom_indices = local_body_2_neigh_list[:,:,0]
        selected_bond_indices = local_body_2_neigh_list[:,:,1]
        shift_indicators = onp.sum(onp.abs(global_body_2_inter_list[selected_bond_indices,4:7] - shifts),axis=2) == 0
        # updated to support not orth. boxes
        #TODO: double check this part
        # matmul does vectorization on the first axis
        neigh_atom_pos = atom_positions[selected_atom_indices] + onp.matmul(orth_matrix, shifts.swapaxes(1,2)).swapaxes(1,2)
        # i is the center atom (which cant be imaginary, the rest can be)
        for i in range(real_atom_count):
            ind2 = i
            type2 = atom_types[ind2]
            pos2 = atom_positions[ind2]

            # find 2different neigh. of i
            for neigh1 in range(local_body_2_neigh_counts[i]):
                ind1 = selected_atom_indices[ind2][neigh1]
                type1 = atom_types[ind1]
                pos1 = neigh_atom_pos[ind2][neigh1]
                bond_ind1 = selected_bond_indices[ind2][neigh1]
                dist1 = global_body_2_distances[bond_ind1]
                # if -1, there is no bond
                if bond_ind1 != -1:

                    for neigh2 in range(neigh1+1,local_body_2_neigh_counts[i]):
                        bond_ind2 = selected_bond_indices[ind2][neigh2]
                        dist2 = global_body_2_distances[bond_ind2]
                        # if -1, there is no bond
                        if neigh1 != neigh2 and bond_ind2 != -1 and dist1 < BODY_3_BOND_CUTOFF and dist2 < BODY_3_BOND_CUTOFF:
                            # make sure atoms are unique


                            ind3 = selected_atom_indices[ind2][neigh2]
                            type3 = atom_types[ind3]
                            pos3 = neigh_atom_pos[ind2][neigh2]

                            # pos2 is the center
                            cos_angle =0
                            #cos_angle = Structure.calculate_valence_angle(pos1,pos2,pos3)
                            # if the bond passes the filtering
                            #if ind2 != ind1 and ind2!= ind3 and Structure.body_3_check(cos_angle, type1, type2, type3,valency_params_mask):
                            if ((ind2 != ind1 or (is_periodic == True and onp.array_equal(pos1,pos2) == False))
                            and (ind2!= ind3 or (is_periodic == True and onp.array_equal(pos2,pos3) == False))
                            and Structure.body_3_check(cos_angle, type1, type2, type3,valency_params_mask)
                            and bo_new[bond_ind1] * bo_new[bond_ind2] > 0.00000): # 0.00001 !Scott Habershon recommendation March 2009

                                #shift_ind1 = np.array_equal(self.global_body_2_inter_list[bond_ind1,5:8], shifts[neigh1])
                                shift_ind1 = shift_indicators[ind2][neigh1]

                                shift_ind2 = shift_indicators[ind2][neigh2]
                                #shift_ind2 = np.array_equal(self.global_body_2_inter_list[bond_ind2,5:8], shifts[neigh2])
                                #self.global_body_3_inter_list[inter_ctr,:] = [cos_angle, ind1, type1, ind2, type2, ind3, type3, bond_ind1,bond_ind2]
                                temp_global_body_3_inter_list.append((ind1, ind2, ind3, bond_ind1,bond_ind2))
                                temp_global_body_3_inter_shift_map.append((shift_ind1,shift_ind2))
                                #self.global_body_3_inter_shift_map[inter_ctr, :] = [shift_ind1,shift_ind2]
                                inter_ctr = inter_ctr + 1


        # add one filler interaction in case there is 0
        temp_global_body_3_inter_list.append((-1, -1, -1, -1, -1))
        temp_global_body_3_inter_shift_map.append((0,1))
        inter_ctr = inter_ctr + 1

        global_body_3_count = inter_ctr

        global_body_3_inter_list_mask = onp.ones(shape=(global_body_3_count),dtype=onp.bool)
        global_body_3_inter_list_mask[-1] = 0 # for the filler inter.
        #self.global_body_3_inter_list_mask[:self.global_body_3_count] = 1

        #self.global_body_3_inter_list[self.global_body_3_count:,[1,2,3,4,5,6,7,8]] = -1
        global_body_3_inter_list = onp.array(temp_global_body_3_inter_list,dtype=onp.int32)
        global_body_3_inter_shift_map = onp.array(temp_global_body_3_inter_shift_map,dtype=onp.bool)
        '''
        self.global_body_3_angles = Structure.calculate_3_body_angles(self.atom_positions,self.box_size,
                                                                       self.global_body_2_inter_list,
                                                                       self.global_body_3_inter_list,
                                                                       self.global_body_3_inter_list_mask,
                                                                       self.global_body_3_inter_shift_map)
        '''
        return global_body_3_inter_list,global_body_3_inter_list_mask,global_body_3_inter_shift_map,global_body_3_count


        # update the mask


    def calculate_3_body_angles(atom_positions,orth_matrix,global_body_2_inter_list,global_body_3_inter_list,global_body_3_inter_list_mask, global_body_3_inter_shift_map):

        atom_indices = global_body_3_inter_list[:,[0,1,2]]
        atom_indices = atom_indices.transpose()
        pos2 = atom_positions[atom_indices[1]]
        bond_indices = global_body_3_inter_list[:,[3,4]]
        bond_indices = bond_indices.transpose()
        shift1 = global_body_2_inter_list[bond_indices[0],4:7]
        shift2 = global_body_2_inter_list[bond_indices[1],4:7]
        shift1_multip = global_body_3_inter_shift_map[:,0] * 2 - 1
        shift2_multip = global_body_3_inter_shift_map[:,1] * 2 - 1
        # updated to support not orth. boxes
        pos1 = atom_positions[atom_indices[0]] + orth_matrix.dot(shift1.transpose()).transpose() * shift1_multip.reshape(-1,1)
        pos3 = atom_positions[atom_indices[2]] + orth_matrix.dot(shift2.transpose()).transpose() * shift2_multip.reshape(-1,1)
        # assign 1 to the masked positions to not get nan in later steps
        angles = jax.vmap(Structure.calculate_valence_angle,in_axes=(0,0,0))(pos1,pos2,pos3) * global_body_3_inter_list_mask + (np.bitwise_not(global_body_3_inter_list_mask))
        #global_body_3_inter_list[:,0] = angles

        return angles

    def body_4_check(dist1,dist2,dist3, type1, type2, type3, type4,torsion_params_mask):
        # param existence
        if torsion_params_mask[type1, type2, type3, type4] == 0 and torsion_params_mask[type4, type3, type2, type1] == 0:
            return False

        return True

    def create_body_4_inter_list_fast(is_periodic,real_atom_count,atom_types,atom_names,atom_positions,orth_matrix,
                                   local_body_2_neigh_counts,local_body_2_neigh_list,
                                   global_body_2_inter_list,global_body_2_distances,bo,global_body_2_count,
                                   torsion_params_mask,cutoff2):
        inter_ctr = 0
        temp_global_body_4_inter_list = []
        temp_global_body_4_inter_shift = []

        #selected_bond_indices = self.local_body_2_neigh_list[:,:,1]
        # updated to support not orth. boxes
        #TODO: double check this part
        # matmul does vectorization on the first axis
        #start = time.time()

        #calculate atom positions for all of the bonds since we are going to need it later
        #all_bonds_ind1 = self.global_body_2_inter_list[:self.global_body_2_count,0].astype(onp.int32)
        #all_bonds_ind2 = self.global_body_2_inter_list[:self.global_body_2_count,2].astype(onp.int32)
        all_bonds_shift = global_body_2_inter_list[:global_body_2_count,4:]
        #all_bonds_pos1 = self.atom_positions[all_bonds_ind1]
        # updated to support not orth. boxes
        #all_bonds_pos2 = self.atom_positions[all_bonds_ind2] + self.orth_matrix.dot(all_bonds_shift.transpose()).transpose()

        for b1 in range(global_body_2_count):
            ind2, type2, ind3, type3 = global_body_2_inter_list[b1,:4]
            dist1 = global_body_2_distances[b1]
            #can be removed later,since the condition for the torsion involves
            #multiplication of 3 bond energy and threshoding it, it is a reasonable condition
            if dist1 > BODY_4_BOND_CUTOFF:
                continue

            ind2 = int(ind2)
            ind3 = int(ind3)
            type2 = int(type2)
            type3 = int(type3)
            shift1 = all_bonds_shift[b1]
            #pos2 = all_bonds_pos1[b1]
            #pos3 = all_bonds_pos2[b1]
            all_neigh_pairs = onp.array(onp.meshgrid(onp.arange(local_body_2_neigh_counts[ind2]), onp.arange(local_body_2_neigh_counts[ind3]))).T.reshape(-1,2)
            neigh1_inds = all_neigh_pairs[:,0]
            neigh2_inds = all_neigh_pairs[:,1]
            bond_indices1 = local_body_2_neigh_list[ind2,neigh1_inds,1]
            bond_indices2 = local_body_2_neigh_list[ind3,neigh2_inds,1]

            bond_check = onp.logical_and(bond_indices1!=-1, bond_indices2!=-1)
            dist_check = onp.logical_and(global_body_2_distances[bond_indices1] < BODY_4_BOND_CUTOFF, global_body_2_distances[bond_indices2] < BODY_4_BOND_CUTOFF)
            bo_check =  onp.where(bo[bond_indices1] * bo[bond_indices2] * bo[b1] > cutoff2, 1, 0)
            #bo_check=1.0
            bond_and_dist_check = onp.logical_and(bond_check, dist_check)
            bond_and_dist_check = onp.logical_and(bond_and_dist_check, bo_check)
            '''
            print('*' *20)
            print(self.global_body_2_distances[bond_indices1] < BODY_4_BOND_CUTOFF)
            print(self.global_body_2_distances[bond_indices2] < BODY_4_BOND_CUTOFF)
            print('*' *20)
            print('bond_check',bond_check)
            print('dist_check',dist_check)
            print(bond_and_dist_check)
            '''
            # filtering
            neigh1_inds = neigh1_inds[bond_and_dist_check]
            neigh2_inds = neigh2_inds[bond_and_dist_check]
            bond_indices1 = bond_indices1[bond_and_dist_check]
            bond_indices2 = bond_indices2[bond_and_dist_check]

            ind1 = local_body_2_neigh_list[ind2,neigh1_inds,0]
            shift2 = local_body_2_neigh_list[ind2,neigh1_inds,2:]
            type1 = atom_types[ind1]

            ind4 = local_body_2_neigh_list[ind3,neigh2_inds,0]
            shift3 = local_body_2_neigh_list[ind3,neigh2_inds,2:]
            type4 = atom_types[ind4]

            # index comparisons
            index_comparesions = onp.where(ind2==ind4, False,
                                  (onp.where(ind3==ind1, False, ind1!=ind4 )))
            # param existence check
            param_existence = onp.logical_or(torsion_params_mask[type1, type2, type3, type4] != 0, torsion_params_mask[type4, type3, type2, type1] != 0)
            sum_shift = shift3 + shift1

            last_cond = onp.logical_and(param_existence,index_comparesions)

            countt = onp.sum(last_cond)
            additional_shift = onp.zeros((countt,12),dtype=onp.int8)
            additional_shift[:,:3] = shift2[last_cond]
            additional_shift[:,6:9] = shift1
            additional_shift[:,9:] = sum_shift[last_cond]

            temp_global_body_4_inter_shift.extend(additional_shift.tolist())

            additional_inter_list = onp.zeros((countt,7),dtype=onp.int32)
            additional_inter_list[:,0] = ind1[last_cond]
            additional_inter_list[:,1] = ind2
            additional_inter_list[:,2] = ind3
            additional_inter_list[:,3] = ind4[last_cond]
            additional_inter_list[:,4] = bond_indices1[last_cond]
            additional_inter_list[:,5] = b1
            additional_inter_list[:,6] = bond_indices2[last_cond]
            temp_global_body_4_inter_list.extend(additional_inter_list.tolist())

            inter_ctr = inter_ctr + countt

        #end = time.time()
        #print("4 body list creation:{}".format(end - start))

        # add one filler interaction in case there is 0
        temp_global_body_4_inter_list.append((-1,-1,-1,-1,-1,-1,-1))
        temp_global_body_4_inter_shift.append((0,0,0,0,0,0,0,0,0,0,0,0))
        inter_ctr = inter_ctr + 1

        global_body_4_count = inter_ctr

        global_body_4_inter_list = onp.array(temp_global_body_4_inter_list,dtype=onp.int32)
        global_body_4_inter_shift = onp.array(temp_global_body_4_inter_shift,dtype=onp.int8)
        global_body_4_inter_list_mask = onp.ones(shape=(global_body_4_count),dtype=onp.bool)
        global_body_4_inter_list_mask[-1] = 0
        '''
        self.global_body_4_angles = Structure.calculate_body_4_angles_new(self.atom_positions,
                                                                           self.box_size,
                                                                           self.global_body_4_inter_list,
                                                                           self.global_body_4_inter_list_mask,
                                                                           self.global_body_4_inter_shift
                                                                           )
        '''
        '''
        # update the mask
        #start = time.time()
        Structure.calculate_body_4_angles(self, all_pos)
        #end = time.time()
        #print("4 body angle calculation:{}".format(end - start))
        self.global_body_4_inter_list_mask[:self.global_body_4_count] = 1

        self.global_body_4_inter_list[self.global_body_4_count:,[0,1,2,3,4]] = 0
        self.global_body_4_inter_list[self.global_body_4_count:,5:] = -1
        '''

        return global_body_4_inter_list,global_body_4_inter_shift,global_body_4_inter_list_mask,global_body_4_count


    def calculate_body_4_angle_single(pos1,pos2,pos3,pos4):
        v1 = pos2 - pos1
        v2 = pos3 - pos2
        v3 = pos4 - pos3
        c0 = v2[1] * v3[2] - v2[2] * v3[1]
        c1 = v2[2] * v3[0] - v2[0] * v3[2]
        c2 = v2[0] * v3[1] - v2[1] * v3[0]
        trip_prod = v1[0] * c0 + v1[1] * c1 +v1[2] * c2 + 1e-10
        val = Structure.calculate_body_4_angles_single(pos1,pos2,pos3,pos4)[-1]
        angle = np.arccos(val)


        return angle

    def calculate_body_4_angles_single(pos1,pos2,pos3,pos4):
        # [1-(2-3)-4] ---- (2-3 is the center)
        angle_123 = Structure.calculate_valence_angle(pos1,pos2,pos3)
        angle_234 = Structure.calculate_valence_angle(pos2,pos3,pos4)
        coshd = np.cos(angle_123)
        coshe = np.cos(angle_234)
        sinhd = np.sin(angle_123)
        sinhe = np.sin(angle_234)

        r4 = Structure.calculate_distance(pos1,pos4)
        d142 = r4 * r4

        rla = Structure.calculate_distance(pos1,pos2)
        rlb = Structure.calculate_distance(pos2,pos3)
        rlc = Structure.calculate_distance(pos3,pos4)

        tel= (rla*rla+rlb*rlb+rlc*rlc-d142-2.0*(rla*rlb*coshd-rla*rlc*
            coshd*coshe+rlb*rlc*coshe))
        poem=2.0*rla*rlc*sinhd*sinhe
        #poem2=poem*poem


        poem = np.where(poem < 1e-20, 1e-20, poem)

        arg=tel/poem

        arg = np.clip(arg, -1.0, 1.0)

        return np.array([coshd,coshe,sinhd,sinhe,arg])


    def calculate_body_4_angles_new(atom_positions,orth_matrix,global_body_4_inter_list,global_body_4_inter_list_mask,global_body_4_inter_shift):
        # updated to support not orth. boxes
        pos1 = atom_positions[global_body_4_inter_list[:,0]] + orth_matrix.dot(global_body_4_inter_shift[:,:3].transpose()).transpose()
        pos2 = atom_positions[global_body_4_inter_list[:,1]] + orth_matrix.dot(global_body_4_inter_shift[:,3:6].transpose()).transpose()
        pos3 = atom_positions[global_body_4_inter_list[:,2]] + orth_matrix.dot(global_body_4_inter_shift[:,6:9].transpose()).transpose()
        pos4 = atom_positions[global_body_4_inter_list[:,3]] + orth_matrix.dot(global_body_4_inter_shift[:,9:12].transpose()).transpose()

        body_4_angles = jax.vmap(Structure.calculate_body_4_angles_single,in_axes=(0,0,0,0))(pos1,pos2,pos3,pos4)

        return body_4_angles

    def calculate_body_4_angles(self, all_pos):
        pos1 = all_pos[:self.global_body_4_count,:3]
        pos2 = all_pos[:self.global_body_4_count,3:6]
        pos3 = all_pos[:self.global_body_4_count,6:9]
        pos4 = all_pos[:self.global_body_4_count,9:12]

        [coshd,coshe,sinhd,sinhe,arg] = jax.vmap(Structure.calculate_body_4_angles_single,in_axes=(0,0,0,0))(pos1,pos2,pos3,pos4)
        self.global_body_4_inter_list[:self.global_body_4_count ,0] = coshd[:self.global_body_4_count]
        self.global_body_4_inter_list[:self.global_body_4_count ,1] = coshe[:self.global_body_4_count]
        self.global_body_4_inter_list[:self.global_body_4_count ,2] = sinhd[:self.global_body_4_count]
        self.global_body_4_inter_list[:self.global_body_4_count ,3] = sinhe[:self.global_body_4_count]
        self.global_body_4_inter_list[:self.global_body_4_count ,4] = arg[:self.global_body_4_count]


    def create_verlet_list(self):
        pass

    def hbond_check(hbond_params_mask,type1,type2,type3):
        return hbond_params_mask[type1,type2,type3]

    def create_global_hbond_inter_list(is_periodic,do_minim,real_atom_count,atom_types,atom_names,atom_positions,orth_matrix,
                                   distance_matrices,all_shift_comb,
                                   local_body_2_neigh_counts,local_body_2_neigh_list,
                                   global_body_2_inter_list,global_body_2_distances,global_body_2_count,
                                   ff_nphb,hbond_params_mask):
        global_hbond_inter_list = []
        global_hbond_shift_list = []
        global_hbond_count = 0
        #TODO: improve this part later by using a neigh. list
        distance_matrices = onp.array(distance_matrices)
        for i1 in range(real_atom_count):
            for i2 in range(real_atom_count):
                all_dist = distance_matrices[:,i1,i2]
                type1 = atom_types[i1]
                type2 = atom_types[i2]

                ihhb1 = ff_nphb[type1]
                ihhb2 = ff_nphb[type2]
                # i1 is the H ATOM (center)
                if ihhb1 == 1 and ihhb2 == 2:
                    for ctr, [kx,ky,kz] in enumerate(all_shift_comb):
                        dist = all_dist[ctr]
                        if dist < HBOND_CUTOFF + BUFFER_DIST * do_minim:
                            # go through neigh. of i1
                            for n1 in range(local_body_2_neigh_counts[i1]):
                                b_ind = int(local_body_2_neigh_list[i1][n1][1])
                                i3 = local_body_2_neigh_list[i1][n1][0]
                                type3 = atom_types[i3]
                                ihhb3 = ff_nphb[type3]
                                if b_ind != -1 and ihhb3 == 2 and i3 != i2 and global_body_2_distances[b_ind] < CLOSE_NEIGH_CUTOFF and Structure.hbond_check(hbond_params_mask,type3,type1,type2): #double check if this is enough
                                    # the order from the fortran code (i3,i1,i2)
                                    global_hbond_inter_list.append((i3,type3,i1,type1,i2,type2, b_ind))
                                    shift2 = local_body_2_neigh_list[i1][n1][2:]
                                    global_hbond_shift_list.append((kx,ky,kz,shift2[0],shift2[1],shift2[2]))
                                    global_hbond_count = global_hbond_count + 1

        #add non-effective inter.
        global_hbond_inter_list.append((-1,-1,-1,-1,-1,-1,-1))
        global_hbond_shift_list.append((0,0,0,0,0,0))
        global_hbond_count = global_hbond_count + 1

        global_hbond_inter_list = onp.array(global_hbond_inter_list)
        global_hbond_shift_list = onp.array(global_hbond_shift_list,dtype=onp.int32)
        global_hbond_count = global_hbond_count
        global_hbond_inter_list_mask = onp.ones(global_hbond_count,dtype=onp.int32)
        global_hbond_inter_list_mask[-1] = 0

        return global_hbond_inter_list,global_hbond_shift_list,global_hbond_inter_list_mask,global_hbond_count



    def calculate_global_hbond_angles_and_dist(atom_positions, orth_matrix, global_hbond_inter_list, global_hbond_shift_list, global_hbond_inter_list_mask):
        atom_indices = global_hbond_inter_list[:,[0,2,4]]
        atom_indices = atom_indices.transpose()
        # center atom (H)
        pos2 =  atom_positions[atom_indices[1]]

        shift1 = global_hbond_shift_list[:,:3]
        shift2 = global_hbond_shift_list[:,3:]
        pos1 = atom_positions[atom_indices[0]] + orth_matrix.dot(shift1.transpose()).transpose()
        pos3 = atom_positions[atom_indices[2]] + orth_matrix.dot(shift2.transpose()).transpose()

        # assign 1 to the masked positions to not get nan in later steps

        angles = jax.vmap(Structure.calculate_valence_angle,in_axes=(0,0,0))(pos1,pos2,pos3)# * global_hbond_inter_list_mask + (np.bitwise_not(global_hbond_inter_list_mask))
        dist =  jax.vmap(Structure.calculate_distance,in_axes=(0,0))(pos2,pos3)# + (np.bitwise_not(global_hbond_inter_list_mask) * 100.0)
        return np.stack((angles,dist)).transpose()

    def calculate_distance(pos1,pos2):
        diff = pos2 - pos1
        distance = safe_sqrt(np.sum(diff * diff))
        return distance


    def create_valency_mask(self):


        for i in range(self.num_atoms):
            for j in range(self.num_atoms):
                for k in range(self.num_atoms):
                    if k > i and j != i and i != k and j != k:
                        self.valency_angle_mask[i,j,k] = 1

    def create_torsion_mask(self):
        na = self.num_atoms

        for i1 in range(self.num_atoms):
            for i2 in range(self.num_atoms):
                for i3 in range(i2, self.num_atoms):
                    for i4 in range(self.num_atoms):
                        if i1 != i2 and i2 != i3 and i3 != i4 and i1 != i3 and i1 != i4 and i2 != i4:
                            self.torsion_angle_mask[i1,i2,i3,i4] = 1


    def fill_atom_types(self, force_field):
        num_atoms = self.num_atoms

        for i in range(num_atoms):
            if self.atom_names[i] == 'FILLER': # filler atom
                self.atom_types[i] = -1 # last index
            else:
                self.atom_types[i] = force_field.name_2_index[self.atom_names[i]]
        return self.atom_types


    # to use to gemerate inter lists
    def flatten_no_inter_list(self):
        return (self.is_periodic,
             self.do_minimization,
             self.num_atoms,
             self.real_atom_count,
             self.atom_types,
             self.atom_names,
             self.atom_positions,
             self.orth_matrix,
             self.all_shift_comb)

    def flatten(self):
        self.flattened_system = [self.atom_types,
                                 self.atom_mask,
                                 self.distance_matrices,
                                 self.global_body_2_inter_list,
                                 self.global_body_2_inter_list_mask,
                                 self.local_body_2_neigh_list,
                                 self.triple_bond_body_2_mask,
                                 self.global_body_3_inter_list,
                                 self.global_body_3_inter_list_mask,
                                 self.global_body_4_inter_list,
                                 self.global_body_4_inter_list_mask]

        return self.flattened_system
