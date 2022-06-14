#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains force field related code

Author: Mehmet Cagri Kaymak
"""


import numpy as onp
import jax.numpy as np
import jax
import pickle

# it fixes nan values issue, from: https://github.com/google/jax/issues/1052
def vectorized_cond(pred, true_fun, false_fun, operand):
  # true_fun and false_fun must act elementwise (i.e. be vectorized)
  #how to use: grad(lambda x: vectorized_cond(x > 0.5, lambda x: np.arctan2(x, x), lambda x: 0., x))(0.)
  true_op = np.where(pred, operand, 0)
  false_op = np.where(pred, 0, operand)
  return np.where(pred, true_fun(true_op), false_fun(false_op))


#https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
@jax.custom_jvp
def safe_sqrt(x):
  return np.sqrt(x)

@safe_sqrt.defjvp
def safe_sqrt_jvp(primals, tangents):
  x = primals[0]
  x_dot = tangents[0]
  #print(x[0])
  primal_out = safe_sqrt(x)
  tangent_out = 0.5 * x_dot / np.where(x > 0, primal_out, np.inf)
  return primal_out, tangent_out


#TODO: this part needs to be improved for faster execution
MAX_NUM_ATOM_TYPES = 20
MY_ATOM_INDICES = {a for a in list(range(MAX_NUM_ATOM_TYPES))}


# These index arrays should be global to not get TypeError: list indices must be integers or slices, not JaxprTracer
# Follow this for solution: https://github.com/google/jax/issues/2962
# These lists can be moved somewhere else later
#body_2_indices = np.tril_indices(TOTAL_ATOM_TYPES,k=-1)
body_3_indices_src = [[],[],[]]
body_3_indices_dst = [[],[],[]]
body_4_indices_src = [[],[],[],[]]
body_4_indices_dst = [[],[],[],[]]

hbond_indices_src = [[],[],[]]
hbond_indices_dst = [[],[],[]]

TYPE = onp.float32
c1c=332.0638      #Coulomb energy conversion

rdndgr=180.0/onp.pi
dgrrdn=1.0/rdndgr


class ForceField:
    def __init__(self, total_num_atom_types=MAX_NUM_ATOM_TYPES, cutoff2 = 0.001):
        self.total_num_atom_types = total_num_atom_types
        self.name_2_index = dict()
        self.params_to_indices = dict()

        # charge solver and coulomb pot. related parameters
        self.electronegativity = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE) #elc
        self.idempotential = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE) #eta
        self.gamma = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)

        # vdw related parameters
        self.rvdw = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.p1co = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.p1co_off = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.p1co_off_mask = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=onp.bool)

        self.eps = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.p2co = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.p2co_off = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.p2co_off_mask = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=onp.bool)

        self.alf = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.p3co = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.p3co_off = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.p3co_off_mask = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=onp.bool)

        self.vop = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)

        self.amas = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)

        self.vdw_shiedling = 0.0 #vpar(29)

        # tapering
        self.low_tap_rad = 0.0
        self.up_tap_rad = 10.0

        # bond energy related parameters
        self.rat = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.rob1 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.rob1_off = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.rob1_off_mask = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=onp.bool)

        self.rapt = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.rob2 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.rob2_off = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.rob2_off_mask = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=onp.bool)

        self.vnq = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.rob3 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.rob3_off = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.rob3_off_mask = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=onp.bool)

        self.ptp = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.pdp = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.popi = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.pdo = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.bop1 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.bop2 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)

        self.de1 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.de2 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.de3 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.psp = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.psi = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)

        self.cutoff = 0

        self.trip_stab4 = 0
        self.trip_stab5 = 0
        self.trip_stab8 = 0
        self.trip_stab11 = 0 # stab. energy


        self.aval = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.vval3 = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.bo131 = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.bo132 = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.bo133 = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.ovc = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.v13cor = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)

        self.over_coord1 = 0
        self.over_coord2 = 0



        # 1 if (i,j) bond parameter exists, otherwise 0.
        self.bond_params_mask = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=onp.int32)


        # valency related parameters
        self.cutoff2 = cutoff2
        # from control, BO-cutoff for valency angles and torsion angles
        self.val_par3 = 0.0
        self.val_par15 = 0.0
        self.val_par17 = 0.0
        self.val_par18 = 0.0
        self.val_par20 = 0.0
        self.val_par21 = 0.0
        self.val_par22 = 0.0
        self.val_par31 = 0.0
        self.val_par34 = 0.0
        self.val_par39 = 0.0


        self.stlp = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.valf = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.vval1 = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.vval2 = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.vval3 = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.vval4 = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)

        self.vkac = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.th0 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.vka = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.vkap = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.vka3 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.vka8 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.vval2 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)





        # 1 if (i,j, k) parameter exists, otherwise 0.
        self.valency_params_mask = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=onp.int32)


        #lone pair
        self.vlp1 = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.par_16 = 0.0

        # over-under coordination
        self.valp1 = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)
        self.vovun = onp.zeros(shape=(self.total_num_atom_types), dtype=TYPE)

        self.vover = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)

        self.par_6 = 0.0
        self.par_7 = 0.0
        self.par_9 = 0.0
        self.par_10 = 0.0
        self.par_32 = 0.0
        self.par_33 = 0.0


        self.torsion_params_mask = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=onp.int32)
        # torsion angle params
        self.v1 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.v2 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.v3 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.v4 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.vconj = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)

        self.par_24 = 0.0
        self.par_25 = 0.0
        self.par_26 = 0.0
        self.par_28 = 0.0


        # h-bond parameters
        self.nphb = onp.zeros(shape=(self.total_num_atom_types), dtype=onp.int32)
        self.rhb = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.dehb = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.vhb1 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.vhb2 = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=TYPE)
        self.hbond_params_mask = onp.zeros(shape=(self.total_num_atom_types,self.total_num_atom_types,self.total_num_atom_types), dtype=onp.int32)

        # these will be part of non_dif and their type will be onp.array or np.array
        self.body_3_indices_src = []
        self.body_3_indices_dst = []
        self.body_4_indices_src = []
        self.body_4_indices_dst = []

        # array for flattened force field
        self.flattened_force_field = []

    def init_params_for_filler_atom_type(self):
        #TODO: make sure that index -1 doesnt belong to a real atom!!!
        self.rat[-1]  = 1
        self.rapt[-1] = 1
        self.vnq[-1] = 1

        self.rvdw[-1] = 1
        self.eps[-1] = 1
        self.alf[-1] = 1

        self.vop[-1] = 1

        self.gamma[-1] = 1
        self.electronegativity[-1] = 1
        self.idempotential[-1] = 1
        #derivative of sqrt is nan at 0
        self.bo131[-1] = 1
        self.bo132[-1] = 1
        self.bo133[-1] = 1


    def random_init(self):
        key = jax.random.PRNGKey(3)
        #self.gamma = np.abs(jax.random.normal(key, shape=(TOTAL_ATOM_TYPES,))) / 2
        #self.idempotential = np.abs(jax.random.normal(key, shape=(TOTAL_ATOM_TYPES,))) * 10
        #self.electronegativity = np.abs(jax.random.normal(key, shape=(TOTAL_ATOM_TYPES,))) * 10
        #gamma_vals = [0.8203, 1.0898]
        #idem_vals = [9.6093,8.3122]
        #electro_vals = [3.7248, 8.5000]
        random_val = jax.random.truncated_normal(key,lower=0.1, upper=0.3, shape=(1,))[0]
        self.gamma = self.gamma + random_val
        self.idempotential = self.idempotential + random_val
        self.electronegativity = self.electronegativity + random_val

        self.p1co = self.p1co + random_val
        self.p2co = self.p2co + random_val
        self.p3co = self.p3co + random_val
        self.vop = self.vop + random_val
        self.vdw_shiedling = self.vdw_shiedling + random_val

    def flatten(self):
        # total size 2 from tapering + gamma + idempotential + electronegativity
        self.flattened_force_field = [self.gamma,
                                      self.idempotential,
                                      self.electronegativity,

                                      self.p1co,
                                      self.p2co,
                                      self.p3co,
                                      self.vop,
                                      self.vdw_shiedling,

                                      #self.low_tap_rad,
                                      #self.up_tap_rad,

                                      #self.rat,
                                      self.rob1, #8
                                      #self.rapt,
                                      self.rob2,
                                      #self.vnq,
                                      self.rob3,
                                      self.ptp,
                                      self.pdp,
                                      self.popi, # 13
                                      self.pdo,
                                      self.bop1,
                                      self.bop2,
                                      self.de1,
                                      self.de2,
                                      self.de3,
                                      self.psp,
                                      self.psi,
                                      #self.bond_params_mask,
                                      #self.cutoff,

                                      self.trip_stab4, #22
                                      self.trip_stab5,
                                      self.trip_stab8,
                                      self.trip_stab11,

                                      self.aval,
                                      self.vval3,
                                      self.bo131,
                                      self.bo132,
                                      self.bo133,
                                      #self.ovc,
                                      #self.v13cor,
                                      self.over_coord1,
                                      self.over_coord2,

                                      # valency parameters,
                                      self.valf, # 33
                                      self.stlp,
                                      self.vval1,
                                      self.vval2, #36
                                      #self.vval3,
                                      self.vval4,

                                      self.vkac,
                                      self.th0, #39
                                      self.vka, #40
                                      self.vkap,
                                      self.vka3, #42
                                      self.vka8, #43

                                      self.val_par3,
                                      self.val_par15,
                                      self.val_par17,
                                      self.val_par18,
                                      self.val_par20,
                                      self.val_par21,
                                      self.val_par22,
                                      self.val_par31,
                                      self.val_par34,
                                      self.val_par39, #53

                                      # lone pair parameters,
                                      self.vlp1,
                                      self.par_16,

                                      #overunder coordination
                                      self.amas, #56
                                      self.vover,
                                      self.valp1,
                                      self.vovun,
                                      self.par_6,
                                      self.par_7,
                                      self.par_9,
                                      self.par_10,
                                      self.par_32,
                                      self.par_33, #65


                                      #torsion
                                      self.v1,
                                      self.v2,
                                      self.v3,
                                      self.v4,
                                      self.vconj,
                                      self.par_24,
                                      self.par_25,
                                      self.par_26,
                                      self.par_28, #74

                                      # extra
                                      self.rat,
                                      self.rapt,
                                      self.vnq,
                                      self.rvdw,
                                      self.eps,
                                      self.alf, #80

                                      self.rob1_off,
                                      self.rob2_off,
                                      self.rob3_off,
                                      self.p1co_off,
                                      self.p2co_off,
                                      self.p3co_off,    #86

                                      #hbond
                                      self.rhb,
                                      self.dehb,
                                      self.vhb1,
                                      self.vhb2    #90


                                    ]



        self.flatten_non_dif_params()
        #return self.flattened_force_field

    def unflatten(self):


        self.gamma = self.flattened_force_field[0]
        self.idempotential = self.flattened_force_field[1]
        self.electronegativity = self.flattened_force_field[2]

        self.p1co = self.flattened_force_field[3]
        self.p2co = self.flattened_force_field[4]
        self.p3co = self.flattened_force_field[5]
        self.vop = self.flattened_force_field[6]
        self.vdw_shiedling = self.flattened_force_field[7]

        self.rob1 = self.flattened_force_field[8]
        self.rob2 = self.flattened_force_field[9]
        self.rob3 = self.flattened_force_field[10]
        self.ptp = self.flattened_force_field[11]
        self.pdp = self.flattened_force_field[12]
        self.popi = self.flattened_force_field[13]
        self.pdo = self.flattened_force_field[14]
        self.bop1 = self.flattened_force_field[15]
        self.bop2 = self.flattened_force_field[16]
        self.de1 = self.flattened_force_field[17]
        self.de2 = self.flattened_force_field[18]
        self.de3 = self.flattened_force_field[19]
        self.psp = self.flattened_force_field[20]
        self.psi = self.flattened_force_field[21]

        self.trip_stab4 = self.flattened_force_field[22]
        self.trip_stab5 = self.flattened_force_field[23]
        self.trip_stab8 = self.flattened_force_field[24]
        self.trip_stab11 = self.flattened_force_field[25]

        self.aval = self.flattened_force_field[26]
        self.vval3 = self.flattened_force_field[27]
        self.bo131 = self.flattened_force_field[28]
        self.bo132 = self.flattened_force_field[29]
        self.bo133 = self.flattened_force_field[30]
        self.over_coord1 = self.flattened_force_field[31]
        self.over_coord2 = self.flattened_force_field[32]

        self.valf = self.flattened_force_field[33]
        self.stlp = self.flattened_force_field[34]
        self.vval1 = self.flattened_force_field[35]
        self.vval2 = self.flattened_force_field[36]
        self.vval4 = self.flattened_force_field[37]

        self.vkac = self.flattened_force_field[38]
        self.th0 = self.flattened_force_field[39]
        self.vka = self.flattened_force_field[40]
        self.vkap = self.flattened_force_field[41]
        self.vka3 = self.flattened_force_field[42]
        self.vka8 = self.flattened_force_field[43]

        self.val_par3 = self.flattened_force_field[44]
        self.val_par15 = self.flattened_force_field[45]
        self.val_par17 = self.flattened_force_field[46]
        self.val_par18 = self.flattened_force_field[47]
        self.val_par20 = self.flattened_force_field[48]
        self.val_par21 = self.flattened_force_field[49]
        self.val_par22 = self.flattened_force_field[50]
        self.val_par31 = self.flattened_force_field[51]
        self.val_par34 = self.flattened_force_field[52]
        self.val_par39 = self.flattened_force_field[53]

        self.vlp1 = self.flattened_force_field[54]
        self.par_16 = self.flattened_force_field[55]

        self.amas = self.flattened_force_field[56]
        self.vover = self.flattened_force_field[57]
        self.valp1 = self.flattened_force_field[58]
        self.vovun = self.flattened_force_field[59]
        self.par_6 = self.flattened_force_field[60]
        self.par_7 = self.flattened_force_field[61]
        self.par_9 = self.flattened_force_field[62]
        self.par_10 = self.flattened_force_field[63]
        self.par_32 = self.flattened_force_field[64]
        self.par_33 = self.flattened_force_field[65]

        self.v1 = self.flattened_force_field[66]
        self.v2 = self.flattened_force_field[67]
        self.v3 = self.flattened_force_field[68]
        self.v4 = self.flattened_force_field[69]
        self.vconj = self.flattened_force_field[70]
        self.par_24 = self.flattened_force_field[71]
        self.par_25 = self.flattened_force_field[72]
        self.par_26 = self.flattened_force_field[73]
        self.par_28 = self.flattened_force_field[74]

        self.rat = self.flattened_force_field[75]
        self.rapt = self.flattened_force_field[76]
        self.vnq = self.flattened_force_field[77]
        self.rvdw = self.flattened_force_field[78]
        self.eps = self.flattened_force_field[79]
        self.alf = self.flattened_force_field[80]

        self.rob1_off = self.flattened_force_field[81]
        self.rob2_off = self.flattened_force_field[82]
        self.rob3_off = self.flattened_force_field[83]
        self.p1co_off = self.flattened_force_field[84]
        self.p2co_off = self.flattened_force_field[85]
        self.p3co_off = self.flattened_force_field[86]

        self.rhb = self.flattened_force_field[87]
        self.dehb = self.flattened_force_field[88]
        self.vhb1 = self.flattened_force_field[89]
        self.vhb2 = self.flattened_force_field[90]


    def flatten_non_dif_params(self):
        rob1_mask = np.where(self.rob1 > 0.0, 1.0, 0.0)
        rob1_mask = rob1_mask + np.triu(rob1_mask, k=1).transpose()
        rob2_mask = np.where(self.rob2 > 0.0, 1.0, 0.0)
        rob2_mask = rob2_mask + np.triu(rob2_mask, k=1).transpose()
        rob3_mask = np.where(self.rob3 > 0.0, 1.0, 0.0)
        rob3_mask = rob3_mask + np.triu(rob3_mask, k=1).transpose()
        self.non_dif_params = [self.low_tap_rad, #0
                              self.up_tap_rad,
                              self.bond_params_mask,
                              self.cutoff,
                              rob1_mask,
                              rob2_mask,
                              rob3_mask,
                              self.ovc,
                              self.v13cor,
                              self.valency_params_mask,
                              self.torsion_params_mask,
                              self.cutoff2,

                              self.p1co_off_mask, #12
                              self.p2co_off_mask,
                              self.p3co_off_mask,

                              self.rob1_off_mask,
                              self.rob2_off_mask,
                              self.rob3_off_mask,
                              self.name_2_index, #18

                              self.body_3_indices_src, #19
                              self.body_3_indices_dst,
                              self.body_4_indices_src,
                              self.body_4_indices_dst,
                              self.nphb
                              ]

        for i in range(len(self.non_dif_params)):
            self.non_dif_params[i] = jax.device_put(self.non_dif_params[i])

        return self.non_dif_params

    def save(self, name="force_field.pkl"):
        with open(name, 'wb') as f:
            pickle.dump(self.flattened_force_field, f)


    def load(self, name="force_field.pkl"):
        with open(name, "rb") as input_file:
            self.flattened_force_field = pickle.load(input_file)

def symm_force_field(flattened_force_field,flattened_non_dif_params):
    # 2 body-params
    # for now global
    body_2_indices = np.tril_indices(len(flattened_force_field[0]),k=-1)
    body_3_indices_src = flattened_non_dif_params[19]
    body_3_indices_dst = flattened_non_dif_params[20]
    body_4_indices_src = flattened_non_dif_params[21]
    body_4_indices_dst = flattened_non_dif_params[22]
    #off diag. ones
    for i in range(3, 6):
        flattened_force_field[i] = jax.ops.index_update(flattened_force_field[i],
                    body_2_indices, flattened_force_field[i].transpose()[body_2_indices])

    for i in range(81, 87):
        flattened_force_field[i] = jax.ops.index_update(flattened_force_field[i],
                    body_2_indices, flattened_force_field[i].transpose()[body_2_indices])

    for i in range(8, 22):
        flattened_force_field[i] = jax.ops.index_update(flattened_force_field[i],
                    body_2_indices, flattened_force_field[i].transpose()[body_2_indices])

    flattened_force_field[57] = jax.ops.index_update(flattened_force_field[57],  #vover
                body_2_indices, flattened_force_field[57].transpose()[body_2_indices])

    # 3-body parameters
    flattened_force_field[36] = jax.ops.index_update(flattened_force_field[36],
                                body_3_indices_dst, flattened_force_field[36][body_3_indices_src])


    for i in range(38, 44):
        flattened_force_field[i] = jax.ops.index_update(flattened_force_field[i],
                                    body_3_indices_dst, flattened_force_field[i][body_3_indices_src])
    #4-body params
    for i in range(66, 71):
        flattened_force_field[i] = jax.ops.index_update(flattened_force_field[i],
                                    body_4_indices_dst, flattened_force_field[i][body_4_indices_src])

    return     flattened_force_field

def handle_offdiag(flattened_force_field,flattened_non_dif_params):
    '''
                          self.p1co_off_mask, #12
                              self.p2co_off_mask,
                              self.p3co_off_mask,

                              self.rob1_off_mask,#15
                              self.rob2_off_mask,
                              self.rob3_off_mask
    '''
    num_rows = flattened_force_field[75].shape[0]

    mat1 = flattened_force_field[75].reshape(1,-1)
    mat1 = np.tile(mat1,(num_rows,1))
    mat1_tr = mat1.transpose()
    rob1_temp = (mat1 + mat1_tr) * 0.5
    rob1_temp = np.where(mat1 > 0.0, rob1_temp, 0.0)
    rob1_temp = np.where(mat1_tr > 0.0, rob1_temp, 0.0)

    mat1 = flattened_force_field[76].reshape(1,-1)
    mat1 = np.tile(mat1,(num_rows,1))
    mat1_tr = mat1.transpose()
    rob2_temp = (mat1 + mat1_tr) * 0.5
    rob2_temp = np.where(mat1 > 0.0, rob2_temp, 0.0)
    rob2_temp = np.where(mat1_tr > 0.0, rob2_temp, 0.0)

    mat1 = flattened_force_field[77].reshape(1,-1)
    mat1 = np.tile(mat1,(num_rows,1))
    mat1_tr = mat1.transpose()
    rob3_temp = (mat1 + mat1_tr) * 0.5
    rob3_temp = np.where(mat1 > 0.0, rob3_temp, 0.0)
    rob3_temp = np.where(mat1_tr > 0.0, rob3_temp, 0.0)
    #TODO: gradient of sqrt. at 0 is nan, use safe sqrt
    p1co_temp = safe_sqrt(4.0 * flattened_force_field[78].reshape(-1,1).dot(flattened_force_field[78].reshape(1,-1)))
    p2co_temp = safe_sqrt(flattened_force_field[79].reshape(-1,1).dot(flattened_force_field[79].reshape(1,-1)))
    p3co_temp = safe_sqrt(flattened_force_field[80].reshape(-1,1).dot(flattened_force_field[80].reshape(1,-1)))


    flattened_force_field[8] = np.where(flattened_non_dif_params[15] == 0, rob1_temp, flattened_force_field[81])
    flattened_force_field[9] = np.where(flattened_non_dif_params[16] == 0, rob2_temp, flattened_force_field[82])
    flattened_force_field[10] = np.where(flattened_non_dif_params[17] == 0, rob3_temp, flattened_force_field[83])

    flattened_force_field[3] = np.where(flattened_non_dif_params[12] == 0, p1co_temp, flattened_force_field[84] * 2.0)
    flattened_force_field[4] = np.where(flattened_non_dif_params[13] == 0, p2co_temp, flattened_force_field[85])
    flattened_force_field[5] = np.where(flattened_non_dif_params[14] == 0, p3co_temp, flattened_force_field[86])

    return flattened_force_field

def preprocess_force_field(flattened_force_field, flattened_non_dif_params):
    return symm_force_field(handle_offdiag(flattened_force_field,flattened_non_dif_params),flattened_non_dif_params)

def generate_random_value(low_limit, high_limit):
    diff = high_limit - low_limit
    return onp.random.random() * diff + low_limit

def random_init_force_field(flattened_force_field, params):
    for i,p in enumerate(params):
        ind = p[0]
        flattened_force_field[ind[0]] = jax.ops.index_update(flattened_force_field[ind[0]], ind[1], generate_random_value(p[2],p[3]))
        #flattened_force_field[ind][param_indices] = generate_random_value(p[2],p[3])

