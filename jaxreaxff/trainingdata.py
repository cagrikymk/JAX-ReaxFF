"""
Dataclasses for training data

Author: Mehmet Cagri Kaymak
"""
from jax_md import dataclasses, util
Array = util.Array

@dataclasses.dataclass
class ChargeItem(object):
    sys_ind: Array
    a_ind: Array
    target: Array
    weight: Array
    mask: Array

@dataclasses.dataclass
class EnergyItem(object):
    sys_inds: Array
    multip: Array
    target: Array
    weight: Array
    mask: Array
    
@dataclasses.dataclass
class DistItem(object):
    sys_ind: Array
    a1_ind: Array
    a2_ind: Array
    target: Array
    weight: Array
    mask: Array
    
@dataclasses.dataclass
class AngleItem(object):
    sys_ind: Array
    a1_ind: Array
    a2_ind: Array
    a3_ind: Array
    target: Array
    weight: Array
    mask: Array
    
@dataclasses.dataclass
class TorsionItem(object):
    sys_ind: Array
    a1_ind: Array
    a2_ind: Array
    a3_ind: Array
    a4_ind: Array
    target: Array
    weight: Array
    mask: Array
@dataclasses.dataclass
class ForceItem(object):
    sys_ind: Array
    a_ind: Array
    target: Array
    weight: Array
    mask: Array
@dataclasses.dataclass
class RMSGItem(object):
    sys_ind: Array
    target: Array
    weight: Array
    mask: Array
    
@dataclasses.dataclass
class TrainingData(object):
    charge_items: ChargeItem = None
    energy_items: EnergyItem = None
    dist_items: DistItem = None
    angle_items: AngleItem = None
    torsion_items: TorsionItem = None
    force_items: ForceItem = None
    RMSG_items: RMSGItem = None
