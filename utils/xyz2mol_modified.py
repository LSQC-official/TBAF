"""
Module for generating rdkit molobj/smiles/molecular graph from free atoms

Implementation by Jan H. Jensen, based on the paper

    Yeonjoon Kim and Woo Youn Kim
    "Universal Structure Conversion Method for Organic Molecules: From Atomic Connectivity
    to Three-Dimensional Geometry"
    Bull. Korean Chem. Soc. 2015, Vol. 36, 1769-1777
    DOI: 10.1002/bkcs.10334

"""

import copy
import itertools

from rdkit.Chem import rdmolops
from rdkit.Chem import rdchem
try:
    from rdkit.Chem import rdEHTTools #requires RDKit 2019.9.1 or later
except ImportError:
    rdEHTTools = None
    
from collections import defaultdict

import numpy as np
import networkx as nx

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
import sys

import random

global __ATOM_LIST__
__ATOM_LIST__ = \
    ['h',  'he',
     'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne',
     'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar',
     'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu',
     'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
     'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
     'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe',
     'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',
     'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt',
     'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
     'fr', 'ra', 'ac', 'th', 'pa', 'u',  'np', 'pu']


global atomic_valence
global atomic_valence_electrons

atomic_valence = defaultdict(list)
atomic_valence[1] = [1]
atomic_valence[5] = [3,4]
atomic_valence[6] = [4]
atomic_valence[7] = [3,4] # 3 for DNA
atomic_valence[8] = [2]
atomic_valence[9] = [1]
atomic_valence[14] = [4,5]
atomic_valence[15] = [5,3] #[5,4,3]
atomic_valence[16] = [6,2,3] #[6,4,2]
atomic_valence[17] = [1]
atomic_valence[32] = [4]
atomic_valence[35] = [1]
atomic_valence[53] = [1]

atomic_valence[3] = [1]
atomic_valence[11] = [1]
atomic_valence[19] = [1]
atomic_valence[37] = [1]
atomic_valence[55] = [1]

atomic_valence_electrons = {}
atomic_valence_electrons[1] = 1
atomic_valence_electrons[5] = 3
atomic_valence_electrons[6] = 4
atomic_valence_electrons[7] = 5
atomic_valence_electrons[8] = 6
atomic_valence_electrons[9] = 7
atomic_valence_electrons[14] = 4
atomic_valence_electrons[15] = 5
atomic_valence_electrons[16] = 6
atomic_valence_electrons[17] = 7
atomic_valence_electrons[32] = 4
atomic_valence_electrons[35] = 7
atomic_valence_electrons[53] = 7

atomic_valence_electrons[3] = 1
atomic_valence_electrons[11] = 1
atomic_valence_electrons[19] = 1
atomic_valence_electrons[37] = 1
atomic_valence_electrons[55] = 1

def most_common_valences(atoms, AC):
    common_valences = {
        1: [1],      # H
        5: [3],      # B
        6: [4],      # C
        7: [3],      # N
        8: [2],      # O
        9: [1],      # F
        14: [4],     # Si
        15: [5, 3],  # P
        16: [2, 4],  # S
        17: [1],     # Cl
        32: [4],     # Ge
        35: [1],     # Br
        53: [1]      # I
    }
    valences = []
    for atom in atoms:
        if atom in common_valences:
            valences.append(common_valences[atom])
        else:
            valences.append(atomic_valence.get(atom, [3]))
    return valences

def heuristic_valences(atoms, AC):
    valences = []
    AC_valence = list(AC.sum(axis=1))
    
    for i, atom in enumerate(atoms):
        if atom in [1, 5, 6, 8, 9, 14, 17, 32, 35, 53]:
            fixed_valences = {
                1: [1], 6: [4], 8: [2], 9: [1], 
                17: [1], 35: [1], 53: [1]
            }
            valences.append(fixed_valences.get(atom, [atomic_valence[atom][0]]))
        else:
            if atom == 7:
                predicted = predict_nitrogen_valence(atoms, AC, i, AC_valence[i])
                valences.append(predicted)
            elif atom == 15:
                predicted = predict_phosphorus_valence(atoms, AC, i, AC_valence[i])
                valences.append(predicted)
            elif atom == 16:
                predicted = predict_sulfur_valence(atoms, AC, i, AC_valence[i])
                valences.append(predicted)
            else:
                valences.append(atomic_valence.get(atom, [3]))
    
    return valences

def predict_nitrogen_valence(atoms, AC, atom_idx, current_valence):
    neighbors = [atoms[j] for j in np.nonzero(AC[atom_idx])[0]]
    
    if len(neighbors) == 4:
        return [4]
    elif len(neighbors) == 3:
        return [3]
    elif len(neighbors) == 2:
        for j in np.nonzero(AC[atom_idx])[0]:
            if atoms[j] == 6 and AC[atom_idx, j] == 1:
                carbon_neighbors = [atoms[k] for k in np.nonzero(AC[j])[0]]
                if 7 in carbon_neighbors:
                    return [1]
        return [3]
    else:
        return [3, 4]
    
def predict_phosphorus_valence(atoms, AC, atom_idx, current_valence):
    neighbors = [atoms[j] for j in np.nonzero(AC[atom_idx])[0]]
    O_count = len([n for n in neighbors if n == 8])
    
    if O_count >= 3:
        return [5]
    elif O_count == 1 or O_count == 2:
        return [5]
    else:
        return [3, 5]

def predict_sulfur_valence(atoms, AC, atom_idx, current_valence):
    neighbors = [atoms[j] for j in np.nonzero(AC[atom_idx])[0]]
    O_count = len([n for n in neighbors if n == 8])
    
    if O_count >= 3:
        return [6]
    elif O_count == 2:
        return [6]
    elif O_count == 1:
        return [4]
    else:
        return [2]

def limited_valence_search(AC, atoms, charge, valence_selector, 
                          max_combinations=100, allow_charged_fragments=True, 
                          use_graph=True):
    AC_valence = list(AC.sum(axis=1))
    
    valences_list_of_lists = valence_selector(atoms, AC)
    
    valences_list = limited_cartesian_product(valences_list_of_lists, max_combinations)
    
    best_BO = AC.copy()
    best_q_count = len(atoms)
    
    for count, valences in enumerate(valences_list):
        UA, DU_from_AC = get_UA(valences, AC_valence)
        
        if len(UA) == 0:
            check_bo = BO_is_OK(AC, AC, charge, DU_from_AC,
                              atomic_valence_electrons, atoms, valences,
                              allow_charged_fragments=allow_charged_fragments)
            if check_bo:
                return AC, atomic_valence_electrons
        else:
            UA_pairs_list = get_UA_pairs(UA, AC, use_graph=use_graph)
            for UA_pairs in UA_pairs_list:
                BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, use_graph=use_graph)
                status = BO_is_OK(BO, AC, charge, DU_from_AC,
                                atomic_valence_electrons, atoms, valences,
                                allow_charged_fragments=allow_charged_fragments)
                charge_ok, q_list = charge_is_OK(BO, AC, charge, DU_from_AC, 
                                               atomic_valence_electrons, atoms, valences,
                                               allow_charged_fragments=allow_charged_fragments)
                
                if status:
                    return BO, atomic_valence_electrons
                elif BO.sum() >= best_BO.sum() and len(q_list) <= best_q_count and charge_ok:
                    best_BO = BO.copy()
                    best_q_count = len(q_list)
    
    return best_BO, atomic_valence_electrons

def limited_cartesian_product(arrays, max_products):
    total_combinations = 1
    for arr in arrays:
        total_combinations *= len(arr)
    
    if total_combinations <= max_products:
        return itertools.product(*arrays)
    else:
        return heuristic_product_sampling(arrays, max_products)

def heuristic_product_sampling(arrays, max_samples):
    sampled = []
    
    base_combo = [arr[0] for arr in arrays]
    sampled.append(base_combo)
    
    for i in range(len(arrays)):
        if len(sampled) >= max_samples:
            break
        for j in range(1, min(3, len(arrays[i]))):
            if len(sampled) >= max_samples:
                break
            new_combo = base_combo.copy()
            new_combo[i] = arrays[i][j]
            sampled.append(new_combo)
    
    remaining = max_samples - len(sampled)
    if remaining > 0:
        for _ in range(remaining):
            random_combo = [random.choice(arr) for arr in arrays]
            sampled.append(random_combo)
    
    return sampled

NCIlist=['Li', 'Na', 'K', 'Rb', 'Cs']
def str_atom(atom):
    """
    convert integer atom to string atom
    """
    global __ATOM_LIST__
    atom = __ATOM_LIST__[atom - 1]
    return atom


def int_atom(atom):
    """
    convert str atom to integer atom
    """
    global __ATOM_LIST__
    #print(atom)
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1


def get_UA(maxValence_list, valence_list):
    """
    """
    UA = []
    DU = []
    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
        if not maxValence - valence > 0:
            continue
        UA.append(i)
        DU.append(maxValence - valence)
    return UA, DU


def get_BO(AC, UA, DU, valences, UA_pairs, use_graph=True):
    """
    """
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i, j in UA_pairs:
            BO[i, j] += 1
            BO[j, i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = get_UA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA, AC, use_graph=use_graph)[0]

    return BO


def valences_not_too_large(BO, valences):
    """
    """
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences, number_of_bonds_list):
        if number_of_bonds > valence:
            return False

    return True

def charge_is_OK(BO, AC, charge, DU, atomic_valence_electrons, atoms, valences,
                 allow_charged_fragments=True):
    # total charge
    Q = 0

    # charge fragment list
    q_list = []
    pt = Chem.GetPeriodicTable()
    if allow_charged_fragments:

        BO_valences = list(BO.sum(axis=1))
        for i, atom in enumerate(atoms):
            
            q = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1
    
            if q != 0:
                q_list.append(q)

    return (charge == Q),np.array(q_list)

def BO_is_OK(BO, AC, charge, DU, atomic_valence_electrons, atoms, valences,
    allow_charged_fragments=True):
    """
    Sanity of bond-orders

    args:
        BO -
        AC -
        charge -
        DU - 


    optional
        allow_charges_fragments - 


    returns:
        boolean - true of molecule is OK, false if not
    """

    if not valences_not_too_large(BO, valences):
        return False

    check_sum = (BO - AC).sum() == sum(DU)
    check_charge,_ = charge_is_OK(BO, AC, charge, DU, atomic_valence_electrons, atoms, valences,
                                allow_charged_fragments)

    if check_charge and check_sum: 
        return True

    return False


def get_atomic_charge(atom, atomic_valence_electrons, BO_valence):
    """
    """

    if atom == 1:
        charge = 1 - BO_valence
    elif atom == 5:
        charge = 3 - BO_valence
    elif atom == 14 and BO_valence == 5:
        charge = -1
    elif atom == 15 and BO_valence == 5:
        charge = 0
    elif atom == 16 and BO_valence == 6:
        charge = 0
    elif atom in [3,11,19,37,55]:
        charge = 1
    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge


def clean_charges(mol):
    """
    This hack should not be needed anymore, but is kept just in case

    """

    Chem.SanitizeMol(mol)
    #rxn_smarts = ['[N+:1]=[*:2]-[C-:3]>>[N+0:1]-[*:2]=[C-0:3]',
    #              '[N+:1]=[*:2]-[O-:3]>>[N+0:1]-[*:2]=[O-0:3]',
    #              '[N+:1]=[*:2]-[*:3]=[*:4]-[O-:5]>>[N+0:1]-[*:2]=[*:3]-[*:4]=[O-0:5]',
    #              '[#8:1]=[#6:2]([!-:6])[*:3]=[*:4][#6-:5]>>[*-:1][*:2]([*:6])=[*:3][*:4]=[*+0:5]',
    #              '[O:1]=[c:2][c-:3]>>[*-:1][*:2][*+0:3]',
    #              '[O:1]=[C:2][C-:3]>>[*-:1][*:2]=[*+0:3]']

    rxn_smarts = ['[#6,#7:1]1=[#6,#7:2][#6,#7:3]=[#6,#7:4][CX3-,NX3-:5][#6,#7:6]1=[#6,#7:7]>>'
                  '[#6,#7:1]1=[#6,#7:2][#6,#7:3]=[#6,#7:4][-0,-0:5]=[#6,#7:6]1[#6-,#7-:7]',
                  '[#6,#7:1]1=[#6,#7:2][#6,#7:3](=[#6,#7:4])[#6,#7:5]=[#6,#7:6][CX3-,NX3-:7]1>>'
                  '[#6,#7:1]1=[#6,#7:2][#6,#7:3]([#6-,#7-:4])=[#6,#7:5][#6,#7:6]=[-0,-0:7]1']

    fragments = Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=False)

    for i, fragment in enumerate(fragments):
        for smarts in rxn_smarts:
            patt = Chem.MolFromSmarts(smarts.split(">>")[0])
            while fragment.HasSubstructMatch(patt):
                rxn = AllChem.ReactionFromSmarts(smarts)
                ps = rxn.RunReactants((fragment,))
                fragment = ps[0][0]
                Chem.SanitizeMol(fragment)
        if i == 0:
            mol = fragment
        else:
            mol = Chem.CombineMols(mol, fragment)

    return mol


def BO2mol(mol, BO_matrix, atoms, atomic_valence_electrons,
           mol_charge, allow_charged_fragments=True,  use_atom_maps=False):
    """
    based on code written by Paolo Toscani

    From bond order, atoms, valence structure and total charge, generate an
    rdkit molecule.

    args:
        mol - rdkit molecule
        BO_matrix - bond order matrix of molecule
        atoms - list of integer atomic symbols
        atomic_valence_electrons -
        mol_charge - total charge of molecule

    optional:
        allow_charged_fragments - bool - allow charged fragments

    returns
        mol - updated rdkit molecule with bond connectivity

    """

    l = len(BO_matrix)
    l2 = len(atoms)
    BO_valences = list(BO_matrix.sum(axis=1))

    if (l != l2):
        raise RuntimeError('sizes of adjMat ({0:d}) and Atoms {1:d} differ'.format(l, l2))

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE
    }

    for i in range(l):
        for j in range(i + 1, l):
            bo = int(round(BO_matrix[i, j]))
            if (bo == 0):
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    if allow_charged_fragments:
        mol = set_atomic_charges(
            mol,
            atoms,
            atomic_valence_electrons,
            BO_valences,
            BO_matrix,
            mol_charge,
            use_atom_maps)
    else:
        mol = set_atomic_radicals(mol, atoms, atomic_valence_electrons, BO_valences,
                                                            use_atom_maps)

    return mol


def set_atomic_charges(mol, atoms, atomic_valence_electrons,
                       BO_valences, BO_matrix, mol_charge,
                       use_atom_maps):
    """
    """
    q = 0
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        if use_atom_maps:
            a.SetAtomMapNum(i+1)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
            if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                q += 1
                charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                q += 2
                charge = 1

        if (abs(charge) > 0):
            a.SetFormalCharge(int(charge))

    #mol = clean_charges(mol)

    return mol


def set_atomic_radicals(mol, atoms, atomic_valence_electrons, BO_valences,
                                                use_atom_maps):
    """

    The number of radical electrons = absolute atomic charge

    """
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        if use_atom_maps:
            a.SetAtomMapNum(i+1)
        charge = get_atomic_charge(
            atom,
            atomic_valence_electrons[atom],
            BO_valences[i])

        if (abs(charge) > 0):
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol


def get_bonds(UA, AC):
    """

    """
    bonds = []

    for k, i in enumerate(UA):
        for j in UA[k + 1:]:
            if AC[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))

    return bonds


def get_UA_pairs(UA, AC, use_graph=True):
    """

    """

    bonds = get_bonds(UA, AC)

    if len(bonds) == 0:
        return [()]

    if use_graph:
        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        return UA_pairs

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]

        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs

def AC2BO(AC, atoms, charge, allow_charged_fragments=True, use_graph=True):
    """
    implemenation of algorithm shown in Figure 2

    UA: unsaturated atoms

    DU: degree of unsaturation (u matrix in Figure)

    best_BO: Bcurr in Figure

    """
    global atomic_valence
    global atomic_valence_electrons

    if len(atoms) <= 8:
        return AC2BO_original(AC, atoms, charge, allow_charged_fragments, use_graph)
    
    return hierarchical_valence_search(AC, atoms, charge, allow_charged_fragments, use_graph)

def hierarchical_valence_search(AC, atoms, charge, allow_charged_fragments=True, use_graph=True):
    
    # Level 1: Common valence search (fast)
    result = limited_valence_search(AC, atoms, charge, 
                                  valence_selector=most_common_valences,
                                  max_combinations=50,
                                  allow_charged_fragments=allow_charged_fragments,
                                  use_graph=use_graph)
    if result[0] is not None and not np.array_equal(result[0], AC):
        return result
    
    # Level 2: Heuristic valence search (medium)
    result = limited_valence_search(AC, atoms, charge,
                                  valence_selector=heuristic_valences,
                                  max_combinations=200,
                                  allow_charged_fragments=allow_charged_fragments,
                                  use_graph=use_graph)
    if result[0] is not None and not np.array_equal(result[0], AC):
        return result
    
    # Level 3: Limited complete search (thorough)
    return AC2BO_original_limited(AC, atoms, charge, allow_charged_fragments, use_graph)

def AC2BO_original(AC, atoms, charge, allow_charged_fragments=True, use_graph=True):
    """

    implemenation of algorithm shown in Figure 2

    UA: unsaturated atoms

    DU: degree of unsaturation (u matrix in Figure)

    best_BO: Bcurr in Figure

    """

    global atomic_valence
    global atomic_valence_electrons

    # make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    AC_valence = list(AC.sum(axis=1))

    for i,(atomicNum,valence) in enumerate(zip(atoms,AC_valence)):
        # valence can't be smaller than number of neighbourgs
        if atomic_valence[atomicNum]:
            possible_valence = [x for x in atomic_valence[atomicNum] if x >= valence]
            if not possible_valence:
                print('Valence of atom',i,'is',valence,'which bigger than allowed max',max(atomic_valence[atomicNum]),'. Stopping')
                sys.exit()
            valences_list_of_lists.append(possible_valence)
    # convert [[4],[2,1]] to [[4,2],[4,1]]
    valences_list = itertools.product(*valences_list_of_lists)

    best_BO = AC.copy()
    numQAtoms = len(AC)
    count = 0
    for valences in valences_list:
        count+=1
        UA, DU_from_AC = get_UA(valences, AC_valence)

        check_len = (len(UA) == 0)
        if check_len:
            check_bo = BO_is_OK(AC, AC, charge, DU_from_AC,
                atomic_valence_electrons, atoms, valences,
                allow_charged_fragments=allow_charged_fragments)
        else:
            check_bo = None

        if check_len and check_bo:
            return AC, atomic_valence_electrons

        UA_pairs_list = get_UA_pairs(UA, AC, use_graph=use_graph)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, use_graph=use_graph)
            status = BO_is_OK(BO, AC, charge, DU_from_AC,
                        atomic_valence_electrons, atoms, valences,
                        allow_charged_fragments=allow_charged_fragments)
            charge_OK,q_list = charge_is_OK(BO, AC, charge, DU_from_AC, atomic_valence_electrons, atoms, valences,
                                     allow_charged_fragments=allow_charged_fragments)

            if status:
                return BO, atomic_valence_electrons
            elif BO.sum() >= best_BO.sum() and len(q_list) <= numQAtoms and charge_OK:
                best_BO = BO.copy()
                numQAtoms = len(q_list)
    # print(count)
    return best_BO, atomic_valence_electrons

def AC2BO_original_limited(AC, atoms, charge, allow_charged_fragments=True, use_graph=True, max_attempts=5000):
    valences_list_of_lists = []
    AC_valence = list(AC.sum(axis=1))
    
    for i, (atomicNum, valence) in enumerate(zip(atoms, AC_valence)):
        if atomic_valence[atomicNum]:
            possible_valence = [x for x in atomic_valence[atomicNum] if x >= valence]
            if not possible_valence:
                print('Valence of atom',i,'is',valence,'which bigger than allowed max',max(atomic_valence[atomicNum]),'. Stopping')
                sys.exit()
            valences_list_of_lists.append(possible_valence)
    
    valences_list = itertools.product(*valences_list_of_lists)
    best_BO = AC.copy()
    numQAtoms = len(AC)
    count = 0
    
    for valences in valences_list:
        count += 1
        if count > max_attempts:
            # print(f"Reached maximum attempts ({max_attempts}), returning best solution found")
            break
            
        UA, DU_from_AC = get_UA(valences, AC_valence)
        check_len = (len(UA) == 0)
        if check_len:
            check_bo = BO_is_OK(AC, AC, charge, DU_from_AC,
                atomic_valence_electrons, atoms, valences,
                allow_charged_fragments=allow_charged_fragments)
        else:
            check_bo = None

        if check_len and check_bo:
            return AC, atomic_valence_electrons

        UA_pairs_list = get_UA_pairs(UA, AC, use_graph=use_graph)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, use_graph=use_graph)
            status = BO_is_OK(BO, AC, charge, DU_from_AC,
                        atomic_valence_electrons, atoms, valences,
                        allow_charged_fragments=allow_charged_fragments)
            charge_OK,q_list = charge_is_OK(BO, AC, charge, DU_from_AC, atomic_valence_electrons, atoms, valences,
                                     allow_charged_fragments=allow_charged_fragments)

            if status:
                return BO, atomic_valence_electrons
            elif BO.sum() >= best_BO.sum() and len(q_list) <= numQAtoms and charge_OK:
                best_BO = BO.copy()
                numQAtoms = len(q_list)
    return best_BO, atomic_valence_electrons

def AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, 
           use_graph=True, use_atom_maps=False):
    """
    """
    # convert AC matrix to bond order (BO) matrix
    BO, atomic_valence_electrons = AC2BO(
        AC,
        atoms,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph)

    # add BO connectivity and charge info to mol object
    mol = BO2mol(
        mol,
        BO,
        atoms,
        atomic_valence_electrons,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_atom_maps=use_atom_maps)

    # If charge is not correct don't return mol
    if Chem.GetFormalCharge(mol) != charge:
        return []
    
    # BO2mol returns an arbitrary resonance form. Let's make the rest
    mols = rdchem.ResonanceMolSupplier(mol, Chem.UNCONSTRAINED_CATIONS, Chem.UNCONSTRAINED_ANIONS)
    mols = [mol for mol in mols]

    return mols


def get_proto_mol(atoms):
    """
    """
    mol = Chem.MolFromSmarts("[#" + str(atoms[0]) + "]")
    rwMol = Chem.RWMol(mol)
    for i in range(1, len(atoms)):
        a = Chem.Atom(atoms[i])
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def read_xyz_file(filename, look_for_charge=True):
    """
    """
    atomic_symbols = []
    xyz_coordinates = []
    charge = 0
    title = ""
    
    if filename.endswith('.xyz'):
        with open(filename, "r") as file:
            for line_number, line in enumerate(file):
                if line_number == 0:
                    num_atoms = int(line)
                elif line_number == 1:
                    title = line
                    if "charge=" in line:
                        charge = int(line.split("=")[1])
                elif line_number in range(2,2+num_atoms):
                    atomic_symbol, x, y, z = line.split()
                    atomic_symbols.append(atomic_symbol)
                    xyz_coordinates.append([float(x), float(y), float(z)])
        atoms = [int_atom(atom) for atom in atomic_symbols]
        
        
    elif filename.endswith('.log') or filename.endswith('.out'):
        with open(filename,'r') as fr:
            lines = fr.readlines()
        coord_start_index_list = []
        for i,line in enumerate(lines):
            if 'Charge' in line and 'Multiplicity' in line:
                charge = int(str(line.split()[2]))
            if 'NAtoms=' in line:
                atom_num = eval(line.split()[1]) 
            if 'Input orientation' in line or 'Standard orientation' in line:
                coord_start_index_list.append(i+5)
        coord_string = lines[coord_start_index_list[-1]:coord_start_index_list[-1]+atom_num]   
        for coord_line in coord_string:
            _,atomic_symbol,_,x, y, z = coord_line.split()
            atomic_symbols.append(atomic_symbol)
            xyz_coordinates.append([float(x), float(y), float(z)])
        atoms = [int(atom) for atom in atomic_symbols]

    return atoms, charge, xyz_coordinates


def xyz2AC(atoms, xyz, charge, use_huckel=False):
    """

    atoms and coordinates to atom connectivity (AC)

    args:
        atoms - int atom types
        xyz - coordinates
        charge - molecule charge

    optional:
        use_huckel - Use Huckel method for atom connecitivty

    returns
        ac - atom connectivity matrix
        mol - rdkit molecule

    """

    if use_huckel:
        return xyz2AC_huckel(atoms, xyz, charge)
    else:
        return xyz2AC_vdW(atoms, xyz)


def xyz2AC_vdW(atoms, xyz):

    # Get mol template
    mol = get_proto_mol(atoms)

    # Set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)

    AC = get_AC(mol)

    return AC, mol

# CSD radius, from Dalton Trans., 2008, 2832–2838
CSDRadius = [0.00,
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76,
    1.70, 1.60, 1.53, 1.39, 1.50, 1.42, 1.38, 1.19, 1.32, 1.22,
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75,
    1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39,
    1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01,
    1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06,
    2.00, 1.96, 1.90, 1.87, 1.80, 1.69]

def get_AC(mol, covalent_factor=1.15):
    """

    Generate adjacent matrix from atoms and coordinates.

    AC is a (num_atoms, num_atoms) matrix with 1 being covalent bond and 0 is not


    covalent_factor - 1.3 is an arbitrary factor

    args:
        mol - rdkit molobj with 3D conformer

    optional
        covalent_factor - increase covalent bond length threshold with facto

    returns:
        AC - adjacent matrix

    """

    # Calculate distance matrix
    dMat = Chem.Get3DDistanceMatrix(mol)

    pt = Chem.GetPeriodicTable()
    num_atoms = mol.GetNumAtoms()
    AC = np.zeros((num_atoms, num_atoms), dtype=int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        if a_i.GetSymbol() not in NCIlist:
            #Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum()) * covalent_factor
            Rcov_i = CSDRadius[a_i.GetAtomicNum()] * covalent_factor
            for j in range(i + 1, num_atoms):
                a_j = mol.GetAtomWithIdx(j)
                if a_j.GetSymbol() not in NCIlist:
                    #Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum()) * covalent_factor
                    Rcov_j = CSDRadius[a_j.GetAtomicNum()] * covalent_factor
                    if dMat[i, j] <= Rcov_i + Rcov_j:
                        AC[i, j] = 1
                        AC[j, i] = 1
    return AC


def xyz2AC_huckel(atomicNumList, xyz, charge):
    """

    args
        atomicNumList - atom type list
        xyz - coordinates
        charge - molecule charge

    returns
        ac - atom connectivity
        mol - rdkit molecule

    """
    mol = get_proto_mol(atomicNumList)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i,(xyz[i][0],xyz[i][1],xyz[i][2]))
    mol.AddConformer(conf)

    num_atoms = len(atomicNumList)
    AC = np.zeros((num_atoms,num_atoms)).astype(int)

    mol_huckel = Chem.Mol(mol)
    mol_huckel.GetAtomWithIdx(0).SetFormalCharge(charge) #mol charge arbitrarily added to 1st atom    

    passed,result = rdEHTTools.RunMol(mol_huckel)
    opop = result.GetReducedOverlapPopulationMatrix()
    tri = np.zeros((num_atoms, num_atoms))
    tri[np.tril(np.ones((num_atoms, num_atoms), dtype=bool))] = opop #lower triangular to square matrix
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            pair_pop = abs(tri[j,i])   
            if pair_pop >= 0.15: #arbitry cutoff for bond. May need adjustment
                AC[i,j] = 1
                AC[j,i] = 1

    return AC, mol


def chiral_stereo_check(mol):
    """
    Find and embed chiral information into the model based on the coordinates

    args:
        mol - rdkit molecule, with embeded conformer

    """
    Chem.SanitizeMol(mol)
    Chem.DetectBondStereochemistry(mol, -1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)

    return


def xyz2mol(atoms, coordinates, charge=0, allow_charged_fragments=True,
            use_graph=True, use_huckel=False, embed_chiral=True,
            use_atom_maps=False):
    """
    Generate a rdkit molobj from atoms, coordinates and a total_charge.

    args:
        atoms - list of atom types (int)
        coordinates - 3xN Cartesian coordinates
        charge - total charge of the system (default: 0)

    optional:
        allow_charged_fragments - alternatively radicals are made
        use_graph - use graph (networkx)
        use_huckel - Use Huckel method for atom connectivity prediction
        embed_chiral - embed chiral information to the molecule

    returns:
        mols - list of rdkit molobjects

    """

    # Get atom connectivity (AC) matrix, list of atomic numbers, molecular charge,
    # and mol object with no connectivity information
    AC, mol = xyz2AC(atoms, coordinates, charge, use_huckel=use_huckel)

    # Convert AC to bond order matrix and add connectivity and charge info to
    # mol object
    new_mols = AC2mol(mol, AC, atoms, charge,
                     allow_charged_fragments=allow_charged_fragments,
                     use_graph=use_graph,
                     use_atom_maps=use_atom_maps)

    # Check for stereocenters and chiral centers
    if embed_chiral:
        for new_mol in new_mols:
            chiral_stereo_check(new_mol)

    return new_mols


def main():


    return


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(usage='%(prog)s [options] molecule.xyz')
    parser.add_argument('structure', metavar='structure', type=str)
    parser.add_argument('-s', '--sdf',
        action="store_true",
        help="Dump sdf file")
    parser.add_argument('--ignore-chiral',
        action="store_true",
        help="Ignore chiral centers")
    parser.add_argument('--no-charged-fragments',
        action="store_true",
        help="Allow radicals to be made")
    parser.add_argument('--no-graph',
        action="store_true",
        help="Run xyz2mol without networkx dependencies")
    
    parser.add_argument('--use_atom_maps',
        action="store_true",
        help="use atom maps")
    
    # huckel uses extended Huckel bond orders to locate bonds (requires RDKit 2019.9.1 or later)
    # otherwise van der Waals radii are used
    parser.add_argument('--use-huckel',
        action="store_true",
        help="Use Huckel method for atom connectivity")
    parser.add_argument('-o', '--output-format',
        action="store",
        type=str,
        help="Output format [smiles,sdf] (default=sdf)")
    parser.add_argument('-c', '--charge',
        action="store",
        metavar="int",
        type=int,
        help="Total charge of the system")

    args = parser.parse_args()

    # read xyz file
    filename = args.structure

    # allow for charged fragments, alternatively radicals are made
    charged_fragments = not args.no_charged_fragments

    # quick is faster for large systems but requires networkx
    # if you don't want to install networkx set quick=False and
    # uncomment 'import networkx as nx' at the top of the file
    quick = not args.no_graph

    # chiral comment
    embed_chiral = not args.ignore_chiral

    # read atoms and coordinates. Try to find the charge
    atoms, charge, xyz_coordinates = read_xyz_file(filename)

    # huckel uses extended Huckel bond orders to locate bonds (requires RDKit 2019.9.1 or later)
    # otherwise van der Waals radii are used
    use_huckel = args.use_huckel
    
    use_atom_maps =args.use_atom_maps
    
    # if explicit charge from args, set it
    if args.charge is not None:
        charge = int(args.charge)

    # Get the molobjs
    mols = xyz2mol(atoms, xyz_coordinates,
        charge=charge,
        use_graph=quick,
        allow_charged_fragments=charged_fragments,
        embed_chiral=embed_chiral,
        use_huckel=use_huckel,
        use_atom_maps=use_atom_maps)

    # Print output
    for mol in mols:
        if args.output_format == "sdf":
            txt = Chem.MolToMolBlock(mol)
            print(txt)

        else:
            # Canonical hack
            isomeric_smiles = not args.ignore_chiral
            #print(use_atom_maps)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles,allHsExplicit=use_atom_maps)
            if not use_atom_maps:
                m = Chem.MolFromSmiles(smiles)
                smiles = Chem.MolToSmiles(m, isomericSmiles=isomeric_smiles, allHsExplicit=use_atom_maps)
            print(smiles)

