#!/usr/bin/env python
import re
from rdkit import Chem
from utils.test import XYZ2MolModified as x2m
import sys
import numpy as np
import math
from scipy.optimize import linear_sum_assignment

params = Chem.SmilesParserParams()
params.removeHs = True
params1 = Chem.SmilesParserParams()
params1.removeHs = False

def log_warning(msg):
    print(f"WARNING: {msg}")

def handle_critical_error(error_msg, txt_file=None):
    print(f"{error_msg}")
    
    if txt_file:
        try:
            with open(txt_file, 'a') as f:
                f.write(error_msg + "\n")
                f.write("Program terminated due to critical error.\n")
        except Exception as e:
            print(f"ERROR: Failed to write to {txt_file}: {str(e)}")
    
    sys.exit(1)


def read_xyz(filename, txt_file=None):
    try:
        xyz_coordinates = []
        
        with open(filename, "r") as file:
            for line_number, line in enumerate(file):
                if line_number == 0:
                    num_atoms = int(line)
                elif line_number in range(2,2+num_atoms):
                    _, x, y, z = line.split()
                    xyz_coordinates.append([float(x), float(y), float(z)])

        return xyz_coordinates
    except FileNotFoundError:
        error_msg = f"XYZ file '{filename}' not found"
        handle_critical_error(error_msg, txt_file)
    except Exception as e:
        error_msg = f"Error reading XYZ file {filename}: {str(e)}"
        handle_critical_error(error_msg, txt_file)

def dis_cal(cation_mol, anion_mol, xyz):
    dis = float('inf')
    for atom1 in cation_mol.GetAtoms():
        index1 = int(atom1.GetProp('molAtomMapNumber'))
        coord1 = [xyz[index1 - 1][0], xyz[index1 - 1][1], xyz[index1 - 1][2]]
        for atom2 in anion_mol.GetAtoms():
            index2 = int(atom2.GetProp('molAtomMapNumber'))
            coord2 = [xyz[index2 - 1][0], xyz[index2 - 1][1], xyz[index2 - 1][2]]
            dis = min(dis, math.sqrt(sum((a-b)**2 for a,b in zip(coord1,coord2))))
    return dis

def getIonicLiquid(input_smiles, input_file):
    try:
        frags = []
        cation_mols = []
        anion_mols = []
        neutral_mols = []
        smiles = input_smiles.split('.')
        xyz = read_xyz(input_file)
        
        positive_charges, negative_charges = extract_charges_from_smiles(input_smiles)
        
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi, params1)
            fragment_charge = 0
            fragment_atom_charges = {}
            
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom_map = int(atom.GetProp('molAtomMapNumber'))
                    if atom_map in positive_charges:
                        charge_val = positive_charges[atom_map]
                        fragment_charge += charge_val
                        fragment_atom_charges[atom_map] = charge_val
                    elif atom_map in negative_charges:
                        charge_val = negative_charges[atom_map]
                        fragment_charge += charge_val
                        fragment_atom_charges[atom_map] = charge_val
            
            if fragment_charge > 0:
                cation_mols.append((mol, fragment_charge, fragment_atom_charges))
            elif fragment_charge < 0:
                anion_mols.append((mol, fragment_charge, fragment_atom_charges))
            else:
                neutral_mols.append(mol)
                
        npairs = min(len(cation_mols), len(anion_mols))
        
        if npairs == 0:
            for mol, charge, atom_charges in cation_mols:
                atom_map = []
                for atom in mol.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                frags.append(atom_map)
            
            for mol, charge, atom_charges in anion_mols:
                atom_map = []
                for atom in mol.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                frags.append(atom_map)
                
            for mol in neutral_mols:
                atom_map = []
                for atom in mol.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                frags.append(atom_map)
            
            return frags
        
        dis = np.zeros((len(cation_mols), len(anion_mols)), dtype=float)
        for i in range(len(cation_mols)):
            for j in range(len(anion_mols)):
                dis[i][j] = dis_cal(cation_mols[i][0], anion_mols[j][0], xyz)
        
        cation_indices, anion_indices = linear_sum_assignment(dis)
        
        used_cations = set()
        used_anions = set()
        
        for i in range(len(cation_indices)):
            cation_idx = cation_indices[i]
            anion_idx = anion_indices[i]
            
            if cation_idx < len(cation_mols) and anion_idx < len(anion_mols):
                cation_mol, cation_charge, cation_atom_charges = cation_mols[cation_idx]
                anion_mol, anion_charge, anion_atom_charges = anion_mols[anion_idx]
                
                atom_map = []
                for atom in cation_mol.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                for atom in anion_mol.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                
                frags.append(atom_map)
                used_cations.add(cation_idx)
                used_anions.add(anion_idx)
        

        for i in range(len(cation_mols)):
            if i not in used_cations:
                cation_mol, cation_charge, cation_atom_charges = cation_mols[i]
                atom_map = []
                for atom in cation_mol.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                frags.append(atom_map)
        
        for i in range(len(anion_mols)):
            if i not in used_anions:
                anion_mol, anion_charge, anion_atom_charges = anion_mols[i]
                atom_map = []
                for atom in anion_mol.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                frags.append(atom_map)
        
        for mol in neutral_mols:
            atom_map = []
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom_map.append(int(atom.GetProp('molAtomMapNumber')))
            frags.append(atom_map)

        return frags
    
    except Exception as e:
        error_msg = f"Error in getIonicLiquid: {str(e)}"
        handle_critical_error(error_msg, txt_file)


def extract_charges_from_smiles(smiles):
    positive_charges = {}
    negative_charges = {}
    
    positive_pattern = r'\[([^]]*?)\+(\d*):(\d+)\]'
    matches = re.findall(positive_pattern, smiles)
    for match in matches:
        atom_desc, charge_str, atom_map = match
        charge = int(charge_str) if charge_str else 1
        positive_charges[int(atom_map)] = charge
    
    negative_pattern = r'\[([^]]*?)\-(\d*):(\d+)\]'
    matches = re.findall(negative_pattern, smiles)
    for match in matches:
        atom_desc, charge_str, atom_map = match
        charge = -int(charge_str) if charge_str else -1
        negative_charges[int(atom_map)] = charge
    
    simple_negative_pattern = r'\[([^]]*?)\-(\d*)\]'
    matches = re.findall(simple_negative_pattern, smiles)
    
    return positive_charges, negative_charges


if __name__ == '__main__':
    fname = sys.argv[1][:-4]
    charge = sys.argv[2]
    input_file = fname + '.xyz'
    txt_file = fname + '.txt'

    try:
        input_smiles = x2m(input_file, use_atom_maps = True, charge = int(charge))
        if not input_smiles:
            handle_critical_error("Xyz2mol failed to generate SMILES", txt_file)
        frags = []
        frags = getIonicLiquid(input_smiles, input_file)
        
        positive_charges, negative_charges = extract_charges_from_smiles(input_smiles)
        with open(fname + '.frg', 'w') as f:  
            for idx in range(len(frags)):
                total_charge = 0
                fragment_charges = {}
                for atom_map in frags[idx]:
                    if atom_map in positive_charges:
                        charge_val = positive_charges[atom_map]
                        fragment_charges[atom_map] = charge_val
                        total_charge += charge_val
                    elif atom_map in negative_charges:
                        charge_val = negative_charges[atom_map]
                        fragment_charges[atom_map] = charge_val
                        total_charge += charge_val
                if total_charge > 0:
                    f.write(str(idx+1) + ' 1 (' + str(frags[idx])[1:-1] + ') +' + str(total_charge) + '\n')
                elif total_charge < 0:
                    f.write(str(idx+1) + ' 1 (' + str(frags[idx])[1:-1] + ') ' + str(total_charge) + '\n')
                else:
                    f.write(str(idx+1) + ' 1 (' + str(frags[idx])[1:-1] + ') 0 \n')
    except Exception as e:
        error_msg = str(e)
        error_msg += '\nPlease check the initial structure. If the structure is correct, you can try to calculate through customized fragmentation method.'
        handle_critical_error(f"Error: {error_msg}", txt_file)
