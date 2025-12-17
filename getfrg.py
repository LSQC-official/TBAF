#!/usr/bin/env python
import re
from rdkit import Chem
from rdkit.Chem import AllChem
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

def load_template_library(filename='gebf_template_library.md', txt_file=None):
    try:
        import os
        lsroot = os.environ.get('lsroot')
        if not lsroot:
            error_msg = "Environment variable 'lsroot' is not set"
            handle_critical_error(error_msg, txt_file)
        
        filename = os.path.join(lsroot, 'bin', filename)
        cyclic_smiles = []
        acyclic_smiles = []
        
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        sections = content.split('\n')
        current_section = None
        
        for line in sections:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('cyclic:'):
                current_section = 'cyclic'
                smiles_part = line.split(':', 1)[1].strip()
            elif line.startswith('acyclic:'):
                current_section = 'acyclic'
                smiles_part = line.split(':', 1)[1].strip()
            else:
                smiles_part = line
                
            if current_section == 'cyclic' and smiles_part:
                cyclic_smiles.extend([s.strip() for s in smiles_part.split(',') if s.strip()])
            elif current_section == 'acyclic' and smiles_part:
                acyclic_smiles.extend([s.strip() for s in smiles_part.split(',') if s.strip()])
        
        return cyclic_smiles, acyclic_smiles
        
    except FileNotFoundError:
        error_msg = f"Template library file '{filename}' not found"
        handle_critical_error(error_msg, txt_file)
    except Exception as e:
        error_msg = f"Error loading template library {filename}: {str(e)}"
        handle_critical_error(error_msg, txt_file)

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

def frag(input_mol,frag_mol): 
    matches = input_mol.GetSubstructMatches(frag_mol,uniquify=False)
    return list(matches[0]),AllChem.ReplaceCore(input_mol,frag_mol,matches[0])

def getRings(input_mol):
    frags = []
    mols = []
    num0 = 0
    num1 = 0
    submols = input_mol.GetSubstructMatches(Chem.MolFromSmarts('[!R;D{2-}][R]'))
    bonds_id = [input_mol.GetBondBetweenAtoms(x,y).GetIdx() for x,y in submols]
    pattern = re.compile(r'(\[(\d+)\*\])')
    if bonds_id !=[]:
        frag_mols = Chem.FragmentOnBonds(input_mol, bonds_id)
        smiles = Chem.MolToSmiles(frag_mols).split('.')
        ring_main = {}
        ring_left = {}
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            heavy_atom_map = []
            ring_num = len(Chem.GetSymmSSSR(mol))
            if ring_num > 0:
                if (len(mol.GetAtoms())) > 20:
                    if ring_num > 1:
                        frags_small, mols_small = getSmallRings(mol)
                        frags.extend(frags_small)
                        mols.extend(mols_small)
                        continue
                    else:
                        mols.append(mol)
                        continue
                num0 +=1
                cuts = [str(int(y)-1) for x, y in re.findall(pattern,smi)]
                for atom in mol.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        heavy_atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                ring_main[num0] = heavy_atom_map
                ring_main[str(num0) + 'cuts'] = cuts
            else:
                if mol.GetNumHeavyAtoms() <= 3:
                    cuts = [y for x, y in re.findall(pattern,smi)]
                    num1 += 1
                    for atom in mol.GetAtoms():
                        if atom.HasProp('molAtomMapNumber'):
                            heavy_atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                    ring_left[num1] = heavy_atom_map
                    ring_left[str(num1) + 'cuts'] = cuts
                else:
                    mols.append(mol)
        if len(ring_left) != 0:
            for i in range(num1):
                cuts1 = ring_left[str(i + 1) + 'cuts']
                flag = False
                for cut in cuts1:
                    for j in range(num0):
                        cuts = ring_main[str(j + 1) + 'cuts'] 
                        if cut in cuts:
                            heavy_atom_map = ring_main[j + 1]
                            heavy_atom_map.extend(ring_left[i + 1])
                            ring_main[j + 1] = heavy_atom_map
                            flag = True
                            break
                    if flag:
                        break
                if not flag:
                    frags.append(ring_left[i + 1])
        for i in range(num0):
            frags.append(ring_main[i + 1])

    elif bonds_id == [] and len(Chem.GetSymmSSSR(input_mol)) > 0: # only one large ring without side chains
        frags, mols = getSmallRings(input_mol)
    else:
        mols.append(input_mol)
    return frags, mols

def getSmallRings(input_mol):
    cyclic_smiles, _ = load_template_library()
    bonds_id = []
    frags = []
    mols = []
    submols = []
    for smile in cyclic_smiles:
        submols_new = list(input_mol.GetSubstructMatches(Chem.MolFromSmiles(smile)))
        if submols_new == []:
            continue
        flag = False
        for i in range(len(submols_new)):
            for j in range(len(submols_new[i])):
                for k in range(len(submols)):
                    if submols_new[i][j] in submols[k]:
                        flag = True
                        continue
            if flag == True:
                continue
            submols.append(submols_new[i])
        
        for i in range(len(submols)):
            for j in range(len(submols[i])):
                startAtom = submols[i][j]
                neighbors = input_mol.GetAtomWithIdx(startAtom).GetNeighbors()
                for k in range(len(neighbors)):
                    nextAtom = neighbors[k].GetIdx()
                    neighbors2 = neighbors[k].GetNeighbors()
                    checksum = 0
                    for n in range(len(neighbors2)):
                        if input_mol.GetBondBetweenAtoms(nextAtom,neighbors2[n].GetIdx()).GetIdx() in bonds_id:
                            checksum += 1
                    if len(neighbors2) == 1 or len(neighbors2) - checksum == 1: # for side chains with only one heavy atom or one left bond, skip
                        continue
                    if nextAtom not in submols[i]:
                        bond_id = input_mol.GetBondBetweenAtoms(startAtom,nextAtom).GetIdx()
                        if bond_id not in bonds_id:
                            bonds_id.append(bond_id)
    if bonds_id != []:
        frag_mols = Chem.FragmentOnBonds(input_mol, bonds_id)
        smiles = Chem.MolToSmiles(frag_mols).split('.')
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            heavy_atom_map = []
            ring_num = len(Chem.GetSymmSSSR(mol))
            if ring_num > 0:
                for atom in mol.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        heavy_atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                frags.append(heavy_atom_map)
            else:
                mols.append(mol)
    
    return frags, mols


def getFrags(input_mol,frag_mol,txt_file=None):
    try:
        if input_mol is None:
            error_msg = "Input molecule is None"
            handle_critical_error(error_msg, txt_file)
        if frag_mol is None or len(frag_mol) == 0:
            error_msg = "Fragment templates are invalid or empty"
            handle_critical_error(error_msg, txt_file)
        
        frags = []
        rings, mol_need_to_be_cut = getRings(input_mol)
        frags.extend(rings)
        input_mol_cached = input_mol
        while True:
            input_mol = False
            for mol in mol_need_to_be_cut:
                for idx in range(len(frag_mol)):
                    if mol.HasSubstructMatch(frag_mol[idx]):
                        input_mol = mol
                        mol_need_to_be_cut.pop(mol_need_to_be_cut.index(mol))
                        heavy_atom_idx, r = frag(input_mol,frag_mol[idx])
                        heavy_atom_map = [int(input_mol.GetAtomWithIdx(i).GetProp('molAtomMapNumber')) for i in heavy_atom_idx]
                        side_mols = Chem.GetMolFrags(r,asMols=True)
                        for side_mol in side_mols:
                            if side_mol.GetNumHeavyAtoms() > 4:
                                side_mol_modified = Chem.RWMol(side_mol)
                                mol_need_to_be_cut.append(side_mol_modified.GetMol())
                            else:
                                for atom in side_mol.GetAtoms():
                                    if atom.HasProp('molAtomMapNumber'):
                                        heavy_atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                        frags.append(heavy_atom_map)
                        break
            
            if not input_mol:
                for mol in mol_need_to_be_cut:
                    heavy_atom_map = []
                    for atom in mol.GetAtoms():
                        if atom.HasProp('molAtomMapNumber'):
                            heavy_atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                    if heavy_atom_map != []:
                        frags.append(heavy_atom_map)
                break
        
        check_atom_coverage(input_mol_cached, frags)
        
        return frags

    except Exception as e:
        error_msg = f"Error in getFrags: {str(e)}"
        handle_critical_error(error_msg, txt_file)

def check_atom_coverage(original_mol, frags, txt_file=None):
    if original_mol is None:
        return
    
    all_heavy_atoms = set()
    for atom in original_mol.GetAtoms():
        if atom.GetAtomicNum() != 1 and atom.HasProp('molAtomMapNumber'):
            all_heavy_atoms.add(int(atom.GetProp('molAtomMapNumber')))
    
    covered_atoms = set()
    for frag in frags:
        covered_atoms.update(frag)
    
    missing_atoms = all_heavy_atoms - covered_atoms
    duplicate_atoms = set()
    atom_count = {}
    for frag in frags:
        for atom in frag:
            atom_count[atom] = atom_count.get(atom, 0) + 1
            if atom_count[atom] > 1:
                duplicate_atoms.add(atom)
    
    if missing_atoms:
        error_msg = "There are several atoms not included in frags."
        handle_critical_error(error_msg, txt_file)
    
    if duplicate_atoms:
        error_msg = "There are duplicate_atoms in frags."
        handle_critical_error(error_msg, txt_file)
    
    return len(missing_atoms) == 0 and len(duplicate_atoms) == 0

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
        mols_smiles = input_smiles.split('.')
        frags = []
        for smi in mols_smiles:
            mol = Chem.MolFromSmiles(smi, params)
            if mol.GetNumHeavyAtoms() > 10:
                frags_mol = []
                _, acyclic_smiles = load_template_library()
                for i in range(len(acyclic_smiles)):
                    frags_mol.append(Chem.MolFromSmiles(acyclic_smiles[i]))
                frags.extend(getFrags(mol, frags_mol))
            else:
                atom_map = []
                for atom in mol.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        atom_map.append(int(atom.GetProp('molAtomMapNumber')))
                frags.append(atom_map)
        
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
