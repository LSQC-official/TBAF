import subprocess
import time, signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Time out with fragmentation!")

def run_with_timeout(timeout_seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)
                return result
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator

@run_with_timeout(120)
def XYZ2MolModified(xyz,to_form='smiles',charge=0,use_atom_maps=False,
                    xyz2mol=None,**kwargs):
    import os
    if xyz2mol is None:
        lsroot = os.environ.get('lsroot')
        if not lsroot:
            raise ValueError("Environment variable 'lsroot' is not set")
        xyz2mol = os.path.join(lsroot, 'bin', 'utils', 'xyz2mol_modified.py')

    commands = fr'python {xyz2mol} {xyz}'
    if to_form == 'sdf':
        commands = commands+' -o sdf'
    if charge != 0:
        commands = commands+f' -c {charge}'
    if use_atom_maps:
        commands = commands+' --use_atom_maps'
    res = subprocess.Popen(
                           commands, 
                           shell=True, 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.STDOUT)
    result = res.stdout.read().decode('utf-8').strip()
    res.stdout.close()
    return result

def xyz2coord(filename):
    atomic_symbols = []
    xyz_coordinates = []
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
    return atomic_symbols,xyz_coordinates
        


