from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
import os

def validity_checker(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        return False
    
def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # If valid smiles
            return Chem.MolToSmiles(mol, canonical=True)
        else:  # otherwise
            return smiles
    except Exception as e:
        return smiles
    
def get_scaffold(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return scaffold_smiles

def make_functional_group(functional_groups, smiles):

    mol = Chem.MolFromSmiles(smiles)

    found_groups  = []
    smarts_list = []
    for name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            found_groups.append(name)
            smarts_list.append(smarts)

    return found_groups, smarts_list

def count_functional_groups(functional_groups, smiles):
  
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"Error": "Invalid SMILES"}  

    functional_group_counts = {
        name: len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
        for name, smarts in functional_groups.items()
    }

    return functional_group_counts

def calculate_tanimoto(smiles_a, smiles_b):

    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)

        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)

        similarity = TanimotoSimilarity(fp_a, fp_b)
        
        return similarity
    
    except:
        return None

def read_table(wavenumber, ir_spectrum_table):
    for (lower, upper), group in ir_spectrum_table.items():
        if lower >= wavenumber >= upper: 
            return (lower, upper), group
    return None, None 



def interpret_table(peaks, ir_spectrum_table):
    group_dicts = dict()
    for wn in peaks:
        range, group = read_table(wn, ir_spectrum_table)
        if group is not None:
            group_dicts[range] = group
        else:
            continue 
      
    table_text = ""
    if len(group_dicts) == 0:

        table_text += "No assigned functional group"

    else:
        for wave_ranges, assigned_fg in group_dicts.items():
            table_text += "Peaks observed between " + str(wave_ranges) + "cm⁻¹ " + "are typically associated with "+ assigned_fg + ". " + "\n"

    return table_text


ir_spectrum_table_final = {
    # 4000-3000 cm⁻¹
    (3700, 3584): "alcohol (O-H)",
    (3550, 3200): "alcohol (O-H)",
    (3500, 3400): "primary amine (N-H)",
    (3400, 3300): "aliphatic primary amine (N-H)",
    (3330, 3250): "aliphatic primary amine (N-H)",
    (3350, 3310): "secondary amine (N-H)",
    (3100, 2900): "carboxylic acid (O-H)", #(3300,2500) --> usually centered on 3000
    (3200, 2700): "alcohol (O-H) ",
    (3000, 2800): "amine salt (N-H)",
    
    # 3000-2500 cm⁻¹
    (3333, 3267): "alkyne (C-H)",
    (3100, 3000): "alkene (C-H)",
    (3080, 2840): "alkane (C-H)",
    (2830, 2695): "aldehyde (C-H)",
    (2600, 2550): "thiol (S-H)",
    
    # 2400-2000 cm⁻¹
    (2354, 2344): "carbon dioxide (O=C=O)",
    (2285, 2250): "isocyanate (N=C=O)",
    (2260, 2222): "nitrile (C≡N)",
    (2260, 2190): "disubstituted alkyne (C≡C)",
    (2175, 2140): "thiocyanate (S-C=N)",
    (2160, 2120): "azide (N=N=N)",
    (2155, 2145): "ketene (C=C=O)",
    (2145, 2120): "carbodiimide (N=C=N)",
    (2140, 2100): "monosubstituted alkyne (C≡C)",
    (2140, 1990): "isothiocyanate (N=C=S)",
    (2000, 1900): "allene (C=C=C)",
    (2005, 1995): "ketenimine (C=C=N)",
    (2000, 1650): "aromatic compound (C-H)",
    
    # 1870-1540 cm⁻¹
    (1818, 1750): "anhydride (C=O)",
    (1815, 1785): "acid halide (C=O)",
    (1800, 1770): "conjugated acid halide (C=O)",
    (1780, 1770): "conjugated anhydride (C=O)",
    (1725, 1715): "conjugated anhydride (C=O)", 
    (1770, 1780): "vinyl/phenyl ester (C=O)",
    (1765, 1755): "carboxylic acid (C=O)",
    (1750, 1735): "δ-lactone (C=O)",
    (1750, 1735): "esters (C=O)",
    (1750, 1740): "cyclopentanone (C=O)",
    (1740, 1720): "aldehyde (C=O)",
    (1730, 1715): "α,β-unsaturated ester (C=O)",
    (1725, 1705): "aliphatic ketone (C=O)",
    (1720, 1706): "carboxylic acid (C=O)",
    (1710, 1680): "conjugated acid (C=O)",
    (1710, 1685): "conjugated aldehyde (C=O)",
    (1695, 1685): "primary amide (C=O)",
    (1690, 1640): "imine/oxime (C=N)",
    (1685, 1666): "conjugated ketone (C=O)",
    (1685, 1675): "secondary amide (C=O)",
    (1685, 1675): "tertiary amide (C=O)",
    (1655, 1645): "δ-lactam (C=O)", 
    # 1678-1610 cm⁻¹
    (1678, 1668): "trans-disubstituted alkene (C=C)",
    (1675, 1665): "trisubstituted alkene (C=C)",
    (1675, 1665): "tetrasubstituted alkene (C=C)",
    (1662, 1626): "cis-disubstituted alkene (C=C)",
    (1658, 1600): "alkene (vinylidene) (C=C)",
    (1650, 1600): "conjugated alkene (C=C)",
    (1650, 1580): "amine (N-H)",
    (1650, 1566): "cyclic alkene (C=C)",
    (1648, 1638): "monosubstituted alkene (C=C)",
    (1620, 1610): "α,β-unsaturated ketone (C=C)",

    # 1600-1300 cm⁻¹
    (1550, 1500): "nitro compound (N-O)",
    (1372, 1290): "nitro compound (N-O)",
    (1470, 1460): "alkane (methylene group) (C-H)",
    (1455, 1445): "alkane (methyl group) (C-H)",
    (1380, 1370): "alkane (methyl group) (C-H)",
    (1390, 1380): "aldehyde (C-H)",
    (1385, 1380): "alkane (gem dimethyl) (C-H)",
    (1370, 1365): "alkane (gem dimethyl) (C-H)",

    # 1400-1000 cm⁻¹
    (1440, 1395): "carboxylic acid (O-H)",
    (1420, 1330): "alcohol (O-H)",
    (1415, 1380): "sulfate (S=O)",
    (1200, 1185): "sulfate (S=O)",
    (1410, 1380): "sulfonyl chloride (S=O)",
    (1204, 1177): "sulfonyl chloride (S=O)",
    (1200, 1000): "fluoro compound (C-F)",
    (1390, 1310): "phenol (O-H)",
    (1372, 1335): "sulfonate (S=O)",
    (1195, 1168): "sulfonate (S=O)",
    (1370, 1335): "sulfonamide (S=O)",  
    (1170, 1155): "sulfonamide (S=O)",

    # 1350-1050 cm⁻¹
    (1350, 1342): "sulfonic acid (S=O)",
    (1165, 1150): "sulfonic acid (S=O)",
    (1350, 1300): "sulfone (S=O)",
    (1160, 1120): "sulfone (S=O)",
    (1342, 1266): "aromatic amine (C-N)",
    (1310, 1250): "aromatic ester (C-O)",
    (1275, 1200): "alkyl aryl ether (C-O)",
    (1075, 1020): "alkyl aryl ether (C-O)",
    (1250, 1195): "phosphorus oxide (P-O)",
    (1300, 1250): "phosphorus oxide (P-O)",
    (1250, 1020): "amine (C-N)",
    (1225, 1200): "vinyl ether (C-O)",
    (1075, 1020): "vinyl ether (C-O)",
    (1210, 1163): "ester (C-O)",
    (1205, 1124): "tertiary alcohol (C-O)",
    (1150, 1085): "aliphatic ether (C-O)",
    (1124, 1087): "secondary alcohol (C-O)",
    (1085, 1050): "primary alcohol (C-O)",
    (1070, 1030): "sulfoxide (S=O)",
    (1050, 1040): "anhydride (CO-O-CO)",

    # 1000-650 cm⁻¹
    (995, 985): "monosubstituted alkene (C=C)",
    (915, 905): "monosubstituted alkene (C=C)",
    (980, 960): "trans-disubstituted alkene (C=C)",
    (895, 885): "alkene (vinylidene) (C=C)",
    (760, 540): "halo compound (C-Cl)",
    (840, 790): "trisubstituted alkene (C=C)",
    (730, 665): "cis-disubstituted alkene (C=C)",
    (690, 515): "halo compound (C-Br)", 
    (600, 500): "halo compound (C-I)",
    
    # 750 ± 20 to 700 ± 20 cm⁻¹
    (750, 700): "monosubstituted benzene derivative",
    (710, 690): "monosubstituted benzene derivative"
}