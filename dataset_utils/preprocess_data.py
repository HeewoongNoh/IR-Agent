import json
import numpy as np
import re
import pandas as pd
import torch.utils.data
from tqdm import tqdm
from sklearn.preprocessing import scale
from scipy import interpolate
from scipy.signal import *
import os
DIM_EA_RESULT = 5
SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])" # from molecular transformer github


class SMILESTokenizer(object):
    def __init__(self):
        self.regex = re.compile(SMI_REGEX_PATTERN)

    def tokenize(self, text):
        return [token for token in self.regex.findall(text)]
    

class IRData:
    def __init__(self, wavenumbers, absorbances, smiles, nist_id):
        self.wavenumbers = torch.tensor(wavenumbers, dtype=torch.float)
        self.absorbances = torch.tensor(absorbances, dtype=torch.float)
        self.smiles = smiles
        self.x  = self.absorbances.view(-1,1)
        self.ir_id = nist_id

class IRDataset(torch.utils.data.Dataset):
    def __init__(self, ir_data, max_len, syms=None):
        self.ir_data = ir_data
        self.x = [d.x for d in self.ir_data]
        self.smiles = [d.smiles for d in self.ir_data]
        self.syms = self.__get_syms(syms)
        self.max_len = max_len
        self.sym_to_idx = {s: i for i, s in enumerate(self.syms)}
        self.idx_to_sym = {i: s for i, s in enumerate(self.syms)}
        self.encodings = self.__set_mol_encodings()
        self.ir_id = [d.ir_id for d in self.ir_data]
        self.config = {
            'dim_spect': self.dim_spect,
            'len_spect': self.len_spect,
            'num_syms': self.num_syms,
            'len_smiles': self.len_smiles
        }

    @property
    def dim_spect(self):
        return self.x[0].shape[1]
    @property
    def len_spect(self):
        return self.x[0].shape[0]

    @property
    def num_syms(self):
        return len(self.syms)

    @property
    def len_smiles(self):
        return self.max_len

    def __len__(self):
        return len(self.ir_data)

    def __getitem__(self, idx):
        return self.x[idx], self.encodings[idx]
    
    def __get_syms(self, syms):
        if syms is None:
            tokenizer = SMILESTokenizer()
            _syms = set()
            for s in self.smiles:

                _syms.update(tokenizer.tokenize(s))
            _syms = ['[nop]', '[start]', '[end]'] + list(sorted(_syms))
        else:
            _syms = syms

        return _syms

    def __set_mol_encodings(self):

        tokenizer = SMILESTokenizer()
        reps = [[self.sym_to_idx[symbol] for symbol in tokenizer.tokenize(s)] for s in self.smiles]
        encodings = torch.zeros((len(reps), self.max_len), dtype=torch.long)

        for i in range(0, len(reps)):
            encodings[i, 0] = self.sym_to_idx['[start]']
            for j in range(0, len(reps[i])):
                encodings[i, j + 1] = reps[i][j]
            encodings[i, len(reps[i]) + 1] = self.sym_to_idx['[end]']

        return encodings


def transmittance_to_absorbance(transmittance):
    # 0~1
    transmittance_fraction = transmittance / 100 if np.max(transmittance) > 1 else transmittance
    transmittance_fraction = np.where(transmittance_fraction > 0, transmittance_fraction, 1e-10) # 1e-10 is added for conversion
    # convert absorbance
    absorbance = -np.log10(transmittance_fraction)
    return absorbance   

def load_dataset(dataset_type, max_len, wave_len, syms=None):

    dataset = []
    wave_len = wave_len
    if dataset_type == "nist":

        nist_metadata = pd.read_csv("")
        nist_directory = ""

        for idx, row in tqdm(enumerate(nist_metadata.itertuples(index=True, name = 'Pandas'))):

            smiles = row[6]
            nist_id = row[1] + ".json"
            nist_file_path = os.path.join(nist_directory, nist_id)

            with open(nist_file_path, 'r') as f:
                nist_data = json.load(f)
            
            x = np.array(nist_data['wavenumber'], dtype=float)
            y = np.array(nist_data['transmittance'], dtype=float)
            assert len(x) == len(y), "different values between x and y"

            absorbances = transmittance_to_absorbance(y)

            f_interpol = interpolate.interp1d(x, absorbances, kind='linear', fill_value='extrapolate')
            wavenumbers = np.arange(500, wave_len)

            absorbances = scale(f_interpol(wavenumbers))
            dataset.append(IRData(wavenumbers, absorbances, smiles, nist_id))
    if syms == None:

        return IRDataset(dataset, max_len)
    
    else:
        return IRDataset(dataset, max_len, syms)

