import dgl
import networkx as nx
from collections import defaultdict
from rdkit import Chem
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import pandas as pd


def preprocess_smiles(smiles_list, d=None):
    atom_dict = defaultdict(lambda: len(atom_dict))
    if d is not None: atom_dict.update(d)
    gs = []
    for smiles in tqdm(smiles_list, 'smiles2graph'):
        mol = Chem.MolFromSmiles(smiles)
        g = nx.Graph()
        for atom in mol.GetAtoms():
            g.add_node(
                atom.GetIdx(),
                AtomElement=atom_dict[atom.GetSymbol()],
                # valence = atom.GetTotalValence(),
                # charge = atom.GetFormalCharge(),
                # hybridization = atom.GetHybridization().real,
                # aromatic = int(atom.GetIsAromatic()),
                AtomicNum=atom.GetAtomicNum(),
                ChiralTag=atom.GetChiralTag().real,
                Degree=atom.GetDegree(),
                ExplicitValence=atom.GetExplicitValence(),
                FormalCharge=atom.GetFormalCharge(),
                Hybridization=atom.GetHybridization().real,
                ImplicitValence=atom.GetImplicitValence(),
                IsAromatic=int(atom.GetIsAromatic()),
                Isotope=atom.GetIsotope(),
                Mass=atom.GetMass(),
                NoImplicit=int(atom.GetNoImplicit()),
                NumExplicitHs=atom.GetNumExplicitHs(),
                NumImplicitHs=atom.GetNumImplicitHs(),
                NumRadicalElectrons=atom.GetNumRadicalElectrons(),
                TotalDegree=atom.GetTotalDegree(),
                TotalNumHs=atom.GetTotalNumHs(),
                IsInRing=int(atom.IsInRing()),
                # IsInRingSize=int(atom.IsInRingSize()),
            )
        for bond in mol.GetBonds():
            g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                       bond_type=int(bond.GetBondTypeAsDouble()))  # str(label)
        dgl_g = dgl.DGLGraph()
        dgl_g.from_networkx(g, node_attrs=['AtomElement', 'AtomicNum', 'ChiralTag', 'Degree', 'ExplicitValence',
                                           'FormalCharge', 'Hybridization', 'ImplicitValence', 'IsAromatic', 'Isotope',
                                           'Mass','NoImplicit', 'NumExplicitHs', 'NumImplicitHs', 'NumRadicalElectrons',
                                           'TotalDegree', 'TotalNumHs', 'IsInRing', ],
                                           # 'IsInRingSize'],
                            edge_attrs=['bond_type']
                            )
        gs.append(dgl_g)
    return gs, dict(atom_dict)


from descproteins import *
from pubscripts import *
class SmiProDataset(Dataset):
    def __init__(self, path, atom_dict=None):
        self.data = pd.read_csv(path)

        self.descproteins_methods = ['AAC', 'APAAC', 'CKSAAGP', 'CTDC', 'CTDD', 'CTDT', 'CTriad', 'DDE',
                        'DPC', 'GAAC', 'GDPC', 'GTPC', 'KSCTriad',
                        # 'CKSAAP', 'PAAC', 'QSOrder','SOCNumber', 'TPC', 'Geary', 'Moran', 'NMBroto' # for too much time consuming
                    ]
        self.descproteins = None
        # self.proteins = self.data['protein'].values.tolist()

        self.graphs = None
        self.atom_dict = atom_dict

        self.labels = self.data['label'].values.tolist()

    def __getitem__(self, index):
        return [[self.graphs[index], self.descproteins.iloc[index, :].values], self.labels[index]]

    def __len__(self):
        return len(self.labels)  # of how many examples(images?) you have

    def preprocess(self):
        fastas = []
        for i in self.data.index:
            row = self.data.iloc[i, :]
            record = [row['ID'], row['protein'].replace('X', ''), row['label'], row['smi']]
            fastas.append(record)
        self.descproteins = self.get_multi_encodings(fastas)

        self.graphs, self.atom_dict = preprocess_smiles(self.data['smi'].values.tolist(), self.atom_dict)

    def get_multi_encodings(self, fastas):
        methods = self.descproteins_methods
        userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
        myOrder = userDefinedOrder
        kw = {'order': myOrder, 'type': 'Protein'}

        cat_encodings = pd.DataFrame()
        for method in tqdm(methods, 'Descproteins encodings '):
            print(method, 'encoding', end=' ')
            cmd = method + '.' + method + '(fastas, **kw)'
            encodings = eval(cmd)

            encodings = pd.DataFrame(encodings)
            columns = encodings.iloc[0, :][2:]
            index = encodings.iloc[:, 0][1:]
            encodings.drop(index=[0], inplace=True)
            encodings.drop(columns=[0, 1], inplace=True)   # 0 id 1 label


            encodings.index = index.values
            encodings.columns = columns.values
            # encodings.iloc[:,:].astype('float')
            # print(method, 'result shape: ', encodings.shape)
            cat_encodings = pd.concat([cat_encodings, encodings], axis=1)
        return cat_encodings.astype('float')




if __name__ == "__main__":
    DATASET = 'refined'  # core refined

    smiles_list = ['C1CCCCC1', 'CCC(=O)O', 'CCCCc1ccccc1']
    gs, atom_dict = preprocess_smiles(smiles_list)
    smiles_list = ['C1CCCCC1', 'CCC(=O)O', 'CCCCc1ccccc1', 'S(=O)(=O)c1ccc(S(=O)(=O)NCc2cccs2)s1']
    gs, atom_dict = preprocess_smiles(smiles_list, atom_dict)
    print(gs)
    print(atom_dict)
