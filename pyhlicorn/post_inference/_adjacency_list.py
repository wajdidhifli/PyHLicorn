import pandas as pd
import numpy as np

def clean_key(key):
    return str(key).strip("[]' ,").strip()

def parse_list_from_string(value):
    if isinstance(value, (list, np.ndarray)):
        unique_items = set()
        cleaned_items = []
        for item in value:
            clean_item = clean_key(str(item))
            if clean_item and clean_item not in unique_items:
                unique_items.add(clean_item)
                cleaned_items.append(clean_item)
        return cleaned_items

    if pd.isna(value) or (isinstance(value, str) and value in ['nan', 'NaN', '[]', '']):
        return []

    return []

def _adjacency_list(self):
    bygene = {}
    bytf = {}

    for idx, row in self.iterrows():
        target = row['Target']
        coact = parse_list_from_string(row['coact'])
        corep = parse_list_from_string(row['corep'])

        for tf in coact:
            if tf not in bytf:
                bytf[tf] = {'act': set(), 'rep': set()}  
            bytf[tf]['act'].add(target)

        for tf in corep:
            if tf not in bytf:
                bytf[tf] = {'act': set(), 'rep': set()}
            bytf[tf]['rep'].add(target)

        if target not in bygene:
            bygene[target] = {'act': set(), 'rep': set()}
        bygene[target]['act'].update(coact)
        bygene[target]['rep'].update(corep)

    
    # self.adjlist = {
    #     'bygene': {gene: {'act': list(data['act']), 'rep': list(data['rep'])} for gene, data in bygene.items()},
    #     'bytf': {tf: {'act': list(data['act']), 'rep': list(data['rep'])} for tf, data in bytf.items()}
    # }
    # return self
    
    bygene = {gene: {'act': list(data['act']), 'rep': list(data['rep'])} for gene, data in bygene.items()}
    bytf = {tf: {'act': list(data['act']), 'rep': list(data['rep'])} for tf, data in bytf.items()}
    return bygene, bytf


def get_regulators(self, target_gene):
    if not hasattr(self, 'adjlist') or 'bygene' not in self.adjlist:
        self.adjacency_list()

    bygene = self.adjlist['bygene']
    
    if target_gene not in bygene:
        return {'act': [], 'rep': []}
    
    return bygene[target_gene]

def get_targets(self, regulator):
    if not hasattr(self, 'adjlist') or 'bytf' not in self.adjlist:
        self.adjacency_list()

    bytf = self.adjlist['bytf']
    
    if regulator not in bytf:
        return {'act': [], 'rep': []}
    
    return bytf[regulator]



