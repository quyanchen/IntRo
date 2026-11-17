import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

DATASET_RELATIONS = {
    'Deng': 65,
    'Ryu': 86,
}


def find_dataset_dir(data_root, dataset_name):
    """Locate dataset directory, supporting both clean and legacy directory names."""
    root = Path(data_root)
    for candidate in [dataset_name, f'{dataset_name}sdataset']:
        target = root / candidate
        if target.is_dir():
            return target
    return root / dataset_name


def load_data(dataset, data_root):
    """Load drug list and split triples (train, val, test) for a dataset."""
    if dataset not in DATASET_RELATIONS:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: {list(DATASET_RELATIONS.keys())}")

    relations = DATASET_RELATIONS[dataset]
    dataset_dir = find_dataset_dir(data_root, dataset)
    drug_list_path = dataset_dir / 'drug_list.csv'
    if not drug_list_path.exists():
        raise FileNotFoundError(f"Missing drug list at {drug_list_path}")

    with drug_list_path.open(encoding='utf-8', newline='') as stream:
        molecules = list(csv.reader(stream))

    drug_ids = [row[0] for row in molecules]
    if len(set(drug_ids)) != len(drug_ids):
        raise ValueError('Duplicate drug IDs detected in drug_list.csv')

    drug_index = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}

    split_filenames = {
        'train': 'ddi_training1.csv',
        'val': 'ddi_validation1.csv',
        'test': 'ddi_test1.csv',
    }
    for split_dir in (dataset_dir / 'normal', dataset_dir):
        if all((split_dir / name).is_file() for name in split_filenames.values()):
            break
    else:
        expected = ', '.join(split_filenames.values())
        raise FileNotFoundError(
            f'Expected a complete set of splits in {dataset_dir / "normal"} or {dataset_dir}: {expected}'
        )

    splits = {}
    for split_name, filename in split_filenames.items():
        path = split_dir / filename
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))

        triples = np.array(
            [(drug_index[row['d1']], drug_index[row['d2']], int(row['type'])) for row in rows],
            dtype=np.int64,
        )
        if len(triples) == 0:
            raise ValueError(f"{dataset} {split_name} split is empty")
        if triples[:, 2].min() < 0 or triples[:, 2].max() >= relations:
            raise ValueError(f"{dataset} {split_name}: relation label out of range [0, {relations - 1}]")

        splits[split_name] = triples

    undirected_splits = {
        name: {(min(left, right), max(left, right), relation) for left, right, relation in triples}
        for name, triples in splits.items()
    }
    for (left_name, left), (right_name, right) in combinations(undirected_splits.items(), 2):
        if not left.isdisjoint(right):
            raise ValueError(f'{dataset}: {left_name} and {right_name} share undirected triples')

    return {
        'ids': drug_ids,
        'molecules': molecules,
        'relations': relations,
        'train': splits['train'],
        'val': splits['val'],
        'test': splits['test'],
    }


def load_features(data, feature_file, ids_file=None):
    """Load pre-trained molecular representations and align them to the dataset drug order."""
    feature_path = Path(feature_file)
    id_path = Path(ids_file) if ids_file else feature_path.with_name('drug_ids.npy')

    features = np.load(feature_path, allow_pickle=False)
    feature_id_array = np.load(id_path, allow_pickle=False)
    if features.ndim != 2 or features.shape[1] == 0 or feature_id_array.ndim != 1:
        raise ValueError('Expected a feature matrix and a one-dimensional drug ID array')
    feature_ids = feature_id_array.astype(str).tolist()

    if len(feature_ids) != len(features) or len(set(feature_ids)) != len(feature_ids):
        raise ValueError('Feature length and ID count mismatch, or duplicate feature IDs')

    feature_index = {drug_id: idx for idx, drug_id in enumerate(feature_ids)}
    missing_ids = [drug_id for drug_id in data['ids'] if drug_id not in feature_index]
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} drugs have no matching pre-trained features")

    row_order = [feature_index[drug_id] for drug_id in data['ids']]
    aligned = features[row_order].astype(np.float32)

    if not np.isfinite(aligned).all():
        raise ValueError('Non-finite values encountered in molecular features')

    return torch.from_numpy(aligned)


def build_training_graph(triples):
    """Construct an undirected graph (edge_index, edge_type) from training triples.
    
    Forward and reverse edges are interleaved consecutively:
    [e0_fwd, e0_rev, e1_fwd, e1_rev, ...] so that edge-type corruption maintains symmetry.
    """
    pairs = torch.as_tensor(triples[:, :2], dtype=torch.long)
    edge_index = torch.stack([pairs, pairs.flip(1)], dim=1).reshape(-1, 2).t().contiguous()
    edge_type = torch.as_tensor(triples[:, 2], dtype=torch.long).repeat_interleave(2)
    return edge_index, edge_type


def compute_relation_profile(edge_index, edge_type, num_nodes, num_relations):
    """Compute the relation participation profile (N x R) for all entities.
    
    Accumulates incidence counts across the bidirectional edges of the training graph.
    """
    table = torch.zeros(num_nodes, num_relations, dtype=torch.float32)
    rel_one_hot = torch.nn.functional.one_hot(edge_type, num_relations).float()
    table.index_add_(0, edge_index[0], rel_one_hot)
    table.index_add_(0, edge_index[1], rel_one_hot)
    return table


def normalize_features(features):
    """Scale by the signed row sum, mapping zero-sum rows to zero."""
    row_sums = features.sum(dim=1, keepdim=True)
    zero_sum = row_sums == 0
    safe_sums = row_sums.masked_fill(zero_sum, 1.0)
    scale = safe_sums.reciprocal().masked_fill(zero_sum, 0.0)
    return features * scale
