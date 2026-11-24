import argparse
import copy
import json
from pathlib import Path
import random

import numpy as np
from rdkit import Chem
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, subgraph as pyg_subgraph

from data import load_data

NUM_ATOM_TYPES = 119
NUM_BOND_TYPES = 4
ATOM_FEATURE_DIM = 300


def smiles_to_graph(smiles, drug_id=''):
    """Convert a SMILES string to a PyG molecular graph with atom and bond features."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES string for drug '{drug_id}': {smiles}")

    chiralities = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bond_directions = [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ]

    atom_features = []
    for atom in molecule.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        chirality = atom.GetChiralTag()
        if not 1 <= atomic_num < NUM_ATOM_TYPES or chirality not in chiralities:
            raise ValueError(f"Unsupported atom features for drug '{drug_id}' at atom {atom.GetIdx()}")
        atom_features.append([atomic_num, chiralities.index(chirality)])

    x = torch.tensor(atom_features, dtype=torch.long)

    edges = []
    bond_features = []
    for bond in molecule.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        bond_type, direction = bond.GetBondType(), bond.GetBondDir()
        if bond_type not in bond_types or direction not in bond_directions:
            raise ValueError(f"Unsupported bond features for drug '{drug_id}' at bond {bond.GetIdx()}")
        attrs = [bond_types.index(bond_type), bond_directions.index(direction)]
        edges.extend(((src, dst), (dst, src)))
        bond_features.extend((attrs, attrs))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(bond_features, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def replace_node_edge(data, mask_ratio):
    """Corrupt a random subset of atoms and their incident bonds (NERep)."""
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f'mask_ratio must be in [0, 1], got {mask_ratio}')
    data = copy.deepcopy(data)
    num_nodes = data.x.size(0)
    selected_nodes = []
    if num_nodes and mask_ratio > 0.0:
        num_masked_nodes = max(1, int(round(num_nodes * mask_ratio)))
        selected_nodes = random.sample(range(num_nodes), min(num_nodes, num_masked_nodes))

    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    selected_indices = torch.tensor(selected_nodes, dtype=torch.long, device=data.x.device)
    node_mask[selected_indices] = True
    data.masked_atom_mask = node_mask
    data.mask_node_label = data.x[node_mask].clone()

    for idx in selected_nodes:
        data.x[idx] = data.x.new_tensor([random.randint(0, NUM_ATOM_TYPES - 2), 0])

    src, dst = data.edge_index
    edge_mask = node_mask[src] | node_mask[dst]
    data.masked_edge_mask = edge_mask
    data.mask_edge_label = data.edge_attr[edge_mask].clone()

    replacement_by_bond = {}
    for edge_idx in edge_mask.nonzero(as_tuple=True)[0].tolist():
        u, v = int(src[edge_idx]), int(dst[edge_idx])
        bond_key = (min(u, v), max(u, v))
        if bond_key not in replacement_by_bond:
            replacement_by_bond[bond_key] = random.randint(0, NUM_BOND_TYPES - 1)
        data.edge_attr[edge_idx] = data.edge_attr.new_tensor([replacement_by_bond[bond_key], 0])

    return data


def sample_subgraph(data, subgraph_ratio):
    """Grow a molecular subgraph from a random atom and visit other components as needed."""
    data = copy.deepcopy(data)
    num_nodes = data.x.size(0)
    if num_nodes == 0 or subgraph_ratio <= 0.0:
        return data

    keep_num = max(1, min(num_nodes, int(round(num_nodes * (1.0 - subgraph_ratio)))))
    if keep_num == num_nodes:
        return data

    edge_index_np = data.edge_index.detach().cpu().numpy()
    selected = []
    selected_set = set()
    frontier = set()

    def add_node(node):
        selected.append(node)
        selected_set.add(node)
        neighbors = edge_index_np[1][edge_index_np[0] == node].tolist()
        frontier.update(n for n in neighbors if n not in selected_set)
        frontier.difference_update(selected_set)

    add_node(int(np.random.randint(num_nodes)))
    while len(selected) < keep_num:
        if frontier:
            next_node = int(np.random.choice(sorted(frontier)))
            frontier.discard(next_node)
        else:
            remaining = sorted(set(range(num_nodes)) - selected_set)
            next_node = int(np.random.choice(remaining))
        add_node(next_node)

    subset = torch.tensor(selected, dtype=torch.long, device=data.edge_index.device)
    new_edge_index, new_edge_attr = pyg_subgraph(
        subset,
        data.edge_index,
        data.edge_attr,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )
    data.x = data.x[subset]
    data.edge_index = new_edge_index.long()
    data.edge_attr = new_edge_attr
    return data


class BondGCN(MessagePassing):
    """Bond-aware GCN layer with self-loops."""

    def __init__(self, hidden_dim=300):
        super().__init__(aggr='add', node_dim=0)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.bond_emb1 = nn.Embedding(6, hidden_dim)
        self.bond_emb2 = nn.Embedding(3, hidden_dim)
        nn.init.xavier_uniform_(self.bond_emb1.weight)
        nn.init.xavier_uniform_(self.bond_emb2.weight)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(x))
        self_loop_attr = edge_attr.new_zeros(len(x), 2)
        self_loop_attr[:, 0] = 4  # self-loop bond type
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_feat = self.bond_emb1(edge_attr[:, 0]) + self.bond_emb2(edge_attr[:, 1])

        src, dst = edge_index
        edge_weight = x.new_ones(edge_index.size(1))
        deg = x.new_zeros(len(x)).index_add_(0, src, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[src] * edge_weight * deg_inv_sqrt[dst]

        return self.propagate(edge_index, x=self.linear(x), edge_attr=edge_feat, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class MolecularGCN(nn.Module):

    def __init__(self, hidden_dim=300, num_layers=5):
        super().__init__()
        self.atom_emb1 = nn.Embedding(120, hidden_dim)
        self.atom_emb2 = nn.Embedding(3, hidden_dim)
        nn.init.xavier_uniform_(self.atom_emb1.weight)
        nn.init.xavier_uniform_(self.atom_emb2.weight)

        self.convs = nn.ModuleList([BondGCN(hidden_dim) for _ in range(num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_attr):
        h = self.atom_emb1(x[:, 0]) + self.atom_emb2(x[:, 1])
        for idx, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h = bn(conv(h, edge_index, edge_attr))
            if idx < len(self.convs) - 1:
                h = F.relu(h)
        return h


class MolecularEncoder(nn.Module):

    def __init__(self, hidden_dim=300):
        super().__init__()
        self.gnn = MolecularGCN(hidden_dim=hidden_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch):
        node_repr = self.gnn(batch.x, batch.edge_index, batch.edge_attr)
        graph_repr = global_mean_pool(node_repr, batch.batch)
        return node_repr, self.projection_head(graph_repr)


class MolecularPretrainer(nn.Module):
    """Molecular pretraining model with MADR and MSC objectives."""

    def __init__(self, config):
        super().__init__()
        self.encoder = MolecularEncoder(hidden_dim=ATOM_FEATURE_DIM)
        self.atom_detector = nn.Linear(ATOM_FEATURE_DIM, 1)
        self.bond_detector = nn.Linear(ATOM_FEATURE_DIM, 1)
        self.atom_recovery = nn.Linear(ATOM_FEATURE_DIM, NUM_ATOM_TYPES)

        self.detection_weight = config.get('detection_weight', 0.3)
        self.consistency_weight = config.get('consistency_weight', 0.1)
        self.temperature = config.get('temperature', 0.1)

    def compute_loss(self, corrupted_batch, subgraph_batch):
        nodes, corrupted_repr = self.encoder(corrupted_batch)
        _, subgraph_repr = self.encoder(subgraph_batch)

        # MADR: atom detection and recovery
        atom_mask = corrupted_batch.masked_atom_mask
        orig_atom_types = corrupted_batch.mask_node_label[:, 0]
        atom_pred_unchanged = self.atom_detector(nodes).sigmoid()

        atom_targets = nodes.new_ones(len(nodes))
        atom_targets[atom_mask] = (orig_atom_types == corrupted_batch.x[atom_mask, 0]).float()
        detection_loss = F.binary_cross_entropy(atom_pred_unchanged, atom_targets[:, None])

        recovery_loss = nodes.new_zeros(())
        if atom_mask.any():
            atom_logits = self.atom_recovery(nodes[atom_mask])
            recovery_prob = atom_logits.softmax(dim=-1).gather(1, orig_atom_types[:, None])
            copy_prob = atom_pred_unchanged[atom_mask]
            correct_prob = atom_targets[atom_mask, None] * copy_prob + (1.0 - copy_prob) * recovery_prob
            recovery_loss = -(correct_prob + 1e-9).log().mean()

        # MADR: bond detection
        bond_mask = corrupted_batch.masked_edge_mask
        if bond_mask.any():
            src, dst = corrupted_batch.edge_index
            bond_pred_unchanged = self.bond_detector(nodes[src] + nodes[dst]).sigmoid()
            bond_targets = nodes.new_ones(len(src))
            orig_bonds = corrupted_batch.mask_edge_label[:, 0]
            sampled_bonds = corrupted_batch.edge_attr[bond_mask, 0]
            bond_targets[bond_mask] = (orig_bonds == sampled_bonds).float()
            detection_loss = detection_loss + F.binary_cross_entropy(bond_pred_unchanged, bond_targets[:, None])

        madr_loss = self.detection_weight * detection_loss + recovery_loss

        # MSC: cross-view alignment and matching consistency
        subgraph_norm = F.normalize(subgraph_repr, p=2, dim=1)
        corrupted_norm = F.normalize(corrupted_repr, p=2, dim=1)
        similarity = subgraph_norm @ corrupted_norm.t()

        targets = torch.arange(len(corrupted_repr), device=corrupted_repr.device)
        alignment_loss = F.cross_entropy(similarity / self.temperature, targets)

        fwd_probs = similarity.softmax(dim=0)
        rev_probs = similarity.t().softmax(dim=0)
        mixture = 0.5 * (fwd_probs + rev_probs)
        fwd_kl = F.kl_div(similarity.log_softmax(dim=0), mixture, reduction='batchmean')
        rev_kl = F.kl_div(similarity.t().log_softmax(dim=0), mixture, reduction='batchmean')
        consistency_loss = 0.5 * (fwd_kl + rev_kl)

        total_loss = madr_loss + alignment_loss + self.consistency_weight * consistency_loss
        return total_loss


def train_molecular(data, config, device, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building molecular graphs for {len(data['molecules'])} molecules...")
    graphs = [smiles_to_graph(smiles, drug_id) for drug_id, smiles in data['molecules']]

    model = MolecularPretrainer(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    batch_size = config['batch_size']
    replace_ratio = config['replace_ratio']
    subgraph_ratio = config['subgraph_ratio']
    epochs = config['epochs']

    log_path = output_dir / 'epochs.jsonl'
    with log_path.open('w', encoding='utf-8') as log_file:
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(graphs)).tolist()
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, len(order), batch_size):
                batch_graphs = [graphs[idx] for idx in order[start:start + batch_size]]
                corrupted = [replace_node_edge(g, replace_ratio) for g in batch_graphs]
                subgraphs = [sample_subgraph(g, subgraph_ratio) for g in batch_graphs]

                corrupted_batch = Batch.from_data_list(corrupted).to(device)
                subgraph_batch = Batch.from_data_list(subgraphs).to(device)

                optimizer.zero_grad(set_to_none=True)
                loss = model.compute_loss(corrupted_batch, subgraph_batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Encountered non-finite loss at epoch {epoch}")

                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach())
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            record = {'epoch': epoch, 'loss': avg_loss}
            log_file.write(json.dumps(record) + '\n')
            log_file.flush()
            print(f"[Pretrain] Epoch {epoch:3d}/{epochs:3d} | Loss: {avg_loss:.5f}", flush=True)

    model.eval()
    all_representations = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start:start + batch_size]).to(device)
            _, projected = model.encoder(batch)
            all_representations.append(projected.cpu().numpy())

    embeddings = np.concatenate(all_representations, axis=0)
    print(f"Generated molecular embeddings: {embeddings.shape}")

    torch.save({'model': model.state_dict(), 'config': config}, output_dir / 'pretrain_model.pt')
    np.save(output_dir / 'drug_embeddings.npy', embeddings)
    np.save(output_dir / 'drug_ids.npy', np.asarray(data['ids']))
    print(f"Saved pretrained embeddings to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Molecular pretraining (MADR + MSC) for IntRo")
    parser.add_argument('--dataset', choices=['Deng', 'Ryu'], required=True, help="Target benchmark dataset")
    parser.add_argument('--data-root', type=Path, default=Path('./data'), help="Path to data directory")
    parser.add_argument('--output-dir', type=Path, required=True, help="Directory to save checkpoints and embeddings")
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.json', help="Path to config.json")
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding='utf-8'))
    pretrain_config = config['pretrain']
    if args.seed is not None:
        pretrain_config['seed'] = args.seed
    if min(pretrain_config['epochs'], pretrain_config['batch_size']) < 1:
        parser.error('epochs and batch_size must be positive')
    if any(not 0.0 <= pretrain_config[key] <= 1.0 for key in ('replace_ratio', 'subgraph_ratio')):
        parser.error('replace_ratio and subgraph_ratio must be in [0, 1]')
    if not pretrain_config['temperature'] > 0:
        parser.error('temperature must be positive')

    random.seed(pretrain_config['seed'])
    np.random.seed(pretrain_config['seed'])
    torch.manual_seed(pretrain_config['seed'])

    data = load_data(args.dataset, args.data_root)
    train_molecular(data, pretrain_config, torch.device(args.device), args.output_dir)


if __name__ == '__main__':
    main()
