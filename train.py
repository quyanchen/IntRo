import argparse
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

from data import (
    DATASET_RELATIONS,
    build_training_graph,
    compute_relation_profile,
    find_dataset_dir,
    load_data,
    load_features,
    normalize_features,
)
from model import IntRo


def compute_metrics(y_true, logits, num_relations, label_policy='observed'):
    if label_policy not in {'fixed', 'observed'}:
        raise ValueError(f'Unknown Macro-F1 label policy: {label_policy}')
    labels = np.arange(num_relations) if label_policy == 'fixed' else None
    y_pred = logits.argmax(axis=1) if logits.ndim > 1 else logits

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
    }


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def evaluate_pairs(model, node_repr, triples, batch_size=2048):
    logits = []
    for batch in triples.split(batch_size):
        pred = model.predict(node_repr, batch[:, :2]).cpu()
        logits.append(pred)
    return torch.cat(logits, dim=0).numpy()


def train_downstream(data, features, config, device, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')

    edge_index, edge_types = build_training_graph(data['train'])
    relation_counts = compute_relation_profile(
        edge_index, edge_types, len(features), data['relations']
    )
    graph_features = normalize_features(features).to(device)
    molecular_features = features.to(device)
    edge_index = edge_index.to(device)
    edge_types = edge_types.to(device)
    relation_counts = relation_counts.to(device)

    train_triples_all = torch.from_numpy(data['train']).to(device)
    val_triples = torch.from_numpy(data['val']).to(device)
    test_triples = torch.from_numpy(data['test']).to(device)

    model = IntRo(
        num_relations=data['relations'],
        config=config,
        in_features=molecular_features.size(1),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    batch_size = config['batch_size']
    best_val_f1 = -1.0
    best_epoch = 0
    checkpoint_path = output_dir / 'best.pt'
    log_path = output_dir / 'epochs.jsonl'

    print(f"Training {config['dataset']} for {config['epochs']} epochs on {device}...")

    with log_path.open('w', encoding='utf-8') as log_file:
        for epoch in range(1, config['epochs'] + 1):
            model.train()

            # Corrupt node features and edge types for DGI
            node_perm = torch.randperm(len(graph_features), device=device)
            corrupted_features = graph_features[node_perm]

            paired_edge_types = edge_types.reshape(-1, 2)
            edge_perm = torch.randperm(len(paired_edge_types), device=device)
            corrupted_types = paired_edge_types[edge_perm].flatten()

            train_perm = torch.randperm(len(train_triples_all), device=device)
            shuffled_train = train_triples_all[train_perm]
            train_batches = shuffled_train.split(batch_size)

            total_loss = 0.0
            for batch_triples in train_batches:
                optimizer.zero_grad(set_to_none=True)
                loss = model.compute_loss(
                    graph_features,
                    edge_index,
                    edge_types,
                    relation_counts,
                    batch_triples,
                    corrupted_features,
                    corrupted_types,
                    molecular_features=molecular_features,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch}")

                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())

            avg_epoch_loss = total_loss / max(len(train_batches), 1)

            model.eval()
            with torch.no_grad():
                node_repr, _ = model.encode(
                    graph_features, edge_index, edge_types, relation_counts,
                    molecular_features=molecular_features,
                )
                val_logits = evaluate_pairs(model, node_repr, val_triples, batch_size)
                val_metrics = compute_metrics(
                    data['val'][:, 2], val_logits, data['relations'], config['macro_labels']
                )
                val_f1 = val_metrics['macro_f1']
                improved = val_f1 > best_val_f1

                record = {
                    'epoch': epoch,
                    'loss': avg_epoch_loss,
                    'val_macro_f1': val_f1,
                    'val_accuracy': val_metrics['accuracy'],
                    'best_updated': improved,
                }
                msg = f"Epoch {epoch:3d}/{config['epochs']:3d} | Loss: {avg_epoch_loss:.5f} | Val Macro-F1: {val_f1:.5f}"

                if improved:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    torch.save(
                        {
                            'model': model.state_dict(),
                            'epoch': epoch,
                            'best_val_macro_f1': val_f1,
                            'selection_metric': 'val_macro_f1',
                            'config': config,
                        },
                        checkpoint_path,
                    )
                    test_logits = evaluate_pairs(model, node_repr, test_triples, batch_size)
                    test_metrics = compute_metrics(
                        data['test'][:, 2], test_logits, data['relations'], config['macro_labels']
                    )
                    record['test'] = test_metrics
                    msg += (
                        f" | Test Acc: {test_metrics['accuracy']:.5f}"
                        f" | Test Macro-F1: {test_metrics['macro_f1']:.5f}"
                        f" | Test Macro-Pre: {test_metrics['macro_precision']:.5f}"
                        f" | Test Macro-Rec: {test_metrics['macro_recall']:.5f} [best]"
                    )

            log_file.write(json.dumps(record) + '\n')
            log_file.flush()
            print(msg, flush=True)

            if epoch - best_epoch >= config['patience']:
                print(f"Early stopping at epoch {epoch} (patience={config['patience']})")
                break

    print(f"\nEvaluating best checkpoint from epoch {best_epoch} (Val Macro-F1: {best_val_f1:.5f})...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with torch.no_grad():
        node_repr, _ = model.encode(
            graph_features, edge_index, edge_types, relation_counts,
            molecular_features=molecular_features,
        )
        final_test_logits = evaluate_pairs(model, node_repr, test_triples, batch_size)

    final_metrics = compute_metrics(
        data['test'][:, 2], final_test_logits, data['relations'], config['macro_labels']
    )

    np.savez_compressed(
        output_dir / 'predictions.npz',
        sample_id=np.arange(len(test_triples)),
        triples=data['test'],
        drug_ids=np.asarray(data['ids']),
        logits=final_test_logits,
        predicted=final_test_logits.argmax(axis=1),
    )

    result_summary = {
        'dataset': config['dataset'],
        'best_epoch': best_epoch,
        'best_val_macro_f1': best_val_f1,
        'test_metrics': final_metrics,
        'selection_metric': 'val_macro_f1',
    }
    (output_dir / 'metrics.json').write_text(
        json.dumps(result_summary, indent=2) + '\n', encoding='utf-8'
    )

    print(f"\nTest results ({config['dataset']}):")
    for metric_name, val in final_metrics.items():
        print(f"  {metric_name:<16}: {val:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Downstream multi-relational DDI training for IntRo")
    parser.add_argument('--dataset', choices=list(DATASET_RELATIONS.keys()), required=True, help="Target dataset (Deng or Ryu)")
    parser.add_argument('--data-root', type=Path, default=Path('./data'), help="Path to data directory")
    parser.add_argument('--features', type=Path, help="Path to pretrained molecular embeddings (drug_embeddings.npy)")
    parser.add_argument('--feature-ids', type=Path, help="Path to matching drug_ids.npy")
    parser.add_argument('--output-dir', type=Path, required=True, help="Directory to save checkpoints and logs")
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.json', help="Path to config.json")
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    full_config = json.loads(args.config.read_text(encoding='utf-8'))
    config = full_config['downstream'].copy()
    config.update(full_config['datasets'][args.dataset])
    config['dataset'] = args.dataset
    if args.seed is not None:
        config['seed'] = args.seed
    if min(config['epochs'], config['batch_size'], config['patience']) < 1:
        parser.error('epochs, batch_size and patience must be positive')
    if config['macro_labels'] not in {'fixed', 'observed'}:
        parser.error('macro_labels must be fixed or observed')

    seed_all(config['seed'])
    device = torch.device(args.device)

    data = load_data(args.dataset, args.data_root)

    feature_file = args.features
    if feature_file is None:
        dataset_dir = find_dataset_dir(args.data_root, args.dataset)
        default_feature = dataset_dir / 'drug_embeddings.npy'
        if default_feature.exists():
            feature_file = default_feature
        else:
            parser.error(f"--features is required. Default feature file not found at {default_feature}")

    features = load_features(data, feature_file, args.feature_ids)

    train_downstream(data, features, config, device, args.output_dir)


if __name__ == '__main__':
    main()
