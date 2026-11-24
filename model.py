import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv


class RelationProfile(nn.Module):
    """Interaction Pattern Encoder (IPE).

    Maps each drug's degree-normalized relation-participation profile to a role embedding.
    """

    def __init__(self, num_relations, embedding_dim=75):
        super().__init__()
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, counts):
        weighted = counts @ self.relation_embeddings.weight
        total_counts = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        return weighted / total_counts

    def contrastive_loss(self, counts, entity_indices, margin=0.3, theta=2.0):
        """Triplet-margin loss over within-support (positive) and support-changing (negative) perturbations."""
        rows = counts[entity_indices]
        positive = rows.clone()
        negative = rows.clone()

        present_mask = rows > 0
        missing_mask = ~present_mask
        has_present = present_mask.any(dim=1)
        has_missing = missing_mask.any(dim=1)

        row_indices = torch.arange(len(rows), device=rows.device)
        avg_count = rows.sum(dim=1) / present_mask.sum(dim=1).float().clamp_min(1.0)

        if has_present.any():
            sampling_weights = present_mask.float() + 1e-10
            rescaled_rel = torch.multinomial(sampling_weights, 1).squeeze(1)
            perturbed_count = torch.rand(len(rows), device=rows.device) * avg_count * theta + 1.0
            positive[row_indices[has_present], rescaled_rel[has_present]] = perturbed_count[has_present]

            delete_mask = (torch.rand(len(rows), device=rows.device) < 0.5) & has_present
            removed_rel = torch.multinomial(sampling_weights, 1).squeeze(1)
            negative[row_indices[delete_mask], removed_rel[delete_mask]] = 0.0

        if has_missing.any():
            insert_mask = (torch.rand(len(rows), device=rows.device) < 0.5) & has_missing
            added_rel = torch.multinomial(missing_mask.float() + 1e-10, 1).squeeze(1)
            inserted_count = torch.rand(len(rows), device=rows.device) * avg_count * theta + 1.0
            negative[row_indices[insert_mask], added_rel[insert_mask]] = inserted_count[insert_mask]

        anchor = self(rows)
        pos_dist = (anchor - self(positive)).norm(p=2, dim=1)
        neg_dist = (anchor - self(negative)).norm(p=2, dim=1)
        return (pos_dist - neg_dist + margin).clamp_min(0.0).mean()


class Discriminator(nn.Module):

    def __init__(self, hidden_dim=75):
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        nn.init.xavier_uniform_(self.bilinear.weight)
        nn.init.zeros_(self.bilinear.bias)

    def forward(self, summary, clean, corrupted):
        context = summary.expand_as(clean)
        score_clean = self.bilinear(clean, context)
        score_corrupted = self.bilinear(corrupted, context)
        return torch.cat([score_clean, score_corrupted], dim=1)


class IntRo(nn.Module):

    def __init__(self, num_relations, config, in_features=300):
        super().__init__()
        self.rgcn1 = RGCNConv(in_features, 150, num_relations)
        self.rgcn2 = RGCNConv(150, 75, num_relations)

        self.layer_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.discriminator = Discriminator(hidden_dim=75)
        self.dropout = config.get('dropout', 0.55)
        mlp_dropout = config.get('mlp_dropout', 0.30)

        self.relation_profile = RelationProfile(num_relations, embedding_dim=75)

        # (150 [rgcn1] + 75 [rgcn2] + 75 [profile] + in_features [mol]) * 2 drugs
        pair_feature_dim = (150 + 75 + 75 + in_features) * 2
        self.mlp = nn.Sequential(
            nn.Linear(pair_feature_dim, 256),
            nn.ELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(128, num_relations),
        )

        self.ipe_weight = config.get('ipe_weight', 0.1)
        self.margin = config.get('margin', 0.3)
        self.theta = config.get('theta', 2.0)
        self.dgi_node_weight = config.get('dgi_node_weight', 0.05)
        self.dgi_edge_weight = config.get('dgi_edge_weight', 0.10)

    def encode_topology(self, features, edge_index, edge_type):
        h1 = self.rgcn1(features, edge_index, edge_type).relu()
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.rgcn2(h1, edge_index, edge_type)
        return h1, h2

    def encode(self, graph_features, edge_index, edge_type, counts, *, molecular_features):
        h1, h2 = self.encode_topology(graph_features, edge_index, edge_type)
        topology = torch.cat([self.layer_weights[0] * h1, self.layer_weights[1] * h2], dim=1)
        profile = self.relation_profile(counts)
        node_repr = torch.cat([topology, profile, molecular_features], dim=1)
        return node_repr, h2

    def predict(self, node_repr, pairs):
        src_feat = node_repr[pairs[:, 0]]
        dst_feat = node_repr[pairs[:, 1]]
        pair_feat = torch.cat([src_feat, dst_feat], dim=1)
        return self.mlp(pair_feat)

    def compute_loss(
        self, graph_features, edge_index, edge_type, counts, triples,
        corrupted_features, corrupted_types, *, molecular_features,
    ):
        node_repr, clean_h2 = self.encode(
            graph_features, edge_index, edge_type, counts, molecular_features=molecular_features,
        )
        logits = self.predict(node_repr, triples[:, :2])
        classification_loss = F.cross_entropy(logits, triples[:, 2])

        # DGI mutual-information objectives
        _, corrupt_nodes_h2 = self.encode_topology(corrupted_features, edge_index, edge_type)
        _, corrupt_edges_h2 = self.encode_topology(graph_features, edge_index, corrupted_types)
        summary = clean_h2.mean(dim=0).sigmoid()

        dgi_labels = graph_features.new_zeros(len(graph_features), 2)
        dgi_labels[:, 0] = 1.0  # col 0: clean, col 1: corrupted

        node_mi_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(summary, clean_h2, corrupt_nodes_h2), dgi_labels
        )
        edge_mi_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(summary, clean_h2, corrupt_edges_h2), dgi_labels
        )

        batch_entities = triples[:, :2].flatten().unique()
        ipe_loss = self.relation_profile.contrastive_loss(counts, batch_entities, self.margin, self.theta)

        total_loss = (
            classification_loss
            + self.dgi_node_weight * node_mi_loss
            + self.dgi_edge_weight * edge_mi_loss
            + self.ipe_weight * ipe_loss
        )
        return total_loss
