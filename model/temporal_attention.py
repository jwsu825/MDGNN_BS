import torch
from torch import nn

from utils.utils import MergeLayer


class TemporalAttentionLayer(torch.nn.Module):
  def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim,
               output_dimension, n_head=2,
               dropout=0.1):
    super(TemporalAttentionLayer, self).__init__()

    self.n_head = n_head

    self.feat_dim = n_node_features
    self.time_dim = time_dim

    self.query_dim = n_node_features + time_dim
    self.key_dim = n_neighbors_features + time_dim + n_edge_features

    self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)

    self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                   kdim=self.key_dim,
                                                   vdim=self.key_dim,
                                                   num_heads=n_head,
                                                   dropout=dropout)

  def forward(self, src_node_features, src_time_features, neighbors_features,
              neighbors_time_features, edge_features, neighbors_padding_mask):

    src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)

    query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)
    key = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2)

    query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
    key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]

    invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
    neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False

    attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key,
                                                              key_padding_mask=neighbors_padding_mask)

    attn_output = attn_output.squeeze()
    attn_output_weights = attn_output_weights.squeeze()

    attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
    attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)
    attn_output = self.merger(attn_output, src_node_features)

    return attn_output, attn_output_weights
