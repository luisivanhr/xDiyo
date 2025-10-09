import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def standardize_stats(batch, mu_atk, sigma_atk, mu_def, sigma_def):
    home_mask = batch['home_team_valid_mask']
    away_mask = batch['away_team_valid_mask']

    batch['home_team_def'] = ((batch['home_team_def'] - mu_def) / sigma_def).float() * home_mask
    batch['home_enemy_def'] = ((batch['home_enemy_def'] - mu_def) / sigma_def).float() * home_mask
    batch['home_team_atk'] = ((batch['home_team_atk'] - mu_atk) / sigma_atk).float() * home_mask
    batch['home_enemy_atk'] = ((batch['home_enemy_atk'] - mu_atk) / sigma_atk).float() * home_mask

    batch['away_team_def'] = ((batch['away_team_def'] - mu_def) / sigma_def).float() * away_mask
    batch['away_enemy_def'] = ((batch['away_enemy_def'] - mu_def) / sigma_def).float() * away_mask
    batch['away_team_atk'] = ((batch['away_team_atk'] - mu_atk) / sigma_atk).float() * away_mask
    batch['away_enemy_atk'] = ((batch['away_enemy_atk'] - mu_atk) / sigma_atk).float() * away_mask
    
    return batch


def inv_standardize_stats(batch, mu_atk, sigma_atk, mu_def, sigma_def):
    home_mask = batch['home_team_valid_mask']
    away_mask = batch['away_team_valid_mask']

    batch['home_team_def']   = (batch['home_team_def']   * sigma_def + mu_def)   * home_mask
    batch['home_enemy_def']  = (batch['home_enemy_def']  * sigma_def + mu_def)   * home_mask
    batch['home_team_atk']   = (batch['home_team_atk']   * sigma_atk + mu_atk)   * home_mask
    batch['home_enemy_atk']  = (batch['home_enemy_atk']  * sigma_atk + mu_atk)   * home_mask

    batch['away_team_def']   = (batch['away_team_def']   * sigma_def + mu_def)   * away_mask
    batch['away_enemy_def']  = (batch['away_enemy_def']  * sigma_def + mu_def)   * away_mask
    batch['away_team_atk']   = (batch['away_team_atk']   * sigma_atk + mu_atk)   * away_mask
    batch['away_enemy_atk']  = (batch['away_enemy_atk']  * sigma_atk + mu_atk)   * away_mask

    return batch

def inv_standardize_sample(
    batch,
    mu_atk,
    sigma_atk,
    mu_def,
    sigma_def,
    device,
    indices=None
):
    mu_atk = mu_atk.to(device)
    sigma_atk = sigma_atk.to(device)
    mu_def = mu_def.to(device)
    sigma_def = sigma_def.to(device)

    mu = torch.cat([mu_atk, mu_def], dim=-1)
    sigma = torch.cat([sigma_atk, sigma_def], dim=-1)

    if indices is not None:
        mu = mu[..., indices]
        sigma = sigma[..., indices]

    reverted = batch * sigma + mu
    return reverted



def right_align_sequences(sequences, lengths):
    """Right-aligns sequences by moving valid rows to the end of each sequence."""

    batch_size, L, dim = sequences.shape
    aligned_sequences = torch.zeros_like(sequences)

    for i in range(batch_size):
        valid_rows = sequences[i, :lengths[i], :]  # Extract valid rows
        aligned_sequences[i, L - lengths[i]:, :] = valid_rows  # Move them to the right
    
    return aligned_sequences


def left_align_sequences(sequences, lengths):
    """Left-aligns sequences by moving valid rows to the end of each sequence."""
    batch_size, L, dim = sequences.shape
    aligned_sequences = torch.zeros_like(sequences)
    
    for i in range(batch_size):
        valid_rows = sequences[i, L-lengths[i]:,:]
        aligned_sequences[i, :lengths[i],:] = valid_rows
    
    return aligned_sequences


def split_current(tensor):
    """Separates current and previous matches"""
    current = tensor[:,-1,:]
    previous = tensor[:,:-1,:]
    
    return previous, current 


def masked_softmax(logits, mask, dim=-1):
    """
    Perform softmax on 'logits' over dimension 'dim',
    ignoring positions where mask == 0.
    
    Args:
        logits: Tensor of shape (..., L, ...)
        mask:   Binary Tensor matching 'logits' shape along 'dim'
                or broadcastable. 1 = valid, 0 = invalid/pad.
        dim:    Dimension along which to apply softmax.
    Returns:
        probs:  Same shape as 'logits' with sum of valid positions == 1.
    """
    # Replace invalid positions with -inf before softmax
    masked_logits = logits.masked_fill(mask == 0, float('-inf'))
    probs = F.softmax(masked_logits, dim=dim)
    # Replace any NaNs in the output with zeros
    probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
    
    return probs

#End of utilities

class CustomDotProductAttention(nn.Module):
    """
    Dot Product Attention between two sequences (X and Y).

    We assume each sequence has:
    - Embedding  (B, Lx, E) or (B, Ly, E)
    - Attack     (B, Lx, A) or (B, Ly, A)
    - Defense    (B, Lx, D) or (B, Ly, D)

    We concatenate them:
    X_feat = [X_embed, X_atk, X_def] -> (B, Lx, x_dim)
    Y_feat = [Y_embed, Y_atk, Y_def] -> (B, Ly, y_dim)

    Then compute dot-product scores e_{ij} = X_feat[i] dot Y_feat[j].
    A softmax over j (the Y dimension) yields alpha_{ij}.
    """
    def __init__(self, 
                do_scale=True):
        """
        Args:
            do_scale (bool): If True, applies scaled dot product by dividing
                            by sqrt(feature_dim). If False, no scaling.
        """
        super().__init__()
        self.do_scale = do_scale

    def forward(self,
                X_embed, X_atk, X_def,   # (B, L_x, E/A/D)
                Y_embed, Y_atk, Y_def,   # (B, L_y, E/A/D)
                valid_mask_y=None):      # (B, L_y,1) or None
        """
        Returns:
            alpha: (B, L_x, L_y) attention weights where sum_j alpha_{ij} = 1 (for valid j).
        """
        B, L_x, _ = X_embed.shape
        _, L_y, _ = Y_embed.shape
        
        # 1) Concatenate features for X and Y
        #    => X_feat (B, L_x, x_dim), Y_feat (B, L_y, y_dim)
        X_feat = torch.cat([X_embed, X_atk, X_def], dim=-1)
        Y_feat = torch.cat([Y_embed, Y_atk, Y_def], dim=-1)
        
        x_dim = X_feat.shape[-1]
        y_dim = Y_feat.shape[-1]
        
        # They must match for a pure dot product:
        if x_dim != y_dim:
            raise ValueError(
                f"Dot product attention requires X_feat and Y_feat to have the same dim, "
                f"but got x_dim={x_dim} and y_dim={y_dim}."
            )
        
        # 2) Dot product: e = Q * K^T => (B, L_x, L_y)
        #    We'll do batch matrix multiplication: 
        #       Q = X_feat  shape (B, L_x, d)
        #       K = Y_feat  shape (B, L_y, d)
        #    so e = Q @ K^T => shape (B, L_x, L_y)
        
        # Q: (B, L_x, d)
        # K^T: (B, d, L_y)
        # => e: (B, L_x, L_y)
        # but we can't directly do Q @ K^T in one step with typical matrix multiply,
        # we use bmm => Q.bmm(K.transpose(1,2))
        
        Q = X_feat
        K = Y_feat
        e = torch.bmm(Q, K.transpose(1, 2))  # (B, L_x, L_y)
        
        # 3) (Optional) scale by sqrt(d)
        if self.do_scale:
            d = float(x_dim)
            e = e / math.sqrt(d)
        
        # 4) Mask + Softmax over j
        #    If valid_mask_y is not None, we do a masked_softmax over dim=2
        if valid_mask_y is not None:
            # Expand mask_y to (B, 1, L_y) so it can broadcast across L_x
            permuted_mask = valid_mask_y.permute(0, 2, 1)  # (B, 1, L_y)
            alpha = masked_softmax(e, permuted_mask, dim=2)  # (B, L_x, L_y)
        else:
            # Normal softmax over the last dimension
            alpha = F.softmax(e, dim=2)
        
        return alpha*valid_mask_y
    
class GridMaker(nn.Module):
    def __init__(self):
        super(GridMaker, self).__init__()

    def forward(self, X, Y):
        """
        Args:
            X: Tensor of shape (B, L, Dim)
            Y: Tensor of shape (B, L, Dim)

        Returns:
            grid_x: (B*L, L, Dim)
            grid_y: (B*L, L, Dim)
        """
        B, L, D = X.shape

        # 1) Expand X from (B, L, D) -> (B, L, 1, D),
        #    then expand along dimension=2 (length L in Y's dimension).
        #    Finally reshape to (B*L, L, D).
        grid_x = (
            X.unsqueeze(2)            # (B, L, 1, D)
            .expand(-1, -1, L, -1)   # (B, L, L, D)
            .contiguous()
             .view(B * L, L, D)       # (B*L, L, D)
        )

        # 2) Expand Y from (B, L, D) -> (B, 1, L, D),
        #    then expand along dimension=1 (length L in X's dimension).
        #    Finally reshape to (B*L, L, D).
        grid_y = (
            Y.unsqueeze(1)            # (B, 1, L, D)
            .expand(-1, L, -1, -1)   # (B, L, L, D)
            .contiguous()
             .view(B * L, L, D)       # (B*L, L, D)
        )

        return grid_x, grid_y
    
def build_edge_index(match_list):
    """
    Build an edge index tensor from a list of matches.
    Each match is a tuple: (home_team_index, away_team_index).
    We assume undirected edges, so we add both directions.
    """
    edges = []
    for home, away in match_list:
        edges.append((home, away))
        edges.append((away, home))
    # Convert to tensor and transpose to shape [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def extract_edge_info(dataset):
    graph_testie = [(dataset[i]['home_data']['team_index'][-1], dataset[i]['home_data']['enemy_index'][-1]) for i in range(len(dataset))]
    graph_testie = list(set(graph_testie))
    edge_index = build_edge_index(graph_testie)
    return edge_index

class PopOff(nn.Module):
    def __init__(self, thresholds=(1.0, 1.5, 2.0), separate_flags=True, create_labels=True, steep=10.0):
        """
        Args:
            thresholds (tuple): Multiples of sigma to use as thresholds.
            separate_flags (bool): If True, returns flags for each threshold separately.
                                If False, returns the sum of the flags.
            create_labels (bool): Default mode for creating flags.
                                If True, uses hard step functions (non-differentiable).
                                If False, uses steep sigmoid approximations (differentiable).
            steep (float): The steepness parameter for the sigmoid approximation.
                        Higher values make the sigmoid more step-like.
        """
        super(PopOff, self).__init__()
        self.thresholds = thresholds
        self.separate_flags = separate_flags
        self.create_labels = create_labels
        self.steep = steep

    def _compute_flags(self, stats, create_labels=None):
        # Use module default if not provided
        if create_labels is None:
            create_labels = self.create_labels

        # stats: tensor of shape [batch_size, num_matches, feature_dim]
        flag_list = []
        for thr in self.thresholds:
            if create_labels:
                # Hard step function: -1 if stat < -thr, 0 if -thr <= stat <= thr, 1 if stat > thr.
                flags_thr = torch.where(
                    stats < -thr,
                    torch.full_like(stats, -1),
                    torch.where(
                        stats > thr,
                        torch.full_like(stats, 1),
                        torch.zeros_like(stats)
                    )
                )
            else:
                # Differentiable approximation using steep sigmoids.
                # Approximates a step from -1 to 0 at -thr and from 0 to 1 at thr.
                flags_thr = torch.sigmoid(self.steep * (stats - thr)) - torch.sigmoid(self.steep * (-stats - thr))
            flag_list.append(flags_thr)
        
        if self.separate_flags:
            # Output shape: [batch_size, num_matches, feature_dim, n_thresholds]
            flags = torch.stack(flag_list, dim=-1)
        else:
            # Sum across thresholds, output shape: [batch_size, num_matches, feature_dim]
            flags = torch.stack(flag_list, dim=-1).sum(dim=-1)
        return flags

    def forward(self, stats, create_labels=None):
        """
        Args:
            stats: Tensor of shape [batch_size, num_matches, feature_dim].
            create_labels (bool, optional): If provided, overrides the module's default for hard
                                            or soft flag creation.
        Returns:
            Tensor containing the flag features. If separate_flags is True, the shape is
            [batch_size, num_matches, feature_dim, n_thresholds]; otherwise, [batch_size, num_matches, feature_dim].
        """
        return self._compute_flags(stats, create_labels)
    
class FeatureGating(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, stats_features, flag_features):
        combined = torch.cat([stats_features, flag_features], dim=-1)
        gate_weights = self.gate(combined)
        # Multiply the flag features by the gate and add to stats features
        return stats_features + gate_weights * flag_features
