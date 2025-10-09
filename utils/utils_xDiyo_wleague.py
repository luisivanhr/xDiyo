import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import colorsys
import pickle

def standardize_stats(batch, mu_atk, sigma_atk, mu_def, sigma_def):
    """
    Standardizes attacking and defensive stats using league-specific means and standard deviations.

    Assumes:
    - mu_atk and sigma_atk are tensors of shape (n_leagues, team_atk_length).
    - mu_def and sigma_def are tensors of shape (n_leagues, team_def_length).
    - batch['leagues'] is a LongTensor of shape (batch_size,) holding the league index for each sample.
    - The stats fields have shape (batch_size, max_seq_length, feature_dim).
    - The valid mask tensors have shape (batch_size, max_seq_length).

    The function selects the appropriate statistics for each sample and standardizes the features.
    """
    # Get league indices for each sample in the batch; shape: (batch_size,)
    league_idx = batch['leagues']  # Expected shape: (batch_size,)

    # For attack stats: shape becomes (batch_size, 1, team_atk_length) for broadcasting.
    per_sample_mu_atk = mu_atk[league_idx].unsqueeze(1)
    per_sample_sigma_atk = sigma_atk[league_idx].unsqueeze(1)

    # For defense stats: shape becomes (batch_size, 1, team_def_length)
    per_sample_mu_def = mu_def[league_idx].unsqueeze(1)
    per_sample_sigma_def = sigma_def[league_idx].unsqueeze(1)

    # Retrieve valid masks; unsqueeze the last dimension to match the stats tensors.
    home_mask = batch['home_team_valid_mask']
    away_mask = batch['away_team_valid_mask']

    # Standardize attack stats using league-specific parameters.
    batch['home_team_atk'] = ((batch['home_team_atk'] - per_sample_mu_atk) / per_sample_sigma_atk).float() * home_mask
    batch['home_enemy_atk'] = ((batch['home_enemy_atk'] - per_sample_mu_atk) / per_sample_sigma_atk).float() * home_mask
    batch['away_team_atk'] = ((batch['away_team_atk'] - per_sample_mu_atk) / per_sample_sigma_atk).float() * away_mask
    batch['away_enemy_atk'] = ((batch['away_enemy_atk'] - per_sample_mu_atk) / per_sample_sigma_atk).float() * away_mask

    # Standardize defense stats using league-specific parameters.
    batch['home_team_def'] = ((batch['home_team_def'] - per_sample_mu_def) / per_sample_sigma_def).float() * home_mask
    batch['home_enemy_def'] = ((batch['home_enemy_def'] - per_sample_mu_def) / per_sample_sigma_def).float() * home_mask
    batch['away_team_def'] = ((batch['away_team_def'] - per_sample_mu_def) / per_sample_sigma_def).float() * away_mask
    batch['away_enemy_def'] = ((batch['away_enemy_def'] - per_sample_mu_def) / per_sample_sigma_def).float() * away_mask

    return batch


def inverse_standardize_stats(home_away_stats, all_leagues, mu_atk, sigma_atk, mu_def, sigma_def):
    """
    Inverts standardization on concatenated home and away outputs using league-specific stats.
    
    Args:
        home_away_stats (Tensor): shape (2 * batch_size, feature_dim)
        all_leagues (Tensor): shape (2 * batch_size,), league indices for each sample
        mu_atk, sigma_atk: (n_leagues, atk_dim)
        mu_def, sigma_def: (n_leagues, def_dim)
    
    Returns:
        Tensor: De-standardized stats, same shape as home_away_stats
    """
    # Combine attack and defense stats along the feature dimension
    reference_mu = torch.cat([mu_atk, mu_def], dim=1).to(home_away_stats.device)     # (n_leagues, feature_dim)
    reference_sigma = torch.cat([sigma_atk, sigma_def], dim=1).to(home_away_stats.device)

    # Get the league-specific mu and sigma for each sample
    sample_mu = reference_mu[all_leagues]        # shape: (2 * batch_size, feature_dim)
    sample_sigma = reference_sigma[all_leagues]  # shape: (2 * batch_size, feature_dim)

    # Invert standardization: x = z * sigma + mu
    original_stats = home_away_stats * sample_sigma + sample_mu

    return original_stats

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

# Plotting heatmaps

def create_single_color_colormap(color='red', lightness_increase=0.1, lightness_decrease=0.05):
    """
    Creates a custom colormap that transitions from a light version of the given color
    to a dark version.

    Parameters:
    color (str): The base color (e.g., 'red', 'blue').
    lightness_increase (float): Amount to increase lightness to get the low-density color.
    lightness_decrease (float): Amount to decrease lightness to get the high-density color.

    Returns:
    LinearSegmentedColormap: A matplotlib colormap object.
    """
    # Convert the base color to RGB and then to HLS.
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    
    # Compute a light variant by increasing the lightness (capped at 1.0)
    light_L = min(1.0, l + lightness_increase)
    # Compute a dark variant by decreasing the lightness (down to a minimum of 0.0)
    dark_L = max(0.0, l - lightness_decrease)
    
    light_variant = colorsys.hls_to_rgb(h, light_L, s)
    dark_variant = colorsys.hls_to_rgb(h, dark_L, s)
    
    return LinearSegmentedColormap.from_list('custom_single_color', [light_variant, dark_variant])

def plot_heatmap(heatmap_tensor, n_x, n_y, title="Heatmap", cmap=None):
    """
    Plots a single heatmap given as a flattened torch tensor, using a custom colormap.
    
    Parameters:
      heatmap_tensor (torch.Tensor or list): Flattened tensor (n_x * n_y) of heatmap density.
    n_x (int): Number of divisions along the x-axis.
    n_y (int): Number of divisions along the y-axis.
    title (str): Title for the plot.
    cmap: Colormap for the plot. If None, a custom single-color colormap (from white to red) will be used.
    """
    if cmap is None:
        cmap = create_single_color_colormap('red')
    
    # Ensure tensor and reshape
    if not isinstance(heatmap_tensor, torch.Tensor):
        heatmap_tensor = torch.tensor(heatmap_tensor, dtype=torch.float32)
    grid = heatmap_tensor.reshape(n_y, n_x)
    
    plt.figure(figsize=(6, 5))
    im = plt.imshow(grid, origin="upper", interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.xlabel("X Grid")
    plt.ylabel("Y Grid")
    plt.colorbar(im, label="Density")
    plt.tight_layout()
    plt.show()

def plot_team_heatmaps(player_tensor, goalkeeper_tensor, n_x, n_y,
                    player_title="Player Heatmap", goalkeeper_title="Goalkeeper Heatmap", cmap=None):
    """
    Plots two heatmaps side-by-side (one for player points and one for goalkeeper points)
    using a custom colormap that ranges from white (low density) to a saturated color.
    
    Parameters:
    player_tensor (torch.Tensor or list): Flattened tensor for the player heatmap.
    goalkeeper_tensor (torch.Tensor or list): Flattened tensor for the goalkeeper heatmap.
    n_x (int): Number of grid divisions along the x-axis.
    n_y (int): Number of grid divisions along the y-axis.
    player_title (str): Title for the player heatmap.
    goalkeeper_title (str): Title for the goalkeeper heatmap.
    cmap: Colormap for the plots. If None, a custom single-color colormap (from white to red) will be used.
    """
    if cmap is None:
        cmap = create_single_color_colormap('red')
    
    # Prepare player tensor
    if not isinstance(player_tensor, torch.Tensor):
        player_tensor = torch.tensor(player_tensor, dtype=torch.float32)
    player_grid = player_tensor.reshape(n_y, n_x)
    
    # Prepare goalkeeper tensor
    if not isinstance(goalkeeper_tensor, torch.Tensor):
        goalkeeper_tensor = torch.tensor(goalkeeper_tensor, dtype=torch.float32)
    goalkeeper_grid = goalkeeper_tensor.reshape(n_y, n_x)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(player_grid, origin="upper", interpolation="nearest", cmap=cmap)
    axes[0].set_title(player_title)
    axes[0].set_xlabel("X Grid")
    axes[0].set_ylabel("Y Grid")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(goalkeeper_grid, origin="upper", interpolation="nearest", cmap=cmap)
    axes[1].set_title(goalkeeper_title)
    axes[1].set_xlabel("X Grid")
    axes[1].set_ylabel("Y Grid")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def save_cache(file_path, **kwargs):
    """
    Saves the given keyword arguments to a pickle file.
    
    Parameters:
        file_path (str): Where to save the pickle file.
        **kwargs: Named variables to save.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(kwargs, f)


def load_cache(file_path):
    """
    Loads variables from a pickle file.

    Parameters:
        file_path (str): The path to the pickle file.

    Returns:
        dict: A dictionary of loaded variables.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)