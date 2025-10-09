from utils_xDiyo import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class CreateEmbMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, num_layers=1, dropout=0.0, bias=None):
        super().__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            final_layer = nn.Linear(input_dim, output_dim)
            if bias is not None:
                # Expecting bias to be a tensor of shape (output_dim,)
                if bias.shape != final_layer.bias.shape:
                    raise ValueError(f"Bias tensor shape mismatch. Expected {final_layer.bias.shape}, got {bias.shape}")
                final_layer.bias.data.copy_(bias)
            self.layers = final_layer
        else:
            hidden_dim = hidden_dim if hidden_dim is not None else input_dim
            net = []
            # First layer
            net.append(nn.Linear(input_dim, hidden_dim))
            net.append(nn.Tanh())
            if dropout > 0:
                net.append(nn.Dropout(dropout))
            # Hidden layers
            for _ in range(num_layers - 2):
                net.append(nn.Linear(hidden_dim, hidden_dim))
                net.append(nn.Tanh())
                if dropout > 0:
                    net.append(nn.Dropout(dropout))
            # Final output layer
            final_layer = nn.Linear(hidden_dim, output_dim)
            if bias is not None:
                if bias.shape != final_layer.bias.shape:
                    raise ValueError(f"Bias tensor shape mismatch. Expected {final_layer.bias.shape}, got {bias.shape}")
                final_layer.bias.data.copy_(bias)
            net.append(final_layer)
            self.layers = nn.Sequential(*net)

    def forward(self, x):
        return self.layers(x)
    
class TeamDense(nn.Module):
    def __init__(self, num_teams, embed_dim, pad_idx=0):
        """
        num_teams: Number of distinct teams (including <PAD>).
        embed_dim: Embedding dimensionality.
        pad_idx:   Index representing <PAD>.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_teams, embed_dim, padding_idx=pad_idx)
        #self.fc = nn.Sequential(
        #    nn.Linear(embed_dim , embed_dim),
        #    nn.Tanh()
        #)

    def forward(self, team_ids):
        """
        team_ids: LongTensor of shape (batch_size, max_seq_length)
                containing the integer IDs of teams (including <PAD> for padding).

        Returns: A FloatTensor of shape (batch_size, max_seq_length, embed_dim).
                Padded positions will automatically have embedding vectors of zeros
                because we specified padding_idx in nn.Embedding.
        """
        # The embedding layer looks up embeddings for all valid indices
        # and returns them in the same shape as `team_ids`, but with an extra embed_dim dimension.
        embedded = self.embedding(team_ids)  # (batch_size, max_seq_length, embed_dim)
        #embedded = self.fc(embedded)
        return embedded


class TeamEmbedding(nn.Module):
    def __init__(self, embed_dim, num_teams, mu_sigma, pad_idx=0, stats_hidden_dim=None, 
                coupled=False, device='cuda', training_embeds=True):
        """
        Args:
            embed_dim: Dimensionality of the team embedding.
            num_teams: Number of teams.
            mu_sigma: Tuple (mu_atk, sigma_atk, mu_def, sigma_def).
            pad_idx: Padding index for the embedding.
            stats_hidden_dim: Hidden dimension for the prediction MLP.
            coupled: If True, uses a coupled prediction branch.
            device: Device string.
            training_embeds: If True, uses only the last row of the sequence (for training);
                            if False, applies the transformations to the entire sequence.
        """
        super(TeamEmbedding, self).__init__()
        self.device = device
        self.training_embeds = training_embeds

        self.mu_atk, self.sigma_atk, self.mu_def, self.sigma_def = mu_sigma

        #self.mu_atk = self.mu_atk.to(self.device)
        #self.sigma_atk = self.sigma_atk.to(self.device)
        #self.mu_def = self.mu_def.to(self.device)
        #self.sigma_def = self.sigma_def.to(self.device)
        self.stats_dim = len(self.mu_atk) + len(self.mu_def)

        self.coupled = coupled

        # Embedding layer
        self.embedding_layer = TeamDense(num_teams, embed_dim, pad_idx)

        # Prediction module (MLP)
        if self.coupled:
            self.prediction = CreateEmbMLP((embed_dim + 2)*2, 2*self.stats_dim,
                                        hidden_dim=stats_hidden_dim, num_layers=5,
                                        dropout=0.0)
        else:
            self.prediction = CreateEmbMLP(embed_dim + 2, self.stats_dim,
                                        hidden_dim=stats_hidden_dim, num_layers=3,
                                        dropout=0.0)

    def forward(self, batch,key=None):
        #batch = standardize_stats(batch, self.mu_atk, self.sigma_atk, self.mu_def, self.sigma_def)
        batch_size = batch['home_team_atk'].shape[0]

        # Obtain masks and lengths for both teams.
        valid_home_mask = batch['home_team_valid_mask']
        valid_away_mask = batch['away_team_valid_mask']
        home_scores_lengths = batch['home_team_scores_lengths'].to('cpu')
        away_scores_lengths = batch['away_team_scores_lengths'].to('cpu')

        if self.training_embeds:
            # Use only the last row of each sequence.
            home_team_atk = batch['home_team_atk'][:,-1:,:]
            home_team_def = batch['home_team_def'][:,-1:,:]
            away_team_atk = batch['away_team_atk'][:,-1:,:]
            away_team_def = batch['away_team_def'][:,-1:,:]

            home_team_is_home = batch['home_team_is_home'][:,-1:]
            home_team_standings = batch['home_team_standings'][:,-1:]
            away_team_is_home = batch['away_team_is_home'][:,-1:]
            away_team_standings = batch['away_team_standings'][:,-1:]

            home_team_indexes = batch['home_team_index'][:,-1:]
            away_team_indexes = batch['away_team_index'][:,-1:]
            home_team_names = [name for names in batch['home_team_names'] for name in names[-1:]]
            away_team_names = [name for names in batch['away_team_names'] for name in names[-1:]]
            
            # Combine indexes and names.
            all_indexes = torch.cat([home_team_indexes, away_team_indexes], dim=0)
            all_names = home_team_names + away_team_names
            
            # Compute embeddings for the indexes.
            home_team_embed = self.embedding_layer(home_team_indexes)
            away_team_embed = self.embedding_layer(away_team_indexes)
            all_teams_embed = torch.cat([home_team_embed, away_team_embed], dim=0)
            
            # Combine attack and defense stats.
            home_stats = torch.cat([home_team_atk, home_team_def], dim=-1)
            away_stats = torch.cat([away_team_atk, away_team_def], dim=-1)
            labels = torch.cat([home_stats, away_stats], dim=0)
        else:
            # Use the entire sequence.
            team_atk = batch[f'{key}_team_atk']      # shape: (B, seq_len, dim)
            team_def = batch[f'{key}_team_def']
            enemy_atk = batch[f'{key}_enemy_atk']
            enemy_def = batch[f'{key}_enemy_def']

            team_is_home = batch[f'{key}_team_is_home']  # shape: (B, seq_len, 1)
            team_standings = batch[f'{key}_team_standings']
            enemy_is_home = batch[f'{key}_enemy_is_home']
            enemy_standings = batch[f'{key}_enemy_standings']

            # For indexes and names, assume that the whole sequence is provided;
            # You might decide to aggregate these (e.g. using the last element or a mean) for downstream usage.
            # Here we simply use the entire sequence for further processing.
            team_indexes = batch[f'{key}_team_index']
            enemy_indexes = batch[f'{key}_enemy_index']
            # For team names, we assume each sample contains a list of names (one per match)
            # and we take the last name in the sequence (or you might choose to do something else).
            team_names = batch[f'{key}_team_names']
            enemy_names = batch[f'{key}_enemy_names']
            
            team_embed = self.embedding_layer(team_indexes)
            enemy_embed = self.embedding_layer(enemy_indexes)
        
        if self.training_embeds:
            labels = labels.squeeze(1)  # when using last row, squeeze the second dimension

        # Predict stats from embeddings.
        if self.coupled:
            
            if self.training_embeds:
                home_away_input = torch.cat([home_team_embed, home_team_is_home, home_team_standings,
                                        away_team_embed, away_team_is_home, away_team_standings], dim=-1)
                predicted_stats = self.prediction(home_away_input)
                # When coupled, split predictions.
                home_stats_pred = predicted_stats[..., :self.stats_dim]
                away_stats_pred = predicted_stats[..., self.stats_dim:]
                home_away_stats = torch.cat([home_stats_pred, away_stats_pred], dim=0).squeeze(1)
                
            else:
                home_away_input = torch.cat([team_embed, team_is_home, team_standings,
                                        enemy_embed, enemy_is_home, enemy_standings], dim=-1)
                predicted_stats = self.prediction(home_away_input)
                # When coupled, split predictions.
                if key == 'home':
                    valid_mask = valid_home_mask
                else:
                    valid_mask = valid_away_mask
                team_pred = predicted_stats[..., :self.stats_dim]*valid_mask
                enemy_pred = predicted_stats[..., self.stats_dim:]*valid_mask
                
                return team_embed,enemy_embed, team_pred, enemy_pred
        
        
        else:
            home_away_input = torch.cat([home_team_embed, home_team_is_home, home_team_standings,
                                        away_team_embed, away_team_is_home, away_team_standings], dim=-1)
            home_away_stats = self.prediction(home_away_input)

        return labels, home_away_stats, all_indexes, all_names, all_teams_embed

    def _embeds_only(self, indexes):
        return self.embedding_layer(indexes)
