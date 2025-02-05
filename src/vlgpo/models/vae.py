import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=28, latent_dim=8, vocab_size=20, embedding_dim=64):
        """
        Variational Autoencoder (VAE) for sequence data.

        Args:
            input_dim (int): Length of input sequences (default=28).
            latent_dim (int): Dimension of the latent space (default=8).
            vocab_size (int): Number of unique tokens in the input (default=20).
            embedding_dim (int): Dimension of the token embedding (default=64).
        """
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.vocab_size = vocab_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1),  # Downsample: 28 -> 14
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),   # Downsample: 14 -> 7
            nn.ReLU()
        )

        # Fully connected layers for latent space
        flat_dim = 32 * (self.input_dim // 4)  # Flattened dimension after downsampling
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose1d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 14 -> 28
            nn.ReLU(),
            nn.ConvTranspose1d(128, embedding_dim, kernel_size=3, stride=1, padding=1)
        )

        # Output projection layer
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

    def encode(self, x):
        x_embed = self.embedding(x)              # (batch, seq_len, embedding_dim)
        x_embed = x_embed.permute(0, 2, 1)       # (batch, embedding_dim, seq_len)

        h = self.encoder(x_embed)                # (batch, 32, 7)
        h = h.flatten(start_dim=1)               # (batch, 32*7)
        mu = self.fc_mu(h)                       # (batch, latent_dim)
        logvar = self.fc_logvar(h)               # (batch, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec(z))               # (batch, 224)
        h = h.view(h.size(0), 32, self.input_dim // 4)  # (batch, 32, 7)

        h = self.decoder(h)                      # (batch, embedding_dim, seq_len=28)
        h = h.permute(0, 2, 1)                   # (batch, 28, embedding_dim)
        logits = self.output_proj(h)             # (batch, 28, vocab_size)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar, z