from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class EmbedLayer(nn.Module):
    """Class for Embedding an Image. It breaks image into patches and embeds patches using a Conv2D
    Operation (Works same as the Linear layer). Next, a learnable positional embedding vector is
    added to all the patch embeddings to provide spatial position. Finally, a classification token
    is added which is used to classify the image.

    Parameters:
        n_channels (int) : Number of channels of the input image
        embed_dim  (int) : Embedding dimension
        image_size (int) : Image size
        patch_size (int) : Patch size
        dropout  (float) : dropout value

    Input:
        x (tensor): Image Tensor of shape B, C, IW, IH

    Returns:
        Tensor: Embedding of the image of shape B, S, E
    """

    def __init__(
        self,
        n_channels: int,
        embed_dim: int,
        image_size: int,
        patch_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, (image_size // patch_size) ** 2, embed_dim), requires_grad=True
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.conv1(x)  # B, C, IH, IW     --> B, E, IH/P, IW/P
        x = x.reshape([B, x.shape[1], -1])  # B, E, IH/P, IW/P --> B, E, (IH/P*IW/P) --> B, E, N
        x = x.permute(0, 2, 1)  # B, E, N          --> B, N, E
        x = x + self.pos_embedding  # B, N, E          --> B, N, E
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 1, 1, E --> B, 1, E
        x = torch.cat((cls_tokens, x), dim=1)  # B, N, E          --> B, (N+1), E       --> B, S, E
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    """Class for computing self attention Self-Attention.

    Parameters:
        embed_dim (int)        : Embedding dimension
        n_attention_heads (int): Number of attention heads to use for performing MultiHeadAttention

    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output after Self-Attention Module of shape B, S, E
    """

    def __init__(self, embed_dim: int, n_attention_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_attention_heads = n_attention_heads
        self.head_embed_dim = embed_dim // n_attention_heads

        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
        self.out_projection = nn.Linear(
            self.head_embed_dim * self.n_attention_heads, self.embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, e = x.shape

        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xq = xq.permute(0, 2, 1, 3)
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xk = xk.permute(0, 2, 1, 3)
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xv = xv.permute(0, 2, 1, 3)

        # Compute Attention presoftmax values
        xk = xk.permute(0, 1, 3, 2)
        x_attention = torch.matmul(xq, xk)

        x_attention /= float(self.head_embed_dim) ** 0.5

        x_attention = torch.softmax(x_attention, dim=-1)

        x = torch.matmul(x_attention, xv)

        # Format the output
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, s, e)

        x = self.out_projection(x)
        return x


class Encoder(nn.Module):
    """Class for creating an encoder layer.

    Parameters:
        embed_dim (int)         : Embedding dimension
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (int)       : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        dropout (float)         : Dropout parameter

    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output of the encoder block of shape B, S, E
    """

    def __init__(
        self, embed_dim: int, n_attention_heads: int, forward_mul: int, dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.attention(self.norm1(x)))
        x = x + self.dropout2(self.fc2(self.activation(self.fc1(self.norm2(x)))))
        return x


class Classifier(nn.Module):
    """Classification module of the Vision Transformer. Uses the embedding of the classification
    token to generate logits.

    Parameters:
        embed_dim (int) : Embedding dimension
        n_classes (int) : Number of classes

    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Logits of shape B, CL
    """

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0, :]  # B, S, E --> B, E          Get CLS token
        x = self.fc1(x)  # B, E    --> B, E
        x = self.activation(x)  # B, E    --> B, E
        x = self.fc2(x)  # B, E    --> B, CL
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer Class compatible with Lightning-Hydra template.

    Parameters:
        input_size (int)        : Input size for backward compatibility (will be ignored)
        n_channels (int)        : Number of channels of the input image
        image_size (int)        : Image size (height and width, assumed square)
        patch_size (int)        : Patch size
        embed_dim (int)         : Embedding dimension
        n_layers (int)          : Number of encoder blocks to use
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (int)       : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        output_size (int)       : Number of classes
        dropout (float)         : dropout value
        use_torch_layers (bool) : Whether to use PyTorch's built-in transformer layers

    Input:
        x (tensor): Image Tensor of shape B, C, IW, IH

    Returns:
        Tensor: Logits of shape B, CL
    """

    def __init__(
        self,
        input_size: int = 784,  # For backward compatibility with template, will be ignored
        n_channels: int = 1,
        image_size: int = 28,
        patch_size: int = 4,
        embed_dim: int = 64,
        n_layers: int = 6,
        n_attention_heads: int = 4,
        forward_mul: int = 2,
        output_size: Optional[int] = None,
        heads_config: Optional[Dict[str, int]] = None,
        dropout: float = 0.1,
        use_torch_layers: bool = False,
        output_mode: str = "classification",
        parameter_names: Optional[List[str]] = None,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        super().__init__()

        self.use_torch_layers = use_torch_layers
        self.n_channels = n_channels
        self.image_size = image_size
        # Store input shape attributes for TensorBoard summaries
        self.input_shape = (n_channels, image_size, image_size)
        self.input_resolution = (image_size, image_size)

        # Store output mode and parameter information
        self.output_mode = output_mode
        self.parameter_names = parameter_names or []
        self.parameter_ranges = parameter_ranges or {}

        # Handle configuration based on output mode
        if output_mode == "regression":
            # For regression, we need parameter names (can be empty if auto-configured later)
            if parameter_names:
                # Create heads_config for regression (each parameter gets 1 output)
                heads_config = {name: 1 for name in parameter_names}
            else:
                # Will be auto-configured later from dataset
                heads_config = {}
        else:
            # Backward compatibility: convert old single-head config to multihead
            if heads_config is None:
                if output_size is not None:
                    heads_config = {"digit": output_size}
                else:
                    heads_config = {"digit": 10}  # Default MNIST

        self.heads_config = heads_config
        self.is_multihead = len(heads_config) > 1

        # Always use custom embedding layer
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout=dropout)

        if use_torch_layers:
            # Use PyTorch's built-in transformer layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_attention_heads,
                dim_feedforward=forward_mul * embed_dim,
                dropout=dropout,
                activation=nn.GELU(),
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, n_layers, norm=nn.LayerNorm(embed_dim)
            )
        else:
            # Use custom scratch implementation
            self.encoder = nn.ModuleList(
                [
                    Encoder(embed_dim, n_attention_heads, forward_mul, dropout=dropout)
                    for _ in range(n_layers)
                ]
            )
            self.norm = nn.LayerNorm(embed_dim)

        # Multiple heads or single head for backward compatibility
        if self.is_multihead:
            if output_mode == "regression":
                # For regression, create heads with sigmoid activation
                self.heads = nn.ModuleDict(
                    {
                        head_name: nn.Sequential(
                            nn.Linear(embed_dim, embed_dim),
                            nn.Tanh(),
                            nn.Linear(embed_dim, 1),
                            nn.Sigmoid()
                        )
                        for head_name in heads_config.keys()
                    }
                )
            else:
                # Classification heads
                self.heads = nn.ModuleDict(
                    {
                        head_name: nn.Sequential(
                            nn.Linear(embed_dim, embed_dim),
                            nn.Tanh(),
                            nn.Linear(embed_dim, num_classes)
                        )
                        for head_name, num_classes in heads_config.items()
                    }
                )
        else:
            # Single head (backward compatibility)
            if heads_config:
                head_name, num_classes = next(iter(heads_config.items()))
                if output_mode == "regression":
                    self.classifier = nn.Sequential(
                        nn.Linear(embed_dim, embed_dim),
                        nn.Tanh(),
                        nn.Linear(embed_dim, 1),
                        nn.Sigmoid()
                    )
                else:
                    self.classifier = Classifier(embed_dim, num_classes)
            # If heads_config is empty, don't create classifier - will be auto-configured later

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize the weights of the Vision Transformer."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, EmbedLayer):
            nn.init.trunc_normal_(m.cls_token, mean=0.0, std=0.02)
            nn.init.trunc_normal_(m.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: A tensor of logits.
        """
        # Handle input format - expect (B, C, H, W) format for images
        if x.dim() == 2:
            # If flattened input (for backward compatibility), reshape to image format
            batch_size = x.size(0)
            # Assume square image
            img_size = int((x.size(1) // self.embedding.conv1.in_channels) ** 0.5)
            x = x.view(batch_size, self.embedding.conv1.in_channels, img_size, img_size)

        x = self.embedding(x)

        if self.use_torch_layers:
            x = self.encoder(x)
        else:
            for block in self.encoder:
                x = block(x)
            x = self.norm(x)

        # Extract CLS token embedding
        cls_token = x[:, 0, :]  # B, S, E --> B, E

        if self.is_multihead:
            return {head_name: head(cls_token) for head_name, head in self.heads.items()}
        else:
            # Single head output (backward compatibility)
            if hasattr(self, 'classifier'):
                if isinstance(self.classifier, Classifier):
                    # Use the original Classifier which handles CLS token extraction
                    return self.classifier(x)
                else:
                    # Use the new sequential classifier with pre-extracted CLS token
                    return self.classifier(cls_token)
            else:
                # No classifier configured - should not happen in normal operation
                raise ValueError("No classifier configured. Model may not be properly initialized.")

    def _build_heads(self, heads_config: Dict[str, int]) -> None:
        """Rebuild heads for auto-configuration (supports both classification and regression modes)."""
        # Get embed_dim from existing layers
        if hasattr(self, 'heads') and self.heads:
            # Get embed_dim from existing head
            first_head = next(iter(self.heads.values()))
            if isinstance(first_head, nn.Sequential):
                embed_dim = first_head[0].in_features
            else:
                embed_dim = first_head.in_features
        elif hasattr(self, 'classifier'):
            if isinstance(self.classifier, Classifier):
                embed_dim = self.classifier.fc1.in_features
            elif isinstance(self.classifier, nn.Sequential):
                embed_dim = self.classifier[0].in_features
            else:
                embed_dim = self.classifier.in_features
        else:
            # Fallback to norm layer size
            embed_dim = self.norm.normalized_shape[0]

        # Update configuration
        self.heads_config = heads_config
        self.is_multihead = len(heads_config) > 1

        # Remove old classifier if transitioning to multihead
        if hasattr(self, 'classifier') and self.is_multihead:
            delattr(self, 'classifier')

        # Remove old heads if transitioning to single head
        if hasattr(self, 'heads') and not self.is_multihead:
            delattr(self, 'heads')

        if self.is_multihead:
            if self.output_mode == "regression":
                # Create regression heads with sigmoid activation
                self.heads = nn.ModuleDict(
                    {
                        head_name: nn.Sequential(
                            nn.Linear(embed_dim, embed_dim),
                            nn.Tanh(),
                            nn.Linear(embed_dim, 1),
                            nn.Sigmoid()
                        )
                        for head_name in heads_config.keys()
                    }
                )
            else:
                # Classification mode: create heads with appropriate number of classes
                self.heads = nn.ModuleDict(
                    {
                        head_name: nn.Sequential(
                            nn.Linear(embed_dim, embed_dim),
                            nn.Tanh(),
                            nn.Linear(embed_dim, num_classes)
                        )
                        for head_name, num_classes in heads_config.items()
                    }
                )
        else:
            # Single head (backward compatibility)
            if heads_config:
                head_name, num_classes = next(iter(heads_config.items()))
                if self.output_mode == "regression":
                    self.classifier = nn.Sequential(
                        nn.Linear(embed_dim, embed_dim),
                        nn.Tanh(),
                        nn.Linear(embed_dim, 1),
                        nn.Sigmoid()
                    )
                else:
                    self.classifier = Classifier(embed_dim, num_classes)
            # If heads_config is empty, don't create classifier - will be auto-configured later


if __name__ == "__main__":
    # Test the Vision Transformer
    print("Testing Vision Transformer:")

    # Test single-head mode (backward compatibility)
    print("Single-head mode:")
    model_single = VisionTransformer(
        n_channels=1,
        image_size=28,
        patch_size=4,
        embed_dim=64,
        n_layers=4,
        n_attention_heads=4,
        forward_mul=2,
        output_size=10,
        dropout=0.1,
    )

    x_single = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images
    output_single = model_single(x_single)
    print(f"  Input: {x_single.shape} -> Output: {output_single.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_single.parameters()):,}")

    # Test multihead mode
    print("\nMultihead mode:")
    model_multi = VisionTransformer(
        n_channels=3,
        image_size=32,
        patch_size=4,
        embed_dim=64,
        n_layers=4,
        n_attention_heads=4,
        forward_mul=2,
        heads_config={"fine_label": 100, "coarse_label": 20, "texture": 8},
        dropout=0.1,
    )

    x_multi = torch.randn(2, 3, 32, 32)  # Batch of 2 CIFAR images
    output_multi = model_multi(x_multi)
    print(f"  Input: {x_multi.shape}")
    print(f"  Output type: {type(output_multi)}")
    for head_name, logits in output_multi.items():
        print(f"    {head_name}: {logits.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_multi.parameters()):,}")
