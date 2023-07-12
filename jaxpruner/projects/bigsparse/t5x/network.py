"""T5.1.1 Transformer model with width scaling."""
from flax import linen as nn
import jax.numpy as jnp
from t5x.examples.t5 import layers
from t5x.examples.t5 import network


class TransformerW(network.Transformer):
  """An encoder-decoder Transformer model."""

  config: network.T5Config
  width: float = 1.0
  scale: float = 0.0  # num-heads constant scaling

  def setup(self):
    if self.scale != 0.0:
      fac = self.scale**0.5
      head_dim = int(self.config.emb_dim * fac) // self.config.num_heads
      emb_dim = head_dim * self.config.num_heads
      new_vals = {
          'emb_dim': emb_dim,
          'mlp_dim': int(self.config.mlp_dim * fac),
          # 'mlp_dim': int(self.config.mlp_dim * fac // 4) * 4,
          'head_dim': head_dim,
      }
    else:
      new_vals = {
          k: int(getattr(self.config, k) * self.width)
          for k in ('emb_dim', 'mlp_dim', 'num_heads')
      }
    new_cfg = self.config.replace(**new_vals)

    self.shared_embedding = layers.Embed(
        num_embeddings=new_cfg.vocab_size,
        features=new_cfg.emb_dim,
        dtype=new_cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='token_embedder',
    )

    self.encoder = network.Encoder(
        config=new_cfg, shared_embedding=self.shared_embedding
    )
    self.decoder = network.Decoder(
        config=new_cfg, shared_embedding=self.shared_embedding
    )
