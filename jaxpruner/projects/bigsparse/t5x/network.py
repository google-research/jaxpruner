"""T5.1.1 Transformer model with width scaling."""
from flax import linen as nn
import jax.numpy as jnp
from t5x.examples.t5 import layers
from t5x.examples.t5 import network


class TransformerW(network.Transformer):
  """An encoder-decoder Transformer model."""

  config: network.T5Config
  width: float = 1.0

  def setup(self):
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
