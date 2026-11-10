
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding # For data and model parallelism (explained in more detail later)
from jax.experimental import mesh_utils

import flax.nnx as nnx
import optax

from dataclasses import dataclass
import grain.python as pygrain
import pandas as pd
import tiktoken
import time