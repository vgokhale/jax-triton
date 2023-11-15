import jax
from jax import random
import jax.numpy as jnp
from jax.experimental.pallas.ops import attention
import sys
import time

def fused_attention_fwd(q, k, v, batch_size, num_heads, seq_len, head_dim,
                        causal):

  @jax.jit
  def f(q, k, v):
    return attention.mha(q, k, v, None, causal=causal)

  o = f(q, k, v)
  iter = 100
  start_time = time.time()
  for i in range (0, iter):
    o = f(q, k, v)
  jax.block_until_ready(o)
  end_time = time.time()
  causal_factor = 2 if causal is False else 1 
  # *2 from ops / MAC and /2 due to causal cancel out. x2 because of 2 GEMMs
  # in fwd
  tflops = iter * batch_size * num_heads * seq_len**2 * head_dim * 2 / (end_time - start_time) / 1e12 * causal_factor
  print(f"Seqlen = {seq_len}")
  print(f"TFLOPS = {tflops}")
  return o

def fused_attention_bwd(q, k, v, batch_size, num_heads, seq_len, head_dim,
                        causal):

  @jax.jit
  def f(q, k, v):
    return attention.mha(q, k, v, None, causal=causal).sum()

  grad_jitted = jax.jit(jax.grad(f, argnums=(0, 1, 2)))
  dq, dk, dv = grad_jitted(q, k, v)
  iter = 1
  start_time = time.time()
  for i in range (0, iter):
    dq, dk, dv = grad_jitted(q, k, v)
  jax.block_until_ready(dq)
  jax.block_until_ready(dk)
  jax.block_until_ready(dv)
  end_time = time.time()
  causal_factor = 2 if causal is False else 1 
  # *2 from ops / MAC and /2 due to causal cancel out. x7 because of 5 GEMMs
  # in bwd + 2 in fwd
  tflops = iter * batch_size * num_heads * seq_len**2 * head_dim * 7 / (end_time - start_time) / 1e12 * causal_factor
  print(f"Seqlen = {seq_len}")
  print(f"TFLOPS = {tflops}")
  return dq, dk, dv

def main(args=None):
    bs, nheads, d = 4, 48, 64
    seqlen = [1024]#, 2048, 4096, 8192, 16384]
    causal = True
    k1, k2, k3 = random.split(random.PRNGKey(0), 3)

    @jax.jit
    def f_ref(q, k, v):
      return attention.mha_reference(q, k, v, None, causal=causal).sum()

    #for sq in seqlen:
    #  q = random.normal(k1, (bs, sq, nheads, d), dtype=jnp.float16)
    #  k = random.normal(k2, (bs, sq, nheads, d), dtype=jnp.float16)
    #  v = random.normal(k3, (bs, sq, nheads, d), dtype=jnp.float16)
    #  o = fused_attention_fwd(q, k, v, bs, nheads, sq, d, causal)
    #o_ref = f_ref(q, k, v)
    #print(f"err o = {jnp.max(o_ref-o)}")

    for sq in seqlen:
      q = random.normal(k1, (bs, sq, nheads, d), dtype=jnp.float16)
      k = random.normal(k2, (bs, sq, nheads, d), dtype=jnp.float16)
      v = random.normal(k3, (bs, sq, nheads, d), dtype=jnp.float16)
    dq, dk, dv = fused_attention_bwd(q, k, v, bs, nheads, sq, d, causal)
    dq_ref, dk_ref, dv_ref = jax.jit(jax.grad(f_ref, argnums=(0, 1, 2)))(q, k, v)
    print(f"err dq = {jnp.max(dq_ref-dq)}")
    print(f"err dk = {jnp.max(dk_ref-dk)}")
    print(f"err dv = {jnp.max(dv_ref-dv)}")

if __name__ == '__main__':
    sys.exit(main())
