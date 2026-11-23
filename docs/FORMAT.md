# GGUF layout

`general.architecture` is unchanged (`qwen3` for the tested checkpoint). For each factorised projection
`<base>` (`blk.N.attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`) the usual
`<base>.weight` is absent and these tensors are present. Shapes are in ggml order (`ne[0]` first).

| Tensor | Type | Shape | Content |
|---|---|---|---|
| `<base>.lb_g` | F32 | [in] | input scale `v2` |
| `<base>.lb_v` | Q1_0 | [in, r_pad] | sign(V); every block of row `k` has scale `d = v1[k] * u2[k]`; padded rows have `d = 0` |
| `<base>.lb_u` | Q1_0 | [r_pad, out] | sign(U); every block of row `i` has scale `d = u1[i]` |
| `<base>.lb_g_r`, `lb_v_r`, `lb_u_r` | same | same | residual path, when the checkpoint has one |

`r_pad` is the rank rounded up to a multiple of 128. The loader takes the rank from the tensor shape.

**Unpadded `lb_v` (format version 2).** `lb_v` may instead be stored at the true rank, shape `[in, r]`, with
`littlebit.format_version = 2`. `lb_u` stays `[r_pad, out]`, because the rank is its `Q1_0` block dimension.
The loader reads `r` and `r_pad` from the two shapes and zero-pads the latent (`ggml_pad`) when they differ.
`convert_littlebit_to_gguf.py --unpadded-v` writes this layout; the default is the padded one (version 1).
This follows the layout proposed in [GeniusDevam/littlebit-gguf issue #1](https://github.com/GeniusDevam/littlebit-gguf/issues/1).

Bit conversion: LittleBit packs rows into int32 words, least significant bit first, bit 1 = -1.
`Q1_0` uses 128-bit blocks, least significant bit first, bit 1 = +d. The converter inverts the bits and
writes the fp16 scale in front of each block. With `d = 1.0` this is byte-identical to gguf-py's own
`Q1_0` writer (`scripts/reference_forward.py`, self-test).

Metadata (informational, the loader does not need them): `littlebit.format_version` (uint32, 1),
`littlebit.residual` (bool), `littlebit.eff_bits` (float32), `littlebit.use_itq` (bool).

Forward pass built by the patch for a factorised weight:

```
t = ggml_mul(x, lb_g)
t = ggml_mul_mat(lb_v, t)
t = ggml_pad(t, r_pad - r)      # only when lb_v is unpadded
y = ggml_mul_mat(lb_u, t)
# plus the same for the residual path, added
```
