"""KV-cache size of Qwen2.5-7B (batch 1, bf16) for 8k and 32k tokens.

    KV bytes = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_value x tokens
    head_dim = hidden_size / attention_heads

Config verified against https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
    python kv_cache.py [--tokens 8192 32768]
"""
import argparse

LAYERS = 28          # num_hidden_layers
ATTN_HEADS = 28      # num_attention_heads (query heads)
KV_HEADS = 4         # num_key_value_heads (grouped-query attention)
HIDDEN = 3584        # hidden_size
BYTES = 2            # bfloat16
HEAD_DIM = HIDDEN // ATTN_HEADS  # 128


def bytes_per_token(kv_heads=KV_HEADS):
    return 2 * LAYERS * kv_heads * HEAD_DIM * BYTES


def kv_cache_bytes(tokens, kv_heads=KV_HEADS):
    return tokens * bytes_per_token(kv_heads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[8 * 1024, 32 * 1024])
    args = ap.parse_args()

    print("Qwen2.5-7B: 28 layers, 28 query heads, 4 KV heads, hidden 3584, bf16 (2 bytes)")
    print(f"head_dim  = {HIDDEN} / {ATTN_HEADS} = {HEAD_DIM}")
    print(f"per token = 2 x {LAYERS} x {KV_HEADS} x {HEAD_DIM} x {BYTES} = "
          f"{bytes_per_token():,} bytes = {bytes_per_token() // 1024} KiB")
    for n in args.tokens:
        b = kv_cache_bytes(n)
        print(f"{n:>6,} tokens x {bytes_per_token():,} B = {b:,} B = {b / 2**20:,.0f} MiB"
              f" = {b / 2**30:.3f} GiB (= {b / 1e9:.3f} GB decimal)")
    full = kv_cache_bytes(32 * 1024, kv_heads=ATTN_HEADS)
    print(f"For comparison, without GQA (28 KV heads) 32k would need {full / 2**30:.2f} GiB (7x).")
    print("Cache only, batch size 1; model weights and runtime buffers are extra.")


if __name__ == "__main__":
    main()
