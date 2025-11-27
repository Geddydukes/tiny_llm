# tiny-llm

Minimal yet production-grade tiny language model stack. Includes:

- SentencePiece tokenizer tooling with chat-friendly special tokens
- Data pipeline that shards token sequences into `.npy` blocks
- PyTorch Transformer with RoPE, RMSNorm, SwiGLU, weight tying
- Training and sanity-check scripts tuned for CPU/MPS/GPU on macOS

## Quickstart

1. **Train tokenizer**
   `python scripts/train_tokenizer.py --input data.txt --model_prefix tokenizer`
2. **Prepare data**
   `python scripts/prepare_data.py --tokenizer tokenizer.model --input data.txt --output_dir data/shards`
3. **Pretrain model**
   `python scripts/train_pretrain.py --tokenizer tokenizer.model --data_dir data/shards --out_dir checkpoints`
4. **Sanity check**
   `python scripts/sanity_check_model.py`

All code targets Python 3.10+.
