# TinyLLM: A Minimal Yet Production-Grade Language Model Stack

A complete, end-to-end implementation of a small language model trained from scratch, featuring modern Transformer architecture, efficient data pipelines, and instruction fine-tuning. This project demonstrates the full ML lifecycle: data preparation, pretraining, fine-tuning, and evaluation.

## 🎯 Overview

TinyLLM is a compact language model (512 hidden dim, 12 layers, 8 heads) designed to generate CLI commands from natural language instructions. The model achieves **93.94% exact-match accuracy** on a held-out test set, demonstrating that small models can be highly effective for domain-specific tasks.

### Key Results

- **Pretraining**: 50,000 steps (~204M tokens seen) from 133 Wikipedia shards
- **Fine-tuning**: 2,000 steps on 2,302 instruction-command pairs
- **Test Accuracy**: 93.94% (93/99 exact matches)
- **Model Size**: 66.73M parameters (266.91 MB FP32)
- **Training Time**: ~13 hours pretraining + ~4 minutes fine-tuning on Apple Silicon

## 🏗️ Architecture

### Model Architecture Diagram

```
<instruction> text
      ↓
[ Tokenizer (SentencePiece) ]
      ↓
┌───────────────────────────────┐
│  TinyLLM Transformer (12L)    │
│   • RoPE                      │
│   • Multi-head Attention      │
│   • SwiGLU FFN                │
│   • RMSNorm                   │
└───────────────────────────────┘
      ↓
<command> tokens
```

### Model Components

- **Transformer Architecture**: 12 layers, 512 hidden dimension, 8 attention heads
- **RoPE (Rotary Position Embedding)**: Modern positional encoding
- **RMSNorm**: Root Mean Square Layer Normalization
- **SwiGLU**: Swish-Gated Linear Unit activation
- **Weight Tying**: Shared embeddings between input and output layers
- **SentencePiece Tokenizer**: 32K vocabulary with special tokens

### Special Tokens

- `<instruction>`: Marks the start of natural language instructions
- `<command>`: Marks the start of CLI command output
- Standard tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`

## 📊 Performance

### 📊 Quantitative Results

| Stage                 | Metric               | Value                    |
|-----------------------|----------------------|--------------------------|
| Pretraining           | Final loss           | **3.59**                 |
| Pretraining           | Tokens processed     | **204.8M**               |
| Fine-tuning (SFT)     | Steps                | **2,000**                |
| Held-out evaluation   | Exact match          | **93.94%** (93/99)       |
| Model size            | Parameters           | **66.73M**               |
| Model size            | FP32 checkpoint size | **266.91 MB**            |
| Hardware              | Training device      | 24GB M4 Mac Mini (MPS)   |
| Inference             | Generation speed     | **103.4 tok/s** (MPS)    |

### Evaluation Results

Tested on 99 held-out instruction-command pairs:

```
Total examples: 99
Exact matches:  93
Accuracy:       93.94%
```

### Failure Analysis

The 6 failure cases (6.06%) fall into specific categories:
1. **Special characters/escape sequences** (1 case): Complex regex patterns
2. **Email addresses** (1 case): Domain completion
3. **Input redirections** (1 case): File redirection syntax
4. **Version specifiers** (1 case): Package version syntax
5. **Complex pipe chains** (1 case): Multi-stage pipelines
6. **Regex patterns** (1 case): Character class syntax

All failures involve the model stopping early (EOS token) rather than generation errors, suggesting training data augmentation opportunities.

### Inference Performance

TinyLLM is designed to run efficiently on consumer hardware. Here is the measured generation throughput on a 24GB M4 Mac Mini (MPS):

```
Total tokens generated: 1280
Total time:             12.38 s
Tokens per second:      103.4 tok/s
Time per token:         9.67 ms
```

**Inference throughput**: ~103 tokens/second  
**Hardware**: Apple Silicon M4 (24GB unified memory)

This performance shows that TinyLLM can generate commands nearly instantly for interactive CLI agents, local assistants, or embedded tools.

## 💡 Why This Project Matters

TinyLLM demonstrates that:

- **You don't need massive GPUs to train an LLM end-to-end** — Trained entirely on a Mac Mini with 24GB unified memory
- **Small, domain-specific LLMs can achieve high accuracy** — 93.94% exact-match accuracy with just 66.73M parameters
- **Training infrastructure matters as much as model size** — Efficient data pipelines enable training on consumer hardware
- **Modern LLM techniques scale down effectively** — RoPE, RMSNorm, and SwiGLU work well even at small scales
- **Building your own stack teaches you more than using off-the-shelf models** — Deep understanding of every component from tokenization to inference

This project proves that with careful engineering, domain-specific language models can be trained from scratch on accessible hardware while maintaining production-grade quality.

## 🧠 Skills Demonstrated

- **Custom tokenizer training** — SentencePiece with domain-specific special tokens
- **Streaming dataset pipelines** — HuggingFace datasets → efficient numpy shards
- **Transformer architecture implementation** — Built from scratch with modern components
- **Training loop engineering** — MPS optimization, checkpointing, learning rate scheduling
- **Instruction tuning (SFT)** — Supervised fine-tuning with proper loss masking
- **Loss masking and formatting strategies** — Only command portion contributes to loss
- **Evaluation harness development** — Comprehensive accuracy testing framework
- **Error analysis and model debugging** — Systematic failure case analysis
- **End-to-end ML pipeline** — From raw data to deployed model
- **Production-ready code** — Type hints, error handling, modular design

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Geddydukes/tiny_llm.git
cd tiny_llm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch datasets sentencepiece tqdm
pip install -e .
```

### Training Pipeline

#### 1. Create Tokenizer Corpus

```bash
python scripts/create_tokenizer_corpus.py \
  --out data/tokenizer_corpus.txt \
  --articles 50000
```

#### 2. Train Tokenizer

```bash
python scripts/train_tokenizer.py \
  --input data/tokenizer_corpus.txt \
  --model_prefix tokenizer
```

This creates `tokenizer.model` with special tokens including `<instruction>` and `<command>`.

#### 3. Stream Wikipedia → Token Shards

```bash
mkdir -p data/wiki_shards

python scripts/stream_wiki_to_shards.py \
  --tokenizer tokenizer.model \
  --out_dir data/wiki_shards \
  --seq_len 512 \
  --shard_tokens 1000000 \
  --max_articles 500000
```

#### 4. Pretrain on Wikipedia

```bash
mkdir -p checkpoints/wiki_pretrain

python scripts/train_pretrain.py \
  --tokenizer tokenizer.model \
  --data_dir data/wiki_shards \
  --out_dir checkpoints/wiki_pretrain \
  --batch_size 8 \
  --max_steps 50000 \
  --warmup_steps 1000
```

#### 5. Convert Instruction/Command Pairs to JSONL

```bash
python scripts/convert_raw_to_jsonl.py \
  --input data/raw_cli_pairs.txt \
  --output data/cli_sft.jsonl
```

#### 6. Fine-tune on CLI Instructions

```bash
mkdir -p checkpoints/cli_sft

python scripts/train_cli_sft.py \
  --tokenizer tokenizer.model \
  --jsonl data/cli_sft.jsonl \
  --out_dir checkpoints/cli_sft \
  --from_ckpt checkpoints/wiki_pretrain/pretrain_step_050000.pt \
  --batch_size 8 \
  --max_steps 2000 \
  --warmup_steps 200 \
  --max_seq_len 256
```

### Inference

#### Interactive REPL Mode

```bash
python scripts/generate_command.py \
  --tokenizer tokenizer.model \
  --ckpt checkpoints/cli_sft/cli_sft_step_002000.pt
```

Then type instructions:
```
> find all .log files not accessed in the last 30 days
[command] find . -type f -name '*.log' -atime +30 -print
```

#### Single Command

```bash
python scripts/generate_command.py \
  --tokenizer tokenizer.model \
  --ckpt checkpoints/cli_sft/cli_sft_step_002000.pt \
  --instruction "list all files in current directory"
```

### Evaluation

```bash
python scripts/eval_cli_accuracy.py \
  --tokenizer tokenizer.model \
  --ckpt checkpoints/cli_sft/cli_sft_step_002000.pt \
  --instructions data/test_instructions_heldout.txt \
  --gold data/test_commands_heldout.txt \
  --max_new_tokens 64 \
  --limit 100 \
  --output_jsonl results/cli_eval_heldout.jsonl
```

## 📁 Project Structure

```
tiny_llm/
├── src/tiny_llm/          # Core model implementation
│   ├── __init__.py
│   ├── config.py          # Model configuration
│   ├── tokenizer.py       # SentencePiece wrapper
│   ├── data.py            # Dataset classes
│   ├── rope.py            # RoPE implementation
│   ├── layers.py          # Transformer blocks
│   └── model.py           # Main model
├── scripts/               # Training and utility scripts
│   ├── train_tokenizer.py
│   ├── create_tokenizer_corpus.py
│   ├── stream_wiki_to_shards.py
│   ├── train_pretrain.py
│   ├── convert_raw_to_jsonl.py
│   ├── train_cli_sft.py
│   ├── generate_command.py
│   └── eval_cli_accuracy.py
├── data/                  # Data files
│   ├── raw_cli_pairs.txt
│   ├── test_instructions_heldout.txt
│   └── test_commands_heldout.txt
└── checkpoints/           # Model checkpoints (gitignored)
```

## 🔬 Technical Details

### Training Configuration

**Pretraining:**
- Optimizer: AdamW (lr=3e-4, β₁=0.9, β₂=0.95, weight_decay=0.01)
- Learning Rate: Cosine schedule with 1,000 step warmup
- Gradient Clipping: 1.0
- Sequence Length: 512 tokens
- Batch Size: 8
- Total Steps: 50,000

**Fine-tuning:**
- Optimizer: AdamW (lr=1e-4, same betas)
- Learning Rate: Cosine schedule with 200 step warmup
- Sequence Length: 256 tokens
- Batch Size: 8
- Total Steps: 2,000

### Loss Masking

During fine-tuning, only the command portion (after `<command>`) contributes to the loss. The instruction portion is masked with `-100` (ignore index).

### Data Pipeline

- **Streaming**: Wikipedia data is streamed directly from HuggingFace datasets
- **Efficient Storage**: Tokenized sequences stored as numpy arrays (`.npy` shards)
- **Memory-Mapped**: Shards loaded with memory mapping for efficient access
- **Low Disk Usage**: ~300MB for 133 shards (~133M tokens)

## 📈 Training Curves

### Pretraining Loss

- Initial loss: ~60.67 (step 100)
- Final loss: ~3.59 (step 50,000)
- **93% reduction** over training

Loss progression:
- Step 1,000: 10.97
- Step 5,000: 6.31
- Step 10,000: 5.39
- Step 20,000: 4.56
- Step 30,000: 4.23
- Step 50,000: 3.59

## 🛠️ Implementation Highlights

### Production-Ready Features

- ✅ **Device Agnostic**: Automatic MPS/CUDA/CPU detection
- ✅ **Efficient Data Loading**: Memory-mapped numpy shards
- ✅ **Streaming Data**: Low-memory Wikipedia streaming
- ✅ **Checkpointing**: Regular model saves during training
- ✅ **Evaluation Suite**: Comprehensive accuracy testing
- ✅ **Error Handling**: Robust parsing and validation

### Code Quality

- Type hints throughout
- Comprehensive error messages
- Modular, reusable components
- Clean separation of concerns

## 🎓 Learning Outcomes

This project demonstrates:

1. **End-to-end ML pipeline**: From raw data to deployed model
2. **Modern architectures**: RoPE, RMSNorm, SwiGLU
3. **Efficient training**: Streaming data, memory-mapped shards
4. **Instruction tuning**: Supervised fine-tuning for specific tasks
5. **Evaluation methodology**: Held-out test sets, exact-match accuracy

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{tiny_llm,
  title = {TinyLLM: A Minimal Yet Production-Grade Language Model Stack},
  author = {Dukes, Geddy},
  year = {2025},
  url = {https://github.com/Geddydukes/tiny_llm}
}
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- SentencePiece for tokenization
- HuggingFace datasets for Wikipedia streaming
- PyTorch for the deep learning framework
- Inspired by modern LLM architectures (LLaMA, GPT, etc.)

## 🔮 Future Improvements

- [ ] Add more training examples for failure cases (email addresses, pipes, etc.)
- [ ] Implement beam search for generation
- [ ] Add support for multi-turn conversations
- [ ] Experiment with larger model sizes
- [ ] Add support for other command types (PowerShell, etc.)

---

**Built with ❤️ for learning and research**
