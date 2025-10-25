# 🧠 Makemore — v2: Multi-Layer Perceptron (MLP)

This is **Version 2** of my journey through *Andrej Karpathy’s Neural Networks: Zero to Hero* series.  

In this version, I evolve the simple **Bigram model** from v1 into a **neural network-based character-level language model** — a **Multi-Layer Perceptron (MLP)** built entirely from scratch using **PyTorch**.

---

## 🚀 What this model does
The goal remains the same:  
> **Generate realistic human names** by learning letter-to-letter relationships from a dataset of real names.

But now, instead of only looking at one character at a time,  
the model uses **multiple previous characters (context)** to predict the next one — allowing it to learn **richer dependencies** and **patterns in names**.

---

## 🧩 Concepts learned in v2

### 1. Data Preparation
- Loaded names from `names.txt`.  
- Built a vocabulary of unique characters and added a special end token `'.'`.  
- Created mappings:
  - `stoi` → string to index  
  - `itos` → index to string  
- Constructed input–target pairs (`X`, `Y`) where:
  - Each `X[i]` = context of last *N* characters
  - Each `Y[i]` = next character to predict  

Example (context size = 3):
```
"emma." → ("emm" → 'a'), ("mma" → '.'), etc.
```

---

### 2. Building the Dataset
- Padded each name with `.` on both sides to mark start/end.  
- Created context windows of size `block_size = 3`.  
- Stored the resulting training pairs as integer indices and converted them to PyTorch tensors.

---

### 3. Model Architecture (MLP)
**Input:** concatenated embeddings of previous `block_size` characters.  
**Layers:**
1. **Embedding layer** — maps character indices to dense vectors.  
2. **Hidden layer (ReLU)** — learns non-linear relationships.  
3. **Output layer (Linear)** — predicts logits for the next character.

**Activation:** ReLU  
**Loss:** Cross-Entropy  
**Optimizer:** Adam / SGD  

Mathematically:
\[
\text{logits} = W_2 \, \text{ReLU}(W_1 [\text{embed}(x_1), \text{embed}(x_2), \text{embed}(x_3)] + b_1) + b_2
\]

---

### 4. Training
- Used batches of `(X, Y)` pairs for efficient gradient updates.  
- Computed loss via `F.cross_entropy(logits, Y)`.  
- Used PyTorch autograd for backpropagation.  
- Trained for several epochs until loss converged.

---

### 5. Name Generation
- Start with an empty context (`[0, 0, 0]`).  
- Predict next character probabilities from the model.  
- Sample from the probability distribution.  
- Append sampled character to context and repeat until '.' is generated.

Example generated names:
```
melina.
jorena.
sariel.
davira.
tomika.
```

---

## 🧮 Technical Summary

| Component | Description |
|------------|--------------|
| **Input** | Previous `block_size` characters (3) |
| **Embedding Dim** | 10 |
| **Hidden Layer** | 100 neurons, ReLU activation |
| **Output** | 27 logits (one per character) |
| **Loss** | Cross-Entropy |
| **Optimizer** | Adam |
| **Dataset** | ~32,000 English names (`names.txt`) |

---

## 🧠 Key Takeaways
- Transitioned from **statistical model** → **neural model**.  
- Introduced **embeddings**, **non-linear activations**, and **hidden layers**.  
- Learned how **context** captures multi-letter dependencies.  
- Built intuition for MLPs — the foundation of modern deep networks.

---

## 🗂️ Project Structure

```
makemore/
│
├── data/
│   └── names.txt                # dataset of names
│
├── v2_mlp/
│   ├── makemore_v2.ipynb        # main notebook
│   ├── README.md                # this file
│   └── output_samples.txt       # generated names
│
└── ...
```

---

## 🧭 Next Steps (v3 Preview)
In **v3**, we’ll:
- Replace the simple MLP with a **true sequence model** (RNN/Transformer-like).  
- Learn longer contexts and capture global dependencies.  
- Explore **weight initialization** and **batch normalization** for stability.

---

## 📚 References
- **Andrej Karpathy** — *Neural Networks: Zero to Hero*  
- GitHub Repo: [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
