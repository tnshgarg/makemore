# 🧠 Makemore — v1: Bigram Language Model

This is **Version 1** of my journey through Andrej Karpathy’s *Neural Networks: Zero to Hero* series.  
In this version, I implemented a **simple bigram character-level language model** from scratch using **Python + PyTorch**, trained on a dataset of human names.

---

## 🚀 What this model does

The goal is to **generate new, realistic-sounding names** by learning letter-to-letter transitions from a text file (`names.txt`).

At its core, the model learns:

> “Given a current character, what is the probability of the next character?”

This version doesn’t use any deep learning yet — only a single **learnable matrix of weights** (`W`) that gets trained via **gradient descent**.  
But it’s the perfect foundation to build intuition before we move on to neural networks (v2, v3, v4…).
---

## 🧩 Concepts learned in v1

### 1. **Data Preparation**
- Loaded the dataset of names and tokenized it into individual characters.
- Added special start (`'.'` or `<S>`) and end (`'.'` or `<E>`) tokens.
- Created mappings:
  - `stoi` (string → index)
  - `itos` (index → string)

### 2. **Bigram Counting (Statistical Model)**
- Counted every letter pair (bigram) across all names.
- Built a **27×27 matrix** `N` where `N[i,j]` = how many times letter *j* follows letter *i*.
- Normalized counts into probabilities `P` using Laplace smoothing.
- Sampled new names purely from this probability matrix.

### 3. **PyTorch Tensors & Autograd**
- Reimplemented the bigram model as a **learnable parameter matrix `W`**.
- Used **one-hot encoding** to represent input characters numerically.
- Computed probabilities with:
  ```
  probs = exp(logits) / sum(exp(logits))
  ```
- Defined a **negative log-likelihood loss** to measure how well the model predicts next letters.
- Used **autograd (`.backward()`)** and **manual gradient updates** to train `W`.

### 4. **Training**
- Ran multiple forward + backward passes to update weights.
- Observed loss decreasing (meaning the model is learning letter transitions).
- Generated names again — this time from a trained model instead of raw counts.

---

## 🧮 Technical Summary

| Component | Description |
|------------|-------------|
| **Input** | Previous character (one-hot encoded, 27-dim) |
| **Model** | Single weight matrix `W` (27×27) |
| **Output** | Probability distribution for next character |
| **Loss** | Negative Log-Likelihood (Cross Entropy equivalent) |
| **Optimizer** | Manual SGD |
| **Dataset** | ~32,000 English names from `names.txt` |

---

## 📊 Sample Generated Output

After training, the model starts generating short, realistic name-like words:

```
morla.
kavon.
emala.
saril.
jodina.
```

These names aren’t perfect, but they already reflect letter patterns found in real names — a huge step from random gibberish!

---

## 🧠 Key Takeaways

- Built first **generative model** from scratch.
- Learned how **probabilistic models** and **softmax** work.
- Understood **autograd** and **gradient descent** in practice.
- Gained intuition for what “training” really means before adding neural layers.

---

## 🗂️ Project Structure

```
makemore/
│
├── data/
│   └── names.txt          # dataset of names
│
├── v1_bigrams/
│   ├── makemore_v1.ipynb  # main notebook
│   ├── README.md           # this file
│   └── output_samples.txt  # generated names (optional)
│
└── ...
```

---

## 🧭 Next Steps (v2 Preview)

In **v2 (MLP)**, we’ll:
- Move from one-letter context → multi-letter context (more memory)
- Add hidden layers (non-linear transformations)
- Learn richer patterns for name generation

This transition marks the start of building a **true neural network**.

---

## 📚 References
- [Andrej Karpathy – Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [GitHub Repo: karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
