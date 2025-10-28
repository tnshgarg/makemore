# 🧠 Makemore — v3: Deep MLP with Batch Normalization

This is **Version 3** of my journey through *Andrej Karpathy's Neural Networks: Zero to Hero* series.  

In this version, I evolve the shallow MLP from v2 into a **deep multi-layer perceptron** with **5 hidden layers** and introduce **Batch Normalization** — a critical technique for training deep networks. I also build **PyTorch-like layer classes from scratch** to deeply understand how modern deep learning frameworks work under the hood.

---

## 🚀 What this model does

The goal remains the same:  
> **Generate realistic human names** by learning character-to-character patterns from a dataset of real names.

But now, the model:
- Uses a **much deeper architecture** (6 layers total) to learn more complex patterns
- Employs **Batch Normalization** to stabilize training and enable deeper networks
- Implements **custom layer classes** (Linear, BatchNorm1d, Tanh) that mimic PyTorch's `nn.Module` API

**Generated names (after full training):**
```
kara.
amelie.
dain.
joshlin.
maren.
```

---

## 🧩 Concepts Learned in v3

### 1. **Building PyTorch-Like Layer Classes**

Instead of using PyTorch's built-in layers, I implemented them from scratch:

#### **Linear Layer**
```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
```

**Key concepts:**
- **Weight initialization**: Divided by `√fan_in` (Kaiming initialization) to prevent exploding/vanishing activations
- **Matrix multiplication** (`@`) vs element-wise multiplication (`*`)
- **Bias** is optional and redundant when using BatchNorm

---

#### **BatchNorm1d Layer**

The star of this version! Normalizes activations to maintain mean≈0, std≈1 across layers.

```python
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.gamma = torch.ones(dim)   # learnable scale
        self.beta = torch.zeros(dim)   # learnable shift
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=False)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        # Update running statistics
        if self.training:
            self.running_mean = (1-momentum)*self.running_mean + momentum*xmean.squeeze(0)
            self.running_var = (1-momentum)*self.running_var + momentum*xvar.squeeze(0)
        
        return self.out
```

**Why BatchNorm is crucial:**
- **Stabilizes training**: Keeps activations in a healthy range
- **Enables deeper networks**: Without it, deep networks suffer from vanishing/exploding gradients
- **Acts as regularization**: Adds slight noise from batch statistics
- **Allows higher learning rates**: More stable optimization

**Training vs Inference:**
- **Training mode**: Uses batch statistics (mean/var of current batch)
- **Inference mode**: Uses running statistics (accumulated over training)

---

#### **Tanh Activation**
```python
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
```

**Properties:**
- Squashes values to [-1, 1]
- **Saturation problem**: When |x| > 2, gradients ≈ 0 (vanishing gradients)
- BatchNorm helps prevent saturation by keeping inputs in a reasonable range

---

### 2. **Network Architecture**

**Much deeper than v2!**

```python
layers = [
    Linear(30, 100), BatchNorm1d(100), Tanh(),
    Linear(100, 100), BatchNorm1d(100), Tanh(),
    Linear(100, 100), BatchNorm1d(100), Tanh(),
    Linear(100, 100), BatchNorm1d(100), Tanh(),
    Linear(100, 100), BatchNorm1d(100), Tanh(),
    Linear(100, 27), BatchNorm1d(27),
]
```

**Flow:**
```
Input: 3 chars × 10-dim embeddings = 30 numbers
    ↓
Linear(30→100) → BatchNorm → Tanh
    ↓
Linear(100→100) → BatchNorm → Tanh
    ↓
Linear(100→100) → BatchNorm → Tanh
    ↓
Linear(100→100) → BatchNorm → Tanh
    ↓
Linear(100→100) → BatchNorm → Tanh
    ↓
Linear(100→27) → BatchNorm
    ↓
Output: 27 logits (one per character)
```

**Total parameters:** ~47,000

---

### 3. **Initialization Tricks**

```python
with torch.no_grad():
    # Make last layer less confident
    layers[-1].gamma *= 0.1
```

**Why?**
- At initialization, we want **soft, uncertain predictions**
- If initial predictions are too confident (peaked), gradients become poor
- Scaling down the last BatchNorm layer creates gentler initial outputs

---

### 4. **Training Loop**

```python
for i in range(200000):
    # 1. Sample minibatch
    ix = torch.randint(0, Xtr.shape[0], (32,))
    Xb, Yb = Xtr[ix], Ytr[ix]
    
    # 2. Forward pass
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)
    
    # 3. Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # 4. Update with learning rate decay
    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
```

**Learning rate schedule:**
- First 150k steps: `lr = 0.1` (explore quickly)
- Last 50k steps: `lr = 0.01` (fine-tune)

**Final loss:** ~1.83 (lower is better!)

---

### 5. **Diagnostic Visualizations**

#### **Activation Distribution**
Checked that Tanh outputs are healthy:
- **Mean ≈ 0**: Balanced (not biased)
- **Std ≈ 0.6-0.7**: Good spread (not too squished)
- **Saturation < 5%**: Not stuck at -1 or 1 (gradients can flow)

```
layer 2 (Tanh): mean -0.00, std 0.63, saturated: 2.78% ✅
layer 5 (Tanh): mean +0.00, std 0.64, saturated: 2.56% ✅
```

**This proves BatchNorm is working!** All layers maintain similar healthy distributions.

---

#### **Gradient Distribution**
Verified gradients flow evenly through all layers:
```
layer 2 (Tanh): std 2.64e-03
layer 5 (Tanh): std 2.25e-03
layer 14 (Tanh): std 1.95e-03
```

Similar magnitudes → no vanishing/exploding gradients!

---

#### **Update-to-Data Ratio**
Monitored the ratio: `(learning_rate × gradient) / parameter`

**Target:** ~10⁻³ (0.001)
- Too small → learning too slowly
- Too large → unstable, overshooting

The plot shows all parameters hover around the -3 line ✅

---

### 6. **Common Issues & Solutions**

#### **Issue 1: NaN in running statistics**
**Symptom:** `running_var` becomes NaN during training

**Cause:** Shape mismatch in exponential moving average
```python
# ❌ WRONG
self.running_mean = (1-momentum)*self.running_mean + momentum*xmean
# xmean has shape (1, 100) but running_mean has shape (100)
```

**Fix:** Squeeze the extra dimension
```python
# ✅ CORRECT
self.running_mean = (1-momentum)*self.running_mean + momentum*xmean.squeeze(0)
```

---

#### **Issue 2: Inference fails with batch_size=1**
**Symptom:** Variance calculation warning during sampling

**Cause:** BatchNorm needs multiple samples to calculate variance

**Fix:** Set layers to eval mode before inference
```python
for layer in layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False
```

In eval mode, BatchNorm uses pre-computed running statistics instead of batch statistics.

---

### 7. **Name Generation (Sampling)**

```python
# Set to eval mode
for layer in layers:
    layer.training = False

# Sample character by character
context = [0, 0, 0]  # start with "..."
while True:
    emb = C[torch.tensor([context])]
    x = emb.view(1, -1)
    for layer in layers:
        x = layer(x)
    probs = F.softmax(x, dim=1)
    
    # Sample next character
    ix = torch.multinomial(probs, num_samples=1).item()
    context = context[1:] + [ix]
    
    if ix == 0:  # stop at '.'
        break
```

---

## 🧮 Technical Summary

| Component | Description |
|-----------|-------------|
| **Architecture** | 6-layer MLP with BatchNorm |
| **Input** | 3 previous characters |
| **Embedding Dim** | 10 |
| **Hidden Layers** | 5 × (Linear→BatchNorm→Tanh), each with 100 neurons |
| **Output Layer** | Linear(100→27) + BatchNorm |
| **Total Parameters** | ~47,000 |
| **Training Steps** | 200,000 |
| **Learning Rate** | 0.1 → 0.01 (decay at 150k) |
| **Batch Size** | 32 |
| **Final Loss** | ~1.83 |
| **Dataset** | 32,000 names from `names.txt` |

---

## 🧠 Key Takeaways

### **1. Batch Normalization is Essential**
- Without BatchNorm, deep networks are extremely hard to train
- Normalizing activations prevents vanishing/exploding gradients
- Enables much deeper architectures and faster training

### **2. Initialization Matters**
- Kaiming initialization (`/ √fan_in`) keeps activations stable
- Scaling down the last layer prevents overconfident initial predictions

### **3. Monitoring is Critical**
- **Activation stats**: Check mean, std, saturation
- **Gradient stats**: Ensure similar magnitudes across layers
- **Update ratios**: Target ~10⁻³ for stable learning

### **4. Training vs Inference**
- BatchNorm behaves differently in training (batch stats) vs inference (running stats)
- Must set `layer.training = False` before sampling

### **5. Building from Scratch Builds Intuition**
- Implementing layers manually reveals how frameworks work
- Understanding the math makes debugging much easier

---

## 🗂️ Project Structure

```
makemore/
│
├── data/
│   └── names.txt                # dataset of names
│
├── v3_deep_mlp_batchnorm/
│   ├── makemore_v3.ipynb        # main notebook
│   ├── README.md                # this file
│   └── output_samples.txt       # generated names
│
└── ...
```

---

## 🐛 Debugging Checklist

If you encounter issues:

1. **Check BatchNorm implementation**
   - Use `keepdim=True` in mean/var calculation
   - Use `unbiased=False` for variance
   - Use `.squeeze(0)` when updating running stats

2. **Verify running statistics are valid**
   ```python
   for layer in layers:
       if isinstance(layer, BatchNorm1d):
           print(torch.isnan(layer.running_mean).any())
   ```

3. **Set eval mode before inference**
   ```python
   for layer in layers:
       layer.training = False
   ```

4. **Monitor training for NaN**
   - Check loss after each step
   - Check gradients before updating

---

## 🧭 Next Steps (v4 Preview)

In **v4**, we'll explore:
- **Residual connections (ResNets)** for even deeper networks
- **Layer Normalization** (alternative to BatchNorm)
- **Attention mechanisms** (foundation for Transformers)
- Moving toward **sequence-to-sequence models**

---

## 📚 References

- **Andrej Karpathy** — *Neural Networks: Zero to Hero*  
  - Video: [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)
- **GitHub Repo**: [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
- **Paper**: [Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167) (Ioffe & Szegedy, 2015)

---

## 🎓 Concepts Mastered

✅ Deep neural networks  
✅ Batch Normalization (theory & implementation)  
✅ Weight initialization strategies  
✅ Activation distribution analysis  
✅ Gradient flow visualization  
✅ Training vs inference modes  
✅ PyTorch-like API design  
✅ Debugging deep learning models  

---

**Made with 🧠 and patience during the debugging process!**