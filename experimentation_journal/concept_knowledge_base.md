# 🧠 NS-ARC Project Concept Knowledge Base

This knowledge base serves as a technical lookup dictionary for every major theoretical abstraction, architecture, and mathematical constraint engineered into the NS-ARC framework.

## 1. Primary Architectures
> [!NOTE]
> The evolution of structural backbones managing visual data.

### 1.1 Joint Embedding Predictive Architecture (JEPA)
A non-generative deep learning architecture pioneered by Yann LeCun. Unlike traditional Autoencoders that decode hidden states directly into images (pixels), JEPA forces an encoder predictor to map its outputs strictly in an abstract hidden space against the representations of an Exponential Moving Average (EMA) Teacher. This dramatically prevents visual hallucination and pixel blurriness on exact geometries like ARC grids.

### 1.2 RiJEPA (Rule-Induction JEPA)
Your custom fork of JEPA specifically engineered for the Abstraction and Reasoning Corpus (ARC). It relies on continuous energy limits to penalize rule violations, gracefully defaulting back to Pure JEPA optimization when JSON rule metadata is missing. 

### 1.3 NS-ARC (Neuro-Symbolic ARC)
The apex project structure. Merges purely continuous neural network components (Transformers, Object Encoders) with symbolic programmatic logic gates (Vector Quantized bottlenecks, Slot logic), allowing gradient-based approximation of strict mathematical rules.

### 1.4 Slot Attention / Object-Centric Routing
A method of forcing deep networks to bind information into permutationally invariant chunks (Slots) rather than smeared continuous vectors. Employs an iterative GRU (Gated Recurrent Unit) and softmax attention cross-matching to tease out and bind objects (lines, colors, shapes) independently.

---

## 2. Mathematical Constraints & Regularizers
> [!IMPORTANT]
> The specific loss penalties applied to force the neural networks into proper structural states.

### 2.1 VICReg (Variance-Invariance-Covariance Regularization)
A self-supervised loss mechanism usually used for Siamesee networks. In NS-ARC, the Variance hinge is used specifically to prevent informational collapse within Slot Attention. Without an explicit penalty demanding variance, slots mathematically prefer to fall into safe but useless constants.

### 2.2 EMA Teacher-Student Paradigm
A mechanism where the model is split into an independently learning "Student" and a "Teacher". The Teacher is exclusively evaluated (never gradients) and its weights are updated via an Exponentially Moving Average of the student. 

### 2.3 Alpha Entropy Regularizer
The "Sharpness Regularizer". Slot Attention naturally wishes to blur its assignments mathematically $1/K$ for safety if it experiences harsh loss. Taking the Shannon Entropy of the $\alpha$ assignments and using it as a penalty forces the GRU assignments to polarize toward 1.0 or 0.0, restoring crisp object boundaries and preventing "Epoch 80 Collapse".

### 2.4 Energy-Based Logical Constraints ($\beta$)
A constraint penalty evaluated across a sequence. If certain predicted dimensions violate physical logic rules mapped via datasets, the $\beta$ regularizer heavily penalizes the system's energy. If rules are non-existent, the $\beta$ scales to $0$, avoiding arbitrary systemic crashes.

---

## 3. Discretization & Representation Mechanics
> [!TIP]
> How the network handles memory and dimensional space.

### 3.1 Harmonic Frequency Validation (Harmonic Priors)
Initializing Slot GRU starting locations using heavily scaled combinations of Sine and Cosine sequences (similar to Positional Encodings). By enforcing that every slot initiates from a totally orthogonally separate mathematical vector, we guarantee Disentangled Distributions instead of homogeneous random noise.

### 3.2 Vector Quantization (VQ Bottlenecks)
A hard mathematical cutoff layer that maps continuous vectors directly to their closest discrete neighbor in a pre-learned "Codebook" via nearest-neighbor approximation. The gradients bypass the broken differentiable logic via a Straight-Through Estimator (STE).

### 3.3 Programmatic Routines Hooking
Rather than applying VQ on color/texture space, hooking Vector Quantization $K$-Vocabularies directly to the 128-dimensional Slot latents forces the codebook embeddings to learn abstract, sequential problem-solving logical states (i.e. "Fill Enclosed Mask", "Rotate 180 degrees") instead of visual smears.

---

## 4. Generative Inference Mechanics
> [!CAUTION]
> Navigating from abstract representations back to the physical ARC problem outputs.

### 4.1 Langevin Dynamics Grid Sculpting
Because JEPA refuses to feature a standard pixel decoder, obtaining a classic matrix-grid answer requires generating visuals via physics. By initializing a grid of total random static, passing it into the JEPA to secure an embedding, and gradient-descending *the actual input grid pixels statically* against the target JEPA space, the visual grid crystallizes iteratively out of the noise natively without AutoEncoders.

### 4.2 Cross-Attention Task Conditioning
Forcing the core transformer sequences not just to parse the current state, but actively cross-attending against a stacked tuple of `[Input, Output]` examples simultaneously. Enables zero-shot generalization of abstract mechanics.

---

## 5. Infrastructure & Environments

### 5.1 SSH Port Tunneling (`-N -L`)
Routing an internal server's localhost (e.g. port 8801 in lab networks) via encrypted SSH layers directly to the physical client's localhost without spawning secondary reverse listener nodes.

### 5.2 PyTorch cu128 for Blackwell (`sm_120`)
Newer GPU paradigms utilizing the Blackwell architecture (like the RTX 5070 running Driver 575) structurally reject execution code mapped underneath CUDA Toolkit v12.8, requiring Nightly PyTorch compilation overrides to compile kernels organically.
