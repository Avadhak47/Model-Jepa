# Remaining Experimentation & Studies for NS-ARC (Clear Accept Target)

Based on the refined NeurIPS/ICLR-level critical analysis, the following tasks must be completed to shift the paper from a "credible empirical study" to a fully validated "Clear Accept." The core focus is now on depth, rigor, and evidential completeness.

## 1. Writing & Scientific Framing (To Be Executed in LaTeX)
**Goal:** Shift from implicit claims to explicit, falsifiable evidence.
- [x] **Operationalize Definitions:** Rewrite claims to include operational definitions and measurement protocols (e.g., define affine invariance as the stability of shape-code assignments under rotation, measured by Hamming distance).
- [x] **Expand Method Section:** Formalize Equivariant Factorization and define the Connected Component Loss explicitly.
- [x] **Improve Experiments Section:** Add sub-sections for "Training Details", "Evaluation Protocol", and "Statistical Significance".
- [x] **Refine Abstract:** Restructure to be less dense: 1 sentence problem, 1 sentence gap, 2 sentences method, 2 sentences results (with numbers), 1 sentence implication.
- [x] **Add Limitations:** Expand the limitations section to include specific failure cases and compute costs.

## 2. End-to-End Validation (Table 2 Completion)
**Goal:** Prove downstream utility with strict evaluation protocols.
- [ ] **Complete Linear Probe Evaluation:**
  - **Define Protocol:** Clearly state the input (latent representation), output (transformation label like rotate, reflect), dataset size, and train/test split.
  - **Metrics Needed:** Accuracy, F1 Score, and Generalization.
  - **Action:** Execute `train_linear_probe.py` and populate the `[XX.X]` placeholders in Table 2.

## 3. Mechanistic Validation (The Core Proofs)
**Goal:** Prove the theoretical claims of the architecture (proto-symbolic vocabulary and equivariance).
- [ ] **Equivariance Behavior Plot:**
  - Apply rotation $\theta$. Measure $\Delta$ shape code (should be near 0) and $\Delta$ pose vector (should correlate with $\theta$).
  - **Action Required:** Calculate the Hamming distance between code assignments before and after transformation to quantify affine invariance.
  - **Visual Required:** Plot rotation angle (x-axis) vs. code change (y-axis).
- [ ] **Codebook Utilization Histograms:**
  - Prove the "proto-symbolic" claim.
  - **Action Required:** Calculate the Shannon entropy of code usage and report the % of dead codes over the validation set.
  - **Visual Required:** Plot histogram of sparse code usage.
- [ ] **Training Dynamics Plots:**
  - Validate the "structural bottlenecks" claim.
  - **Visual Required:** Plot entropy vs. epochs, slot diversity vs. epochs, and codebook stability vs. epochs.

## 4. Generalization Benchmarks (Higher-Level Reasoning)
**Goal:** Move beyond low-level affine transformations to true compositional logic.
- [ ] **High-Level Reasoning Tasks:** 
  - Evaluate the model on at least one higher-level rule-based transformation (e.g., symmetry completion, relational reasoning/object alignment, or pattern repetition).

## 5. Qualitative Visualizations (Non-Negotiable)
**Goal:** Visually demonstrate object-centric extraction quality.
- [ ] **Object-Centric Figure Generation:**
  - Create the grid for Figure 6 showing: Input Grid, Slot Masks, Reconstructed Objects, and Specific Failure Cases (e.g., complex overlaps).

## 6. Diagnostic Component Ablations (Table 3 Completion)
**Goal:** Prove whether the proposed mechanisms are additive or synergistic.
- [ ] **Interaction Evidence:**
  - Execute ablation variants: (1) `- Factorization`, (2) `- Anchoring`, (3) `- Factorization + Anchoring`.
  - Fill the `[XX.X]` placeholders in Table 3.

## 7. Additional Bibliography Enrichment
**Goal:** Strengthen positioning and prevent "incremental" pushback.
- [ ] Add remaining missing citations:
  - **Object-Centric:** MONet (Burgess et al.), GENESIS, SAVi.
  - **Disentanglement:** $\beta$-VAE, Higgins et al.
  - **Compositionality:** SCAN benchmark.
  - **ARC:** ARC-AGI benchmark follow-ups, Chollet generalization papers.
  - **Discrete Representations:** Masked Autoencoders (MAE), DINO.
