# NS-ARC: Evidence & Validation Analysis

To meet the standards of a high-tier ML conference (NeurIPS/ICLR), every core claim in your introduction must be anchored by either a foundational study (Literature) or a statistically significant metric from your runs (Internal Evidence).

## 1. Perceptual vs. Logical Dichotomy
| Claim | Evidence Source | Requirement |
| :--- | :--- | :--- |
| **LLMs fail at 2D topology.** | Literature | Cite **Chollet (2019)** regarding "core knowledge" priors. Reference **Webb et al. (2023)** on the performance drop of LLMs in visual-spatial analogical reasoning compared to linguistic logic. |
| **Vision models resist discrete logic.** | Literature | Cite **Greff et al. (2020)** on the "Binding Problem." This identifies the fundamental difficulty of extracting symbols from continuous neural activations. |

## 2. Architectural Novelties
| Claim | Evidence Source | Requirement |
| :--- | :--- | :--- |
| **MSE induces "Catastrophic Blurriness."** | Internal Evidence | **Comparison Plot:** Show a side-by-side reconstruction of an ARC task using a standard Pixel-Decoder vs. our JEPA Sculpting. Report the **MSE gap** (e.g., 0.81 vs 0.99 Accuracy). |
| **Factorization decouples Shape/Pose.** | Internal Evidence | **Ablation Study:** Compare "Factorized" vs "Holistic" slots. Measure the **Mutual Information** between latent codes and grid coordinates. (Lower is better—means shape identity doesn't care where it is). |
| **Slot Attention causes "Collapse."** | Internal Evidence | **Occupancy Metric:** Chart the percentage of slots used per scene. Standard SA often shows 1-2 slots taking the whole grid. Our "Six-Loss" regime should show a distribution matching the true object count in the JSON. |

## 3. Training & Convergence
| Claim | Evidence Source | Requirement |
| :--- | :--- | :--- |
| **Codebook Anchoring stabilizes routing.** | Internal Evidence | **Convergence Speed:** Plot the "Slot Purity" over Epochs. Our model should "lock" into objects faster than a standard unanchored Slot Attention model. |
| **"Six-Loss" Regime is orthogonal.** | Internal Evidence | **Loss Sensitivity:** Show that removing one loss (e.g., Coverage or Diversity) leads to a specific visual failure (e.g., objects being split across slots or multiple objects in one slot). |

## 4. Final Performance (The "Killer" Metric)
| Claim | Evidence Source | Requirement |
| :--- | :--- | :--- |
| **38.7% vs 14.2% Task Success.** | Internal Evidence | **Final Evaluation:** This is your strongest claim. You must provide a table showing the **Success @ 1** and **Success @ 3** metrics on the `arc_data/original/evaluation` set, comparing your model to a standard Vision-Transformer baseline. |

---
> [!IMPORTANT]
> **Action Item for Phase 1 Audit:**
> When you run the `audit_codebook.py` (using the command I gave earlier), pay close attention to the **Utilization** and **Entropy** stats. If entropy is high, the "primitives" are blurry, and we need to increase the `slot_sharpness_loss` weight.
