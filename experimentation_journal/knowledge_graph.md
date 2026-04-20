# 🧠 NS-ARC Project Knowledge Graph

This graph structurally maps the historical evolution of your Model-JEPA repository, highlighting the major technical bottlenecks we hit, the concepts used to explain them, and the solutions implemented.

```mermaid
graph TD
    %% Styling
    classDef problem stroke-width:3px;
    classDef solution stroke-width:3px;
    classDef concept stroke-dasharray: 5 5;

    %% Root
    ROOT((**NS-ARC Framework**)):::concept

    %% 1. Evolution of the World Model
    subgraph "Architectural Evolution"
        AE[Vanilla Pixel AutoEncoder]
        SLOT_AE[Slot Attention AutoEncoder]
        JEPA[JEPA: Joint Embedding Arch]
        RIJEPA[RiJEPA: Rule-Induction JEPA]
        NS_ARC[Neuro-Symbolic Slotted JEPA]
    end

    ROOT &mdash;> AE
    AE &mdash;>|No Object Segregation| SLOT_AE
    AE &mdash;>|Pixel Generation Hallucination| JEPA
    JEPA &mdash;>|Energy-Based Logical Constraints| RIJEPA
    SLOT_AE &mdash;> NS_ARC
    RIJEPA &mdash;> NS_ARC

    %% 2. Problems & Bottlenecks Experienced
    subgraph "Problems & Bottlenecks"
        P1(Catastrophic Blurriness)
        P2(GPU Server Incompatiblility<br/>RTX 5070 Blackwell)
        P3(Slot Representation Collapse<br/>Epoch 80)
        P4(Continuous Latent Smear<br/>Poor Rules)
    end

    AE -.-> P1
    ROOT -.-> P2
    SLOT_AE -.-> P3
    NS_ARC -.-> P4

    %% 3. Technical Solutions Applied
    subgraph "Engineered Solutions"
        S1{EMA Teacher-Student Mapping}:::solution
        S2{PyTorch Nightly cu128}:::solution
        S3{Alpha Entropy Regularizer & Warmup}:::solution
        S4{Slot-Level Vector Quantization}:::solution
    end

    %% Edge Mappings
    P1 ===>|Solves decoded noise| S1
    S1 ===> JEPA

    P2 ===>|Bridges driver v575 mismatch| S2

    P3 ===>|Protects Harmonic Disentangled Priors| S3
    S3 ===> NS_ARC

    P4 ===>|Forces discrete states| S4
    S4 ===>|Maps to Programmatic Routines| NS_ARC

    %% 4. Theoretical Concepts
    C1[<B>Langevin Grid Sculpting</B><br/>Inference by Latent Backprop]:::concept
    C2[<B>VICReg Variance</B><br/>Stops network lazy collapse]:::concept
    C3[<B>Cross-Attention Context</B><br/>Queries task parameters]:::concept

    RIJEPA -.-> C1
    NS_ARC -.-> C2
    RIJEPA -.-> C3
```

# Core Learnings Summary

### 1. From Generative to Predictive (The JEPA Shift)
We learned that attempting to map ARC geometries via a raw Pixel Autoencoder introduces horrific blurriness. By adopting a **Joint Embedding Predictive Architecture (JEPA)**, we eliminated pixel-generation entirely, mapping hidden geometries against a target via **EMA Teacher-Student** constraints and generating the final grids backwards via **Langevin Dynamics Sculpting**.

### 2. Guarding the Harmonic Disentanglement
We theorized that Slot Attention models fail unless slots start distinctly disjointed. We utilized **Harmonic Priors** over Sine/Cosine waves to flawlessly disentangle them. We learned that the Iterative GRU is structurally violent and will collapse these priors back to an identical background average (Epoch 80 failure) without an opposing **Alpha Entropy Loss** penalizing overlap alongside **Curriculum Warmup** limiting the initial explosion.

### 3. Programmatic Vocabulary
We learned that applying Vector Quantization ($K$) into the 2D spatial dimensions merely creates a color/texture discretizer. To enforce true, rule-based reasoning, $K$-Bottlenecks must be mounted strictly onto the deepest mathematical representations: the `128`-dimensional **Slot Latents**. This forces the latent to assume "Programmatic Routines" (discrete configurations) enabling perfect structural scaling against ARC tasks.
