import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_rollout_diagnostics(predicted_rewards: list, actual_rewards: list):
    """Visualizes dynamics model accuracy by comparing predicted vs real environmental rewards over a horizon sequence."""
    plt.figure(figsize=(10, 4))
    plt.plot(actual_rewards, label="Actual Truth Reward", color='dodgerblue', linewidth=2)
    plt.plot(predicted_rewards, label="World Model Prediction", color='darkorange', linestyle='dashed', linewidth=2)
    plt.title("World Model Rollout Diagnostics")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_model_error_heatmap(errors_2d: np.ndarray):
    """
    Constructs a spatial representation of domain zones where mathematical dynamics prediction error is structurally high.
    Requires state-compression into a 2D integer mapping matrix beforehand.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(errors_2d, cmap="Reds", annot=False)
    plt.title("Dynamics Model Epistemic Error Heatmap")
    plt.xlabel("Spatial/Abstract X")
    plt.ylabel("Spatial/Abstract Y")
    plt.show()

def plot_ensemble_disagreement(variances: list):
    """Tracks intrinsic disagreement variance triggered by a designated Ensemble across multiple models."""
    plt.figure(figsize=(10, 4))
    plt.fill_between(range(len(variances)), 0, variances, color='mediumpurple', alpha=0.3)
    plt.plot(variances, color='indigo', linewidth=2)
    plt.title("Epistemic Uncertainty Tracking (Ensemble Variance)")
    plt.xlabel("Trajectory Timesteps")
    plt.ylabel("Disagreement Variance Scale")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_rewards(extrinsic: list, intrinsic: list):
    """Identifies correlation structures between extrinsic environment-governed goals vs curiosity injection scaling."""
    plt.figure(figsize=(10, 4))
    plt.plot(extrinsic, label="Extrinsic Raw Reward", color='mediumseagreen', linewidth=2)
    plt.plot(intrinsic, label="Intrinsic Curiosity Injection", color='crimson', alpha=0.7, linestyle='-.')
    plt.title("Reward Matrix Components Profile")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_policy_exploration(unique_states_count: list, entropy: list):
    """
    Crucial dual-axis framework monitoring Proximal Policy Optimization (PPO) exploration physics.
    Validates structural Entropy degradation against progressive structural space discovery.
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    color = 'tab:blue'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Cumulative Unique States Visited', color=color)
    ax1.plot(unique_states_count, color=color, label='Exploration Count', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Policy Categorical Entropy', color=color)
    ax2.plot(entropy, color=color, linestyle='--', label='Action Entropy', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title("Policy Exploration Mechanics & Entropy Decay Thresholds")
    plt.show()
