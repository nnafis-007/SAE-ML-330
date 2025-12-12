"""
Visualization utilities for understanding SAE architecture and data flow.

Run this script to generate architecture diagrams.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def plot_sae_architecture():
    """
    Visualize the Sparse Autoencoder architecture and data flow.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Overall pipeline
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('SAE Pipeline: GPT-2 to Interpretation', fontsize=16, fontweight='bold', pad=20)
    
    # Step 1: GPT-2
    box1 = FancyBboxPatch((1, 10), 3, 1, boxstyle="round,pad=0.1", 
                          edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(box1)
    ax1.text(2.5, 10.5, 'GPT-2 Model\n(12 layers)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Arrow to activations
    arrow1 = FancyArrowPatch((2.5, 10), (2.5, 8.5), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax1.add_patch(arrow1)
    ax1.text(3.2, 9.2, 'Extract\nLayer 8', ha='left', va='center', fontsize=9)
    
    # Step 2: Activations
    box2 = FancyBboxPatch((1, 7.5), 3, 1, boxstyle="round,pad=0.1",
                         edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax1.add_patch(box2)
    ax1.text(2.5, 8, 'Activations\n(N × 768)', ha='center', va='center',
            fontsize=11, fontweight='bold')
    
    # Arrow to SAE
    arrow2 = FancyArrowPatch((2.5, 7.5), (2.5, 6),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax1.add_patch(arrow2)
    ax1.text(3.2, 6.7, 'Train', ha='left', va='center', fontsize=9)
    
    # Step 3: SAE
    box3 = FancyBboxPatch((0.5, 4), 4, 2, boxstyle="round,pad=0.1",
                         edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax1.add_patch(box3)
    ax1.text(2.5, 5.3, 'Sparse Autoencoder', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax1.text(2.5, 4.8, '768 → 12,288 → 768', ha='center', va='center', fontsize=10)
    ax1.text(2.5, 4.3, 'Loss: MSE + L1', ha='center', va='center', fontsize=9, style='italic')
    
    # Arrow to features
    arrow3 = FancyArrowPatch((2.5, 4), (2.5, 2.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax1.add_patch(arrow3)
    ax1.text(3.2, 3.2, 'Encode', ha='left', va='center', fontsize=9)
    
    # Step 4: Features
    box4 = FancyBboxPatch((1, 1.5), 3, 1, boxstyle="round,pad=0.1",
                         edgecolor='purple', facecolor='plum', linewidth=2)
    ax1.add_patch(box4)
    ax1.text(2.5, 2, 'Sparse Features\n(~1-5% active)', ha='center', va='center',
            fontsize=11, fontweight='bold')
    
    # Arrow to interpretation
    arrow4 = FancyArrowPatch((4.5, 2), (6.5, 2),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax1.add_patch(arrow4)
    ax1.text(5.5, 2.4, 'Analyze', ha='center', va='bottom', fontsize=9)
    
    # Step 5: Interpretation
    box5 = FancyBboxPatch((6.5, 1), 2.5, 2, boxstyle="round,pad=0.1",
                         edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax1.add_patch(box5)
    ax1.text(7.75, 2.3, 'Interpretation', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax1.text(7.75, 1.8, '• Max activating\n  examples\n• Feature patterns\n• Monosemanticity',
            ha='center', va='center', fontsize=8)
    
    # Right plot: SAE architecture detail
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('Sparse Autoencoder Architecture', fontsize=16, fontweight='bold', pad=20)
    
    # Input
    input_neurons = 8
    for i in range(input_neurons):
        y = 10 - i * 1.2
        circle = plt.Circle((2, y), 0.3, color='lightblue', ec='blue', linewidth=2)
        ax2.add_patch(circle)
    ax2.text(2, 11.5, 'Input x\n(d=768)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Hidden layer
    hidden_neurons = 16
    for i in range(hidden_neurons):
        y = 10.5 - i * 0.6
        circle = plt.Circle((5, y), 0.2, color='lightcoral', ec='red', linewidth=1.5)
        ax2.add_patch(circle)
    ax2.text(5, 11.5, 'Features f\n(d=12,288)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax2.text(5, 0.2, 'ReLU(W_enc @ x + b)', ha='center', va='center', fontsize=9, style='italic')
    
    # Output
    for i in range(input_neurons):
        y = 10 - i * 1.2
        circle = plt.Circle((8, y), 0.3, color='lightgreen', ec='green', linewidth=2)
        ax2.add_patch(circle)
    ax2.text(8, 11.5, 'Output x̂\n(d=768)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw connections (sample)
    for i in range(input_neurons):
        for j in range(0, hidden_neurons, 2):
            y1 = 10 - i * 1.2
            y2 = 10.5 - j * 0.6
            ax2.plot([2.3, 4.8], [y1, y2], 'k-', alpha=0.1, linewidth=0.5)
    
    for j in range(0, hidden_neurons, 2):
        for i in range(input_neurons):
            y1 = 10.5 - j * 0.6
            y2 = 10 - i * 1.2
            ax2.plot([5.2, 7.7], [y1, y2], 'k-', alpha=0.1, linewidth=0.5)
    
    # Add labels
    ax2.text(3.5, 5.5, 'W_enc\n(12288 × 768)', ha='center', va='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(6.5, 5.5, 'W_dec\n(768 × 12288)', ha='center', va='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Loss equation
    ax2.text(5, -0.8, 'Loss = ||x - x̂||² + λ||f||₁', ha='center', va='center',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            fontweight='bold')
    ax2.text(5, -1.5, 'Reconstruction   +   Sparsity', ha='center', va='center',
            fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig('sae_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Architecture diagram saved to: sae_architecture.png")
    plt.show()


def plot_training_process():
    """
    Visualize the training process and what to expect.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Simulated training curves
    epochs = np.arange(1, 31)
    
    # Plot 1: Loss curves
    train_loss = 0.5 * np.exp(-epochs/10) + 0.05
    val_loss = 0.5 * np.exp(-epochs/10) + 0.06 + np.random.normal(0, 0.01, 30)
    
    axes[0, 0].plot(epochs, train_loss, label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training Convergence', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Feature density
    density = 10 - 5 * (1 - np.exp(-epochs/8))
    axes[0, 1].plot(epochs, density, color='green', linewidth=2)
    axes[0, 1].axhline(y=3, color='r', linestyle='--', label='Target: 1-5%')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Feature Density (%)', fontsize=11)
    axes[0, 1].set_title('Sparsity Development', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Feature activation distribution
    np.random.seed(42)
    activations = np.random.exponential(0.1, 1000)
    activations[activations > 0.5] = 0  # Enforce sparsity
    
    axes[1, 0].hist(activations, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Activation Value', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Feature Activation Distribution\n(Most features = 0)', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reconstruction quality
    cos_sim = 0.7 + 0.25 * (1 - np.exp(-epochs/5))
    axes[1, 1].plot(epochs, cos_sim, color='purple', linewidth=2)
    axes[1, 1].axhline(y=0.9, color='r', linestyle='--', label='Target: >0.90')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Cosine Similarity', fontsize=11)
    axes[1, 1].set_title('Reconstruction Quality', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim([0.6, 1.0])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_expectations.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Training expectations diagram saved to: training_expectations.png")
    plt.show()


if __name__ == "__main__":
    print("Generating visualization diagrams...\n")
    plot_sae_architecture()
    print()
    plot_training_process()
    print("\n✅ All diagrams generated successfully!")
