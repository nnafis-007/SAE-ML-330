"""
Interpretation and Visualization Module

This module provides tools to:
1. Analyze learned features
2. Find which inputs activate specific features
3. Visualize feature activations
4. Identify interpretable patterns

Key interpretability techniques:
- Max activating examples: Find inputs that most strongly activate a feature
- Feature dashboards: Visualize what each feature represents
- Activation distributions: Understand feature usage patterns
- Feature steering: Modify activations to test feature effects
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm.auto import tqdm

from sae_model import SparseAutoencoder


class FeatureAnalyzer:
    """
    Analyze and interpret learned SAE features.
    
    The goal: Understand what concepts each feature represents by:
    1. Finding examples that strongly activate the feature
    2. Looking at patterns in those examples
    3. Testing hypotheses about what the feature detects
    """
    
    def __init__(
        self,
        sae: SparseAutoencoder,
        tokenizer: GPT2Tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the analyzer.
        
        Args:
            sae: Trained Sparse Autoencoder
            tokenizer: GPT-2 tokenizer (for displaying text)
            device: Computing device
        """
        self.sae = sae.to(device)
        self.sae.eval()  # Set to evaluation mode
        self.tokenizer = tokenizer
        self.device = device
        
    @torch.no_grad()
    def get_feature_activations(
        self,
        activations: torch.Tensor,
        feature_idx: int
    ) -> torch.Tensor:
        """
        Get activations for a specific feature across all samples.
        
        Args:
            activations: Input activations, shape (n_samples, d_model)
            feature_idx: Index of the feature to analyze
        
        Returns:
            Feature activation values for all samples
            
        This tells us how strongly the feature responds to each input.
        """
        activations = activations.to(self.device)
        
        # Encode to get all feature activations
        features = self.sae.encode(activations)
        
        # Extract specific feature
        feature_acts = features[:, feature_idx].cpu()
        
        return feature_acts
    
    @torch.no_grad()
    def find_max_activating_examples(
        self,
        activations: torch.Tensor,
        texts: List[str],
        feature_idx: int,
        k: int = 10,
        token_positions: Optional[List[int]] = None
    ) -> List[Tuple[str, float, int]]:
        """
        Find examples that most strongly activate a feature.
        
        Args:
            activations: Input activations
            texts: Corresponding text inputs
            feature_idx: Feature to analyze
            k: Number of top examples to return
            token_positions: Optional list of token positions for each activation
        
        Returns:
            List of (text, activation_value, position) tuples
            
        This is the key interpretability technique:
        - If a feature consistently activates on similar concepts, 
          it's likely detecting that concept
        - Example: Feature might activate on country names, or past tense verbs
        """
        # Get feature activations
        feature_acts = self.get_feature_activations(activations, feature_idx)
        
        # Find top-k activating examples
        top_k_values, top_k_indices = torch.topk(feature_acts, min(k, len(feature_acts)))
        
        results = []
        for value, idx in zip(top_k_values, top_k_indices):
            idx = idx.item()
            text = texts[idx] if idx < len(texts) else "N/A"
            pos = token_positions[idx] if token_positions else idx
            results.append((text, value.item(), pos))
        
        return results
    
    def create_feature_dashboard(
        self,
        activations: torch.Tensor,
        texts: List[str],
        feature_idx: int,
        save_path: Optional[str] = None
    ):
        """
        Create a visualization dashboard for a single feature.
        
        Args:
            activations: Input activations
            texts: Corresponding text inputs
            feature_idx: Feature to analyze
            save_path: Optional path to save the figure
            
        The dashboard shows:
        1. Activation distribution (how often/strongly it fires)
        2. Top activating examples
        3. Statistics about the feature
        """
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Get feature activations
        feature_acts = self.get_feature_activations(activations, feature_idx)
        
        # Plot 1: Activation distribution
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(feature_acts.numpy(), bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel("Activation Value")
        ax1.set_ylabel("Count")
        ax1.set_title(f"Feature {feature_idx} - Activation Distribution")
        ax1.axvline(feature_acts.mean().item(), color='r', linestyle='--', label='Mean')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Statistics
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')
        
        active_frac = (feature_acts > 0).float().mean().item()
        stats_text = f"""
        Feature {feature_idx} Statistics:
        
        Activation Rate: {active_frac:.2%}
        Mean (when active): {feature_acts[feature_acts > 0].mean().item():.4f}
        Max Activation: {feature_acts.max().item():.4f}
        Std Deviation: {feature_acts.std().item():.4f}
        
        Total Samples: {len(feature_acts)}
        Active Samples: {(feature_acts > 0).sum().item()}
        """
        ax2.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center')
        
        # Plot 3: Top activating examples
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        top_examples = self.find_max_activating_examples(
            activations, texts, feature_idx, k=10
        )
        
        examples_text = f"Top 10 Activating Examples:\n\n"
        for i, (text, value, pos) in enumerate(top_examples, 1):
            # Truncate long texts
            display_text = text[:60] + "..." if len(text) > 60 else text
            examples_text += f"{i}. [{value:.3f}] {display_text}\n"
        
        ax3.text(0.05, 0.95, examples_text, fontsize=9, family='monospace',
                verticalalignment='top', wrap=True)
        
        # Plot 4: Activation pattern over samples
        ax4 = fig.add_subplot(gs[2, :])
        sample_indices = np.arange(min(1000, len(feature_acts)))
        ax4.scatter(sample_indices, feature_acts[:len(sample_indices)].numpy(), 
                   alpha=0.3, s=1)
        ax4.set_xlabel("Sample Index")
        ax4.set_ylabel("Activation Value")
        ax4.set_title("Activation Pattern Across Samples")
        ax4.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        
        plt.show()
    
    @torch.no_grad()
    def analyze_feature_correlations(
        self,
        activations: torch.Tensor,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Compute correlation matrix between features.
        
        Args:
            activations: Input activations
            top_k: Analyze correlations among top-k most active features
        
        Returns:
            Correlation matrix
            
        Why this matters:
        - Ideally, features should be uncorrelated (orthogonal)
        - High correlation suggests features are redundant
        - Helps identify which features are independent
        """
        activations = activations.to(self.device)
        features = self.sae.encode(activations)
        
        # Find most active features
        mean_activations = features.mean(dim=0)
        top_features = torch.topk(mean_activations, top_k).indices
        
        # Get activations for these features
        top_feature_acts = features[:, top_features].cpu()
        
        # Compute correlation matrix
        correlation = torch.corrcoef(top_feature_acts.T)
        
        return correlation, top_features
    
    def plot_feature_correlations(
        self,
        activations: torch.Tensor,
        top_k: int = 50,
        save_path: Optional[str] = None
    ):
        """
        Visualize feature correlations as a heatmap.
        
        Args:
            activations: Input activations
            top_k: Number of top features to analyze
            save_path: Optional path to save figure
        """
        correlation, feature_indices = self.analyze_feature_correlations(
            activations, top_k
        )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation.numpy(),
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=feature_indices.numpy(),
            yticklabels=feature_indices.numpy()
        )
        plt.title(f"Feature Correlations (Top {top_k} Features)")
        plt.xlabel("Feature Index")
        plt.ylabel("Feature Index")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation plot saved to {save_path}")
        
        plt.show()
    
    @torch.no_grad()
    def get_reconstruction_quality(
        self,
        activations: torch.Tensor,
        batch_size: int = 1024
    ) -> Dict[str, float]:
        """
        Measure how well the SAE reconstructs activations.
        
        Args:
            activations: Input activations to reconstruct
            batch_size: Process in batches to avoid OOM
        
        Returns:
            Dictionary with reconstruction metrics
            
        Metrics:
        - MSE: Mean squared error
        - Cosine similarity: Direction preservation
        - Explained variance: How much variance is captured
        """
        n = activations.shape[0]
        
        total_mse = 0.0
        total_cos_sim = 0.0
        total_l1 = 0.0
        total_frac_active = 0.0
        
        # For explained variance we need running sums
        sum_var_original = 0.0
        sum_var_residual = 0.0
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = activations[start:end].to(self.device)
            bs = batch.shape[0]
            
            reconstructed, loss, loss_dict = self.sae(batch, return_loss_components=True)
            
            total_mse += loss_dict["mse_loss"] * bs
            total_l1 += loss_dict["l1_loss"] * bs
            total_frac_active += loss_dict["frac_active"] * bs
            
            cos_sim = F.cosine_similarity(batch, reconstructed, dim=1).sum().item()
            total_cos_sim += cos_sim
            
            sum_var_original += batch.var(dim=0).sum().item() * bs
            sum_var_residual += (batch - reconstructed).var(dim=0).sum().item() * bs
            
            del batch, reconstructed
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        mse = total_mse / n
        cos_sim = total_cos_sim / n
        mean_l1 = total_l1 / n
        feature_density = total_frac_active / n
        explained_var = 1 - (sum_var_residual / (sum_var_original + 1e-8))
        
        metrics = {
            "mse": mse,
            "cosine_similarity": cos_sim,
            "explained_variance": explained_var,
            "mean_l1": mean_l1,
            "feature_density": feature_density
        }
        
        return metrics
    
    def analyze_dead_features(
        self,
        activations: torch.Tensor,
        threshold: float = 1e-6,
        freq_threshold: float = 0.001,
        batch_size: int = 1024
    ) -> Dict[str, any]:
        """
        Identify "dead" features that rarely activate.
        
        Args:
            activations: Input activations
            threshold: Minimum mean activation to be considered "alive"
            batch_size: Process activations in batches to avoid OOM on large datasets
        
        Returns:
            Dictionary with dead feature analysis
            
        Dead features are a common problem in SAE training:
        - They learn nothing useful
        - Waste model capacity
        - Can be reinitialized or pruned
        """
        # Process in batches to avoid OOM
        n = activations.shape[0]
        d_hidden = self.sae.d_hidden
        
        sum_acts = torch.zeros(d_hidden, device=self.device)
        max_acts = torch.full((d_hidden,), float('-inf'), device=self.device)
        count_active = torch.zeros(d_hidden, device=self.device)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = activations[start:end].to(self.device)
            features = self.sae.encode(batch)
            
            sum_acts += features.sum(dim=0)
            max_acts = torch.maximum(max_acts, features.max(dim=0).values)
            count_active += (features > 0).float().sum(dim=0)
            
            del features, batch
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        mean_acts = sum_acts / n
        activation_freq = count_active / n

        # Identify dead features (multiple criteria)
        dead_by_mean = mean_acts < threshold
        dead_by_max = max_acts < threshold
        dead_by_freq = activation_freq < freq_threshold

        dead_features = dead_by_mean | dead_by_max | dead_by_freq

        # Helpful breakdown: distinguish truly-never-active from just very rare.
        never_active = activation_freq == 0
        rare_but_nonzero = (activation_freq > 0) & (activation_freq < freq_threshold)
        
        analysis = {
            "num_dead": dead_features.sum().item(),
            "frac_dead": dead_features.float().mean().item(),
            "dead_indices": dead_features.nonzero().squeeze().cpu().tolist(),
            "dead_threshold": threshold,
            "freq_threshold": freq_threshold,
            "num_dead_by_mean": dead_by_mean.sum().item(),
            "num_dead_by_max": dead_by_max.sum().item(),
            "num_dead_by_freq": dead_by_freq.sum().item(),
            "num_never_active": never_active.sum().item(),
            "num_rare_but_nonzero": rare_but_nonzero.sum().item(),
            "mean_activations": mean_acts.cpu(),
            "max_activations": max_acts.cpu(),
            "activation_frequencies": activation_freq.cpu()
        }
        
        return analysis

    @torch.no_grad()
    def prune_dead_features_and_save(
        self,
        activations: torch.Tensor,
        save_path: str,
        threshold: float = 1e-6,
        freq_threshold: float = 0.001
    ):
        """Create and save a pruned SAE with dead features removed.

        This does not change the current SAE in memory.
        """
        analysis = self.analyze_dead_features(activations, threshold=threshold, freq_threshold=freq_threshold)
        dead = analysis["dead_indices"]

        dead_mask = torch.zeros(self.sae.d_hidden, dtype=torch.bool)
        if isinstance(dead, int):
            dead = [dead]
        if isinstance(dead, list) and len(dead) > 0:
            dead_mask[torch.tensor(dead, dtype=torch.long)] = True

        keep_indices = (~dead_mask).nonzero().flatten()
        pruned = self.sae.pruned_copy(keep_indices.to(self.device))

        payload = {
            "model_state_dict": pruned.state_dict(),
            "d_model": pruned.d_model,
            "d_hidden": pruned.d_hidden,
            "l1_coeff": pruned.l1_coeff,
            "keep_indices": keep_indices.cpu(),
            "dead_threshold": analysis.get("dead_threshold", threshold),
            "freq_threshold": analysis.get("freq_threshold", freq_threshold),
        }
        torch.save(payload, save_path)
        print(f"Pruned model saved to {save_path}")

        return pruned
    
    def create_summary_report(
        self,
        activations: torch.Tensor,
        texts: List[str],
        save_path: str = "sae_analysis_report.txt",
        dead_threshold: float = 1e-6,
        freq_threshold: float = 0.001,
    ):
        """
        Generate a comprehensive text report about the SAE.
        
        Args:
            activations: Input activations
            texts: Corresponding texts
            save_path: Path to save report
        """
        report = []
        report.append("="*80)
        report.append("SPARSE AUTOENCODER ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        # Model architecture
        report.append("MODEL ARCHITECTURE")
        report.append("-"*80)
        report.append(f"Input dimension (d_model): {self.sae.d_model}")
        report.append(f"Hidden dimension (d_hidden): {self.sae.d_hidden}")
        report.append(f"Expansion factor: {self.sae.d_hidden / self.sae.d_model:.1f}x")
        report.append(f"L1 coefficient: {self.sae.l1_coeff}")
        report.append(f"Total parameters: {sum(p.numel() for p in self.sae.parameters()):,}")
        report.append("")
        
        # Reconstruction quality
        report.append("RECONSTRUCTION QUALITY")
        report.append("-"*80)
        metrics = self.get_reconstruction_quality(activations)
        for key, value in metrics.items():
            report.append(f"{key}: {value:.6f}")
        report.append("")
        
        # Feature statistics
        report.append("FEATURE STATISTICS")
        report.append("-"*80)
        features = self.sae.encode(activations.to(self.device))
        report.append(f"Mean feature activation: {features.mean().item():.6f}")
        report.append(f"Max feature activation: {features.max().item():.6f}")
        report.append(f"Feature density: {(features > 0).float().mean().item():.2%}")
        report.append("")
        
        # Dead features
        report.append("DEAD FEATURES ANALYSIS")
        report.append("-"*80)
        dead_analysis = self.analyze_dead_features(
            activations,
            threshold=dead_threshold,
            freq_threshold=freq_threshold,
        )
        report.append(f"Number of dead features: {dead_analysis['num_dead']}")
        report.append(f"Fraction of dead features: {dead_analysis['frac_dead']:.2%}")
        report.append(f"Dead thresholds: mean/max < {dead_analysis.get('dead_threshold', 1e-6)} OR freq < {dead_analysis.get('freq_threshold', 0.001)}")
        # Breakdown to help interpret what 'dead' means under the current thresholds
        report.append(
            "Breakdown: "
            f"dead_by_mean={dead_analysis.get('num_dead_by_mean', 0)}, "
            f"dead_by_max={dead_analysis.get('num_dead_by_max', 0)}, "
            f"dead_by_freq={dead_analysis.get('num_dead_by_freq', 0)}, "
            f"never_active={dead_analysis.get('num_never_active', 0)}, "
            f"rare_but_nonzero={dead_analysis.get('num_rare_but_nonzero', 0)}"
        )
        report.append("")
        
        # Most active features
        report.append("TOP 10 MOST ACTIVE FEATURES")
        report.append("-"*80)
        mean_acts = features.mean(dim=0)
        top_features = torch.topk(mean_acts, 10)
        for idx, (feat_idx, act) in enumerate(zip(top_features.indices, top_features.values)):
            feat_idx = feat_idx.item()
            act = act.item()
            freq = (features[:, feat_idx] > 0).float().mean().item()
            report.append(f"{idx+1}. Feature {feat_idx}: mean={act:.4f}, freq={freq:.2%}")
        report.append("")
        
        report.append("="*80)
        
        # Write to file
        report_text = "\n".join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"Report saved to {save_path}")
        print("\n" + report_text)
        
        return report_text


def interactive_feature_explorer(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    texts: List[str],
    tokenizer: GPT2Tokenizer
):
    """
    Interactive tool to explore features.
    
    Args:
        sae: Trained Sparse Autoencoder
        activations: Input activations
        texts: Corresponding texts
        tokenizer: GPT-2 tokenizer
    
    Usage:
        Allows you to interactively query features and see examples.
        Great for exploratory analysis during development.
    """
    analyzer = FeatureAnalyzer(sae, tokenizer)
    
    print("Interactive Feature Explorer")
    print("=" * 60)
    print("Commands:")
    print("  <number>    - Analyze feature by index")
    print("  stats       - Show overall statistics")
    print("  dead        - Show dead features")
    print("  quality     - Show reconstruction quality")
    print("  quit        - Exit")
    print()
    
    while True:
        try:
            command = input("Enter command: ").strip().lower()
            
            if command == "quit":
                break
            
            elif command == "stats":
                features = sae.encode(activations.to(analyzer.device))
                print(f"\nTotal features: {sae.d_hidden}")
                print(f"Mean activation: {features.mean().item():.6f}")
                print(f"Feature density: {(features > 0).float().mean().item():.2%}")
                print()
            
            elif command == "dead":
                dead_analysis = analyzer.analyze_dead_features(activations)
                print(f"\nDead features: {dead_analysis['num_dead']} ({dead_analysis['frac_dead']:.2%})")
                print()
            
            elif command == "quality":
                metrics = analyzer.get_reconstruction_quality(activations)
                print("\nReconstruction Quality:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.6f}")
                print()
            
            elif command.isdigit():
                feature_idx = int(command)
                if 0 <= feature_idx < sae.d_hidden:
                    print(f"\nAnalyzing Feature {feature_idx}...")
                    analyzer.create_feature_dashboard(
                        activations, texts, feature_idx
                    )
                else:
                    print(f"Invalid feature index. Must be 0-{sae.d_hidden-1}")
            
            else:
                print("Unknown command. Try 'help' or a feature number.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    """
    Example usage of interpretation tools
    """
    from sae_model import create_sae_for_gpt2
    from transformers import GPT2Tokenizer
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create and load a trained SAE (replace with your trained model)
    sae = create_sae_for_gpt2("gpt2", expansion_factor=16)
    
    # Generate sample data (replace with real activations)
    n_samples = 1000
    activations = torch.randn(n_samples, 768)
    texts = [f"Sample text {i}" for i in range(n_samples)]
    
    # Create analyzer
    analyzer = FeatureAnalyzer(sae, tokenizer)
    
    # Generate report
    analyzer.create_summary_report(activations, texts)
    
    # Analyze a specific feature
    analyzer.create_feature_dashboard(activations, texts, feature_idx=100)
    
    # Plot feature correlations
    analyzer.plot_feature_correlations(activations, top_k=30)
