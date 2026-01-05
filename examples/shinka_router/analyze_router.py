#!/usr/bin/env python3
"""
ShinkaRouter Analysis Tools

Visualize evolution progress, Pareto frontier, and routing patterns.
"""

import sqlite3
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class RouterAnalyzer:
    """Analyze ShinkaRouter evolution results."""
    
    def __init__(self, db_path: str = "router_evolution.sqlite"):
        """
        Initialize analyzer.
        
        Args:
            db_path: Path to evolution database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
    def get_all_evaluations(self) -> List[Dict]:
        """Get all evaluations from database."""
        query = """
        SELECT generation, island_id, metrics_json, program_hash
        FROM evaluations
        WHERE metrics_json IS NOT NULL
        ORDER BY generation, island_id
        """
        
        self.cursor.execute(query)
        results = []
        
        for row in self.cursor.fetchall():
            generation, island_id, metrics_json, program_hash = row
            metrics = json.loads(metrics_json)
            
            if 'public' in metrics:
                results.append({
                    'generation': generation,
                    'island_id': island_id,
                    'program_hash': program_hash,
                    'accuracy': metrics['public'].get('accuracy', 0),
                    'avg_calls': metrics['public'].get('avg_calls', 0),
                    'cost': metrics['public'].get('cost', 0),
                    'combined_score': metrics.get('combined_score', 0),
                    'lambda_used': metrics['public'].get('lambda_used', 0),
                })
        
        return results
    
    def plot_pareto_frontier(
        self,
        save_path: Optional[str] = None,
        show_generations: bool = True,
    ):
        """
        Plot Pareto frontier of accuracy vs. efficiency.
        
        Args:
            save_path: Path to save plot (if None, display)
            show_generations: Color points by generation
        """
        data = self.get_all_evaluations()
        
        if not data:
            print("No evaluation data found!")
            return
        
        # Extract data
        calls = [d['avg_calls'] for d in data]
        accuracy = [d['accuracy'] for d in data]
        generations = [d['generation'] for d in data]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if show_generations:
            scatter = ax.scatter(
                calls, accuracy,
                c=generations,
                cmap='viridis',
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5,
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Generation', fontsize=12)
        else:
            ax.scatter(calls, accuracy, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Find Pareto frontier
        pareto_points = self._find_pareto_frontier(calls, accuracy)
        pareto_calls, pareto_acc = zip(*pareto_points)
        
        # Plot Pareto frontier
        ax.plot(
            pareto_calls, pareto_acc,
            'r--', linewidth=2, label='Pareto Frontier', alpha=0.8
        )
        ax.scatter(
            pareto_calls, pareto_acc,
            c='red', s=100, marker='*', edgecolors='black', linewidth=1,
            label='Pareto Optimal', zorder=5
        )
        
        # Highlight best solutions
        best_acc_idx = np.argmax(accuracy)
        best_eff_idx = np.argmin(calls)
        
        ax.scatter(
            calls[best_acc_idx], accuracy[best_acc_idx],
            c='green', s=200, marker='^', edgecolors='black', linewidth=2,
            label=f'Best Accuracy ({accuracy[best_acc_idx]:.1f}%)', zorder=6
        )
        ax.scatter(
            calls[best_eff_idx], accuracy[best_eff_idx],
            c='blue', s=200, marker='v', edgecolors='black', linewidth=2,
            label=f'Most Efficient ({calls[best_eff_idx]:.1f} calls)', zorder=6
        )
        
        # Add baseline reference (if gen 0 exists)
        gen0_data = [d for d in data if d['generation'] == 0]
        if gen0_data:
            baseline_calls = np.mean([d['avg_calls'] for d in gen0_data])
            baseline_acc = np.mean([d['accuracy'] for d in gen0_data])
            ax.scatter(
                baseline_calls, baseline_acc,
                c='orange', s=200, marker='s', edgecolors='black', linewidth=2,
                label=f'Baseline (Gen 0)', zorder=6
            )
            
            # Add reference lines
            ax.axhline(y=baseline_acc, color='orange', linestyle=':', alpha=0.3)
            ax.axvline(x=baseline_calls, color='orange', linestyle=':', alpha=0.3)
        
        ax.set_xlabel('Average LLM Calls per Problem', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('ShinkaRouter: Pareto Frontier (Accuracy vs. Efficiency)', 
                     fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Pareto frontier plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _find_pareto_frontier(
        self,
        calls: List[float],
        accuracy: List[float],
    ) -> List[Tuple[float, float]]:
        """
        Find Pareto frontier (minimize calls, maximize accuracy).
        
        Args:
            calls: List of average calls
            accuracy: List of accuracies
            
        Returns:
            List of (calls, accuracy) tuples on Pareto frontier
        """
        points = list(zip(calls, accuracy))
        pareto = []
        
        for point in points:
            is_dominated = False
            for other in points:
                # other dominates point if:
                # - other has <= calls AND >= accuracy
                # - at least one is strictly better
                if (other[0] <= point[0] and other[1] >= point[1] and
                    (other[0] < point[0] or other[1] > point[1])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto.append(point)
        
        # Sort by calls for plotting
        pareto.sort(key=lambda x: x[0])
        return pareto
    
    def plot_evolution_progress(self, save_path: Optional[str] = None):
        """Plot evolution progress over generations."""
        data = self.get_all_evaluations()
        
        if not data:
            print("No evaluation data found!")
            return
        
        # Group by generation
        gen_stats = defaultdict(lambda: {
            'accuracy': [], 'calls': [], 'score': [], 'lambda': []
        })
        
        for d in data:
            gen = d['generation']
            gen_stats[gen]['accuracy'].append(d['accuracy'])
            gen_stats[gen]['calls'].append(d['avg_calls'])
            gen_stats[gen]['score'].append(d['combined_score'])
            gen_stats[gen]['lambda'].append(d['lambda_used'])
        
        generations = sorted(gen_stats.keys())
        
        # Compute statistics
        max_acc = [max(gen_stats[g]['accuracy']) for g in generations]
        mean_acc = [np.mean(gen_stats[g]['accuracy']) for g in generations]
        min_calls = [min(gen_stats[g]['calls']) for g in generations]
        mean_calls = [np.mean(gen_stats[g]['calls']) for g in generations]
        max_score = [max(gen_stats[g]['score']) for g in generations]
        mean_score = [np.mean(gen_stats[g]['score']) for g in generations]
        lambda_vals = [gen_stats[g]['lambda'][0] for g in generations]  # Same for all in gen
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy over time
        ax = axes[0, 0]
        ax.plot(generations, max_acc, 'g-', linewidth=2, label='Max Accuracy', marker='o')
        ax.plot(generations, mean_acc, 'g--', linewidth=1, label='Mean Accuracy', alpha=0.7)
        ax.fill_between(generations, mean_acc, max_acc, alpha=0.2, color='green')
        ax.set_xlabel('Generation', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Accuracy Over Generations', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Efficiency over time
        ax = axes[0, 1]
        ax.plot(generations, min_calls, 'b-', linewidth=2, label='Min Calls', marker='o')
        ax.plot(generations, mean_calls, 'b--', linewidth=1, label='Mean Calls', alpha=0.7)
        ax.fill_between(generations, min_calls, mean_calls, alpha=0.2, color='blue')
        ax.set_xlabel('Generation', fontweight='bold')
        ax.set_ylabel('Avg LLM Calls', fontweight='bold')
        ax.set_title('Efficiency Over Generations', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Combined score over time
        ax = axes[1, 0]
        ax.plot(generations, max_score, 'r-', linewidth=2, label='Max Score', marker='o')
        ax.plot(generations, mean_score, 'r--', linewidth=1, label='Mean Score', alpha=0.7)
        ax.fill_between(generations, mean_score, max_score, alpha=0.2, color='red')
        ax.set_xlabel('Generation', fontweight='bold')
        ax.set_ylabel('Combined Score', fontweight='bold')
        ax.set_title('Combined Score Over Generations', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Lambda schedule
        ax = axes[1, 1]
        ax.plot(generations, lambda_vals, 'purple', linewidth=2, marker='s')
        ax.set_xlabel('Generation', fontweight='bold')
        ax.set_ylabel('Lambda (Efficiency Penalty)', fontweight='bold')
        ax.set_title('Lambda Schedule', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved evolution progress plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_summary_statistics(self):
        """Print summary statistics of evolution."""
        data = self.get_all_evaluations()
        
        if not data:
            print("No evaluation data found!")
            return
        
        print("\n" + "=" * 80)
        print("ShinkaRouter Evolution Summary")
        print("=" * 80)
        
        # Overall statistics
        all_acc = [d['accuracy'] for d in data]
        all_calls = [d['avg_calls'] for d in data]
        all_scores = [d['combined_score'] for d in data]
        
        print(f"\nTotal Evaluations: {len(data)}")
        print(f"Generations: {max(d['generation'] for d in data) + 1}")
        print(f"Islands: {len(set(d['island_id'] for d in data))}")
        
        print(f"\nAccuracy:")
        print(f"  Best: {max(all_acc):.2f}%")
        print(f"  Mean: {np.mean(all_acc):.2f}%")
        print(f"  Std: {np.std(all_acc):.2f}%")
        
        print(f"\nEfficiency (Avg Calls):")
        print(f"  Best (min): {min(all_calls):.2f}")
        print(f"  Mean: {np.mean(all_calls):.2f}")
        print(f"  Std: {np.std(all_calls):.2f}")
        
        print(f"\nCombined Score:")
        print(f"  Best: {max(all_scores):.4f}")
        print(f"  Mean: {np.mean(all_scores):.4f}")
        print(f"  Std: {np.std(all_scores):.4f}")
        
        # Find Pareto frontier points
        pareto = self._find_pareto_frontier(all_calls, all_acc)
        print(f"\nPareto Frontier:")
        print(f"  Number of Pareto-optimal solutions: {len(pareto)}")
        
        # Best by different criteria
        best_acc_idx = np.argmax(all_acc)
        best_eff_idx = np.argmin(all_calls)
        best_score_idx = np.argmax(all_scores)
        
        print(f"\nBest Solutions:")
        print(f"  Highest Accuracy: {all_acc[best_acc_idx]:.2f}% "
              f"({all_calls[best_acc_idx]:.2f} calls)")
        print(f"  Most Efficient: {all_calls[best_eff_idx]:.2f} calls "
              f"({all_acc[best_eff_idx]:.2f}% accuracy)")
        print(f"  Best Combined Score: {all_scores[best_score_idx]:.4f} "
              f"({all_acc[best_score_idx]:.2f}% accuracy, "
              f"{all_calls[best_score_idx]:.2f} calls)")
        
        # Baseline comparison (gen 0)
        gen0_data = [d for d in data if d['generation'] == 0]
        if gen0_data:
            baseline_acc = np.mean([d['accuracy'] for d in gen0_data])
            baseline_calls = np.mean([d['avg_calls'] for d in gen0_data])
            
            print(f"\nBaseline (Gen 0):")
            print(f"  Accuracy: {baseline_acc:.2f}%")
            print(f"  Avg Calls: {baseline_calls:.2f}")
            
            print(f"\nImprovement from Baseline:")
            print(f"  Accuracy: {max(all_acc) - baseline_acc:+.2f}%")
            print(f"  Efficiency: {baseline_calls - min(all_calls):+.2f} fewer calls")
        
        print("=" * 80 + "\n")
    
    def export_best_programs(self, output_dir: str = "best_routers"):
        """Export best programs for analysis."""
        Path(output_dir).mkdir(exist_ok=True)
        
        data = self.get_all_evaluations()
        
        if not data:
            print("No evaluation data found!")
            return
        
        # Find best programs
        best_acc_idx = np.argmax([d['accuracy'] for d in data])
        best_eff_idx = np.argmin([d['avg_calls'] for d in data])
        best_score_idx = np.argmax([d['combined_score'] for d in data])
        
        best_programs = {
            'best_accuracy': data[best_acc_idx]['program_hash'],
            'most_efficient': data[best_eff_idx]['program_hash'],
            'best_combined': data[best_score_idx]['program_hash'],
        }
        
        print(f"\nBest program hashes:")
        for name, hash_val in best_programs.items():
            print(f"  {name}: {hash_val}")
        
        print(f"\nLook for these programs in your evolution results directory!")
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze ShinkaRouter evolution results")
    parser.add_argument(
        "--db_path",
        type=str,
        default="router_evolution.sqlite",
        help="Path to evolution database",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of saving",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if not args.show:
        Path(args.output_dir).mkdir(exist_ok=True)
    
    # Create analyzer
    analyzer = RouterAnalyzer(args.db_path)
    
    # Print summary
    analyzer.print_summary_statistics()
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    pareto_path = None if args.show else f"{args.output_dir}/pareto_frontier.png"
    analyzer.plot_pareto_frontier(save_path=pareto_path)
    
    progress_path = None if args.show else f"{args.output_dir}/evolution_progress.png"
    analyzer.plot_evolution_progress(save_path=progress_path)
    
    # Export best programs
    analyzer.export_best_programs()
    
    analyzer.close()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()