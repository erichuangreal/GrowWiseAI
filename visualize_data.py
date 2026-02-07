"""
Data Visualization for Tree Survivability Analysis

This script provides visualization tools to explore your tree data
and understand the relationships between environmental factors and survival rates.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from data_parser import TreeDataParser


class TreeDataVisualizer:
    """
    Visualizer for tree survivability data.
    """
    
    def __init__(self, parser: TreeDataParser):
        """
        Initialize visualizer with a TreeDataParser instance.
        
        Args:
            parser: TreeDataParser instance with loaded data
        """
        self.parser = parser
        self.data = parser.data
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
    def plot_survival_distribution(self, save_path: str = None):
        """
        Plot the distribution of survival rates.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.data['survival_rate'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Survival Rate')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Tree Survival Rates')
        axes[0].axvline(self.data['survival_rate'].mean(), 
                       color='red', linestyle='--', label=f'Mean: {self.data["survival_rate"].mean():.2f}')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(self.data['survival_rate'], vert=True)
        axes[1].set_ylabel('Survival Rate')
        axes[1].set_title('Survival Rate Box Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.show()
        
    def plot_correlation_matrix(self, save_path: str = None):
        """
        Plot correlation matrix of numeric variables.
        """
        # Select only numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix: Environmental Factors & Survival Rate', 
                 fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.show()
        
    def plot_environmental_vs_survival(self, env_columns: list = None, save_path: str = None):
        """
        Plot scatter plots of environmental factors vs survival rate.
        
        Args:
            env_columns: List of environmental column names to plot
            save_path: Path to save the figure
        """
        if env_columns is None:
            # Auto-detect numeric environmental columns
            categorized = self.parser.identify_columns()
            env_columns = [col for col in categorized['environmental'] 
                          if col in self.data.select_dtypes(include=[np.number]).columns]
        
        if not env_columns:
            print("No numeric environmental columns found.")
            return
        
        # Create subplots
        n_cols = 3
        n_rows = (len(env_columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(env_columns):
            row = idx // n_cols
            col_idx = idx % n_cols
            ax = axes[row, col_idx]
            
            # Scatter plot with regression line
            ax.scatter(self.data[col], self.data['survival_rate'], alpha=0.6)
            
            # Add trend line
            z = np.polyfit(self.data[col].dropna(), 
                          self.data.loc[self.data[col].notna(), 'survival_rate'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(self.data[col].min(), self.data[col].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel(col)
            ax.set_ylabel('Survival Rate')
            ax.set_title(f'{col} vs Survival Rate')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(env_columns), n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            axes[row, col_idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.show()
        
    def plot_species_survival(self, save_path: str = None):
        """
        Plot survival rates by tree species (if species column exists).
        """
        if 'species' not in self.data.columns:
            print("No 'species' column found in data.")
            return
        
        # Calculate mean survival rate by species
        species_survival = self.data.groupby('species')['survival_rate'].agg(['mean', 'std', 'count'])
        species_survival = species_survival.sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(species_survival))
        ax.bar(x, species_survival['mean'], yerr=species_survival['std'], 
               capsize=5, alpha=0.7, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(species_survival.index, rotation=45, ha='right')
        ax.set_ylabel('Average Survival Rate')
        ax.set_xlabel('Tree Species')
        ax.set_title('Average Survival Rate by Tree Species')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add sample size annotations
        for i, (species, row) in enumerate(species_survival.iterrows()):
            ax.text(i, row['mean'] + row['std'] + 0.02, f"n={int(row['count'])}", 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.show()
        
    def plot_wildfire_impact(self, save_path: str = None):
        """
        Plot the impact of wildfire history on survival rates.
        """
        if 'wildfire_history' not in self.data.columns:
            print("No 'wildfire_history' column found in data.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot
        self.data.boxplot(column='survival_rate', by='wildfire_history', ax=axes[0])
        axes[0].set_xlabel('Wildfire History')
        axes[0].set_ylabel('Survival Rate')
        axes[0].set_title('Survival Rate by Wildfire History')
        axes[0].get_figure().suptitle('')  # Remove auto title
        
        # Bar plot with counts
        wildfire_stats = self.data.groupby('wildfire_history')['survival_rate'].agg(['mean', 'count'])
        x = range(len(wildfire_stats))
        axes[1].bar(x, wildfire_stats['mean'], alpha=0.7, edgecolor='black')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(wildfire_stats.index)
        axes[1].set_ylabel('Average Survival Rate')
        axes[1].set_xlabel('Wildfire History')
        axes[1].set_title('Average Survival Rate by Wildfire History')
        
        # Add sample size annotations
        for i, (hist, row) in enumerate(wildfire_stats.iterrows()):
            axes[1].text(i, row['mean'] + 0.02, f"n={int(row['count'])}", 
                        ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.show()
        
    def generate_full_report(self, output_dir: str = 'visualizations'):
        """
        Generate a full visualization report with all plots.
        
        Args:
            output_dir: Directory to save all visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Generating visualization report...")
        print("="*60)
        
        # Create all visualizations
        print("\n1. Survival Distribution...")
        self.plot_survival_distribution(save_path=output_path / 'survival_distribution.png')
        
        print("\n2. Correlation Matrix...")
        self.plot_correlation_matrix(save_path=output_path / 'correlation_matrix.png')
        
        print("\n3. Environmental Factors vs Survival...")
        self.plot_environmental_vs_survival(save_path=output_path / 'environmental_vs_survival.png')
        
        print("\n4. Species Survival Comparison...")
        self.plot_species_survival(save_path=output_path / 'species_survival.png')
        
        print("\n5. Wildfire Impact Analysis...")
        self.plot_wildfire_impact(save_path=output_path / 'wildfire_impact.png')
        
        print("\n" + "="*60)
        print(f"✓ All visualizations saved to {output_dir}/")
        print("="*60)


def visualize_tree_data(csv_path: str, generate_report: bool = True):
    """
    Convenience function to load and visualize tree data.
    
    Args:
        csv_path: Path to the CSV file
        generate_report: Whether to generate full visualization report
    
    Returns:
        TreeDataVisualizer instance
    """
    # Load data
    from data_parser import parse_csv
    parser = parse_csv(csv_path, explore=False, clean=False)
    
    # Create visualizer
    visualizer = TreeDataVisualizer(parser)
    
    # Generate report
    if generate_report:
        visualizer.generate_full_report()
    
    return visualizer


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_data.py <path_to_csv>")
        print("\nExample:")
        print("  python visualize_data.py data/tree_data.csv")
        print("\nThis will generate all visualizations in the 'visualizations/' folder")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        visualizer = visualize_tree_data(csv_file, generate_report=True)
        print("\n✓ Visualization complete!")
        
    except FileNotFoundError:
        print(f"\n✗ Error: CSV file not found at '{csv_file}'")
        print("Please check the file path and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)
