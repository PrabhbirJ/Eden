"""
Visualization utilities for Trust ABM
Create plots for analysis and paper figures
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


class TrustVisualizer:
    """
    Create visualizations for trust simulation results.
    
    Methods create publication-quality plots similar to Klein & Marx (2018).
    """
    
    def __init__(self, output_dir='figures'):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save figures (default 'figures/')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Figures will be saved to: {self.output_dir.absolute()}")
    
    def plot_single_run_timeseries(self, metrics, params=None, save=True, filename=None):
        """
        Plot time series from a single run.
        
        Shows how trust evolves over time.
        
        Parameters
        ----------
        metrics : MetricsTracker
            Metrics from a single run
        params : dict, optional
            Parameters (for title)
        save : bool, optional
            Save figure to file (default True)
        filename : str, optional
            Output filename
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rounds = metrics.history['round']
        level_of_trust = metrics.history['level_of_trust']
        avg_trust_expectation = metrics.history['avg_trust_expectation']
        
        # Plot both metrics
        ax.plot(rounds, level_of_trust, 'b-', linewidth=2, label='Level of Trust')
        ax.plot(rounds, avg_trust_expectation, 'r--', linewidth=2, label='Avg Trust Expectation')
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Trust Metrics', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Decision Threshold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Title
        if params:
            title = f"Trust Dynamics Over Time\n"
            title += f"(mobility={params.get('mobility', '?')}, "
            title += f"initial_trust={params.get('initial_trust_mean', '?'):.2f}, "
            title += f"share_trustworthy={params.get('share_trustworthy', '?'):.2f})"
            ax.set_title(title, fontsize=13)
        else:
            ax.set_title('Trust Dynamics Over Time', fontsize=13)
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = 'single_run_timeseries.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {filepath}")
        
        return fig
    
    def plot_batch_runs(self, results, params=None, save=True, filename=None):
        """
        Plot results from multiple runs (spaghetti plot).
        
        Shows variation across runs.
        
        Parameters
        ----------
        results : list of MetricsTracker
            Results from multiple runs
        params : dict, optional
            Parameters (for title)
        save : bool, optional
            Save figure
        filename : str, optional
            Output filename
        
        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each run
        for i, metrics in enumerate(results):
            rounds = metrics.history['round']
            level_of_trust = metrics.history['level_of_trust']
            
            # Color based on final outcome
            if metrics.converged_to_trust():
                color = 'blue'
                alpha = 0.3
            elif metrics.converged_to_distrust():
                color = 'red'
                alpha = 0.3
            else:
                color = 'gray'
                alpha = 0.2
            
            ax.plot(rounds, level_of_trust, color=color, alpha=alpha, linewidth=1)
        
        # Plot average
        avg_by_round = {}
        for metrics in results:
            for i, (round_num, level) in enumerate(zip(metrics.history['round'], 
                                                        metrics.history['level_of_trust'])):
                if round_num not in avg_by_round:
                    avg_by_round[round_num] = []
                avg_by_round[round_num].append(level)
        
        rounds_sorted = sorted(avg_by_round.keys())
        avg_level = [np.mean(avg_by_round[r]) for r in rounds_sorted]
        
        ax.plot(rounds_sorted, avg_level, 'k-', linewidth=3, label='Average')
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Level of Trust', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Title
        from metrics import compute_share_trusting
        share_trusting = compute_share_trusting(results)
        
        title = f"Trust Dynamics - {len(results)} Runs (Share Trusting: {share_trusting:.2f})\n"
        if params:
            title += f"(mobility={params.get('mobility', '?')}, "
            title += f"initial_trust={params.get('initial_trust_mean', '?'):.2f})"
        
        ax.set_title(title, fontsize=13)
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = 'batch_runs.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {filepath}")
        
        return fig
    def plot_2d_sweep_multiline(self, sweep_2d, metric='share_trusting', save=True, filename=None):
        """
        Plot 2D sweep as multi-line plot (like Figures 2, 4, 5 from paper).
        
        Creates a plot with:
        - X-axis: param1 values
        - Y-axis: Share of Trusting Sim (or other metric)
        - Multiple lines: one per param2 value (typically mobility)
        
        Parameters
        ----------
        sweep_2d : dict
            Results from run_2d_parameter_sweep()
            Must contain:
            - 'param1_name': str (X-axis variable)
            - 'param1_values': list (X-axis points)
            - 'param2_name': str (line variable, typically 'mobility')
            - 'param2_values': list (creates multiple lines)
            - 'share_trusting': 2D array [param2, param1]
            - 'avg_final_trust': 2D array [param2, param1]
        metric : str, optional
            Which metric to plot: 'share_trusting' or 'avg_final_trust'
            Default: 'share_trusting'
        save : bool, optional
            Save figure to file (default True)
        filename : str, optional
            Output filename (auto-generated if None)
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        
        Examples
        --------
        >>> results = run_2d_parameter_sweep(
        ...     param1_name='initial_trust_mean',
        ...     param1_values=[0.4, 0.5, 0.6, 0.7, 0.8],
        ...     param2_name='mobility',
        ...     param2_values=[1, 2, 5, 10]
        ... )
        >>> viz = TrustVisualizer()
        >>> viz.plot_2d_sweep_multiline(results)
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Extract data
        param1_name = sweep_2d['param1_name']
        param1_values = sweep_2d['param1_values']
        param2_name = sweep_2d['param2_name']
        param2_values = sweep_2d['param2_values']
        
        # Get the data array
        data = sweep_2d[metric]
        
        # Define colors and markers for different lines
        # Using distinct colors for visibility
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        markers = ['o', 's', '^', 'D', 'v', '*', 'p']
        linestyles = ['-', '--', '-.', ':']
        
        # Plot each line (one per param2 value)
        for j, param2_val in enumerate(param2_values):
            color = colors[j % len(colors)]
            marker = markers[j % len(markers)]
            linestyle = linestyles[j % len(linestyles)]
            
            # Format label
            label = f"{param2_name}={param2_val}"
            
            # Plot line
            ax.plot(param1_values, data[j, :], 
                    color=color, 
                    marker=marker, 
                    markersize=8,
                    linewidth=2.5, 
                    linestyle=linestyle,
                    label=label,
                    alpha=0.9)
        
        # Formatting
        # Clean up parameter names for axis labels
        param1_label = param1_name.replace('_', ' ').title()
        
        # Special cases for better labels
        if 'trust' in param1_name.lower():
            if 'mean' in param1_name.lower():
                param1_label = 'Initial Trust Expectation'
        elif 'sensitivity' in param1_name.lower():
            param1_label = 'Weight on New Information (Sensitivity)'
        elif 'trustworthy' in param1_name.lower():
            param1_label = 'Share of Untrustworthy Agents (%)'
        elif 'social_learning' in param1_name.lower():
            param1_label = 'Relative Social Learning Factor'
        
        ax.set_xlabel(param1_label, fontsize=13, fontweight='bold')
        
        # Y-axis label
        if metric == 'share_trusting':
            ax.set_ylabel('Share of Trusting Sim.', fontsize=13, fontweight='bold')
        else:
            ax.set_ylabel('Avg Final Level of Trust', fontsize=13, fontweight='bold')
        
        # Y-axis limits
        ax.set_ylim(-0.05, 1.05)
        
        # Add reference line at 0.5
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=0)
        
        # Legend
        ax.legend(fontsize=11, loc='best', framealpha=0.9, edgecolor='gray')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Title
        title = f"{metric.replace('_', ' ').title()}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        if save:
            if filename is None:
                # Auto-generate filename
                filename = f'figure_{param1_name}_vs_{param2_name}_{metric}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {filepath}")
        
        return fig
    def plot_parameter_sweep(self, sweep_results, save=True, filename=None):
        """
        Plot parameter sweep results (like Figure 2 from paper).
        
        Parameters
        ----------
        sweep_results : dict
            Results from run_parameter_sweep()
        save : bool
            Save figure
        filename : str, optional
            Output filename
        
        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        param_name = sweep_results['param_name']
        param_values = sweep_results['param_values']
        share_trusting = sweep_results['share_trusting']
        avg_final_trust = sweep_results['avg_final_trust']
        
        # Plot 1: Share trusting
        ax1.plot(param_values, share_trusting, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel(param_name.replace('_', ' ').title(), fontsize=12)
        ax1.set_ylabel('Share Trusting', fontsize=12)
        ax1.set_ylim(-0.05, 1.05)
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Share Trusting vs {param_name.replace("_", " ").title()}', fontsize=13)
        
        # Plot 2: Avg final trust
        ax2.plot(param_values, avg_final_trust, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel(param_name.replace('_', ' ').title(), fontsize=12)
        ax2.set_ylabel('Average Final Level of Trust', fontsize=12)
        ax2.set_ylim(-0.05, 1.05)
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Avg Final Trust vs {param_name.replace("_", " ").title()}', fontsize=13)
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f'sweep_{param_name}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {filepath}")
        
        return fig
    
    def plot_phase_diagram(self, sweep_data_x, sweep_data_y, save=True, filename=None):
        """
        Create 2D phase diagram (like Figure 2 from paper).
        
        This requires running sweeps over TWO parameters.
        
        Parameters
        ----------
        sweep_data_x : dict
            Sweep results for parameter 1
        sweep_data_y : dict
            Sweep results for parameter 2
        save : bool
            Save figure
        filename : str, optional
            Output filename
        
        Returns
        -------
        matplotlib.figure.Figure
        """
        # TODO: Implement 2D heatmap
        # This requires coordinated 2D sweep
        # For now, placeholder
        print("Phase diagram plotting not yet implemented")
        print("Requires 2D parameter sweep (will add in next iteration)")
        
        return None


# =============================================================================
# QUICK PLOTTING FUNCTIONS
# =============================================================================

def quick_plot_timeseries(metrics, show=True, save=False):
    """
    Quick plot of time series.
    
    Parameters
    ----------
    metrics : MetricsTracker
        Results to plot
    show : bool
        Show plot (default True)
    save : bool
        Save plot (default False)
    """
    viz = TrustVisualizer()
    fig = viz.plot_single_run_timeseries(metrics, save=save)
    
    if show:
        plt.show()
    
    return fig


def quick_plot_sweep(sweep_results, show=True, save=False):
    """
    Quick plot of parameter sweep.
    
    Parameters
    ----------
    sweep_results : dict
        Sweep results
    show : bool
        Show plot
    save : bool
        Save plot
    """
    viz = TrustVisualizer()
    fig = viz.plot_parameter_sweep(sweep_results, save=save)
    
    if show:
        plt.show()
    
    return fig