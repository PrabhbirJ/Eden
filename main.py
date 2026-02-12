"""
Main script for Trust ABM
Klein & Marx (2018) implementation

This file provides:
1. Quick single-run test
2. Batch simulation
3. Parameter sweep experiments
4. Verification experiments (reproduce paper figures)
"""

from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from config import *
from core_engine import TrustSimulation, run_batch_simulations, run_parameter_sweep
from metrics import compute_share_trusting
import time


# =============================================================================
# SINGLE RUN TEST
# =============================================================================

def test_single_run():
    """
    Quick test: run one simulation and print results.
    
    Use this to verify the model is working.
    """
    print("\n" + "="*60)
    print("SINGLE RUN TEST")
    print("="*60)
    
    # Parameters
    params = {
        'grid_size': GRID_SIZE,
        'n_agents': N_AGENTS,
        'share_trustworthy': SHARE_TRUSTWORTHY,
        'initial_trust_mean': INITIAL_TRUST_MEAN,
        'initial_trust_std': INITIAL_TRUST_STD,
        'sensitivity': SENSITIVITY,
        'mobility': MOBILITY,
        'trust_threshold': TRUST_THRESHOLD
    }
    
    # Create and run
    sim = TrustSimulation(params, seed=42, verbose=True)
    metrics = sim.run(n_rounds=N_ROUNDS, record_interval=RECORD_INTERVAL)
    
    # Print summary
    print("\n" + "-"*60)
    print("SUMMARY")
    print("-"*60)
    summary = metrics.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return metrics


# =============================================================================
# BATCH RUN TEST
# =============================================================================

def test_batch_runs(n_runs=10):
    """
    Test: run multiple simulations with same parameters.
    
    This computes "Share trusting" metric from paper.
    
    Parameters
    ----------
    n_runs : int, optional
        Number of runs (default 10, paper uses 100)
    """
    print("\n" + "="*60)
    print(f"BATCH RUN TEST ({n_runs} runs)")
    print("="*60)
    
    # Parameters
    params = {
        'grid_size': GRID_SIZE,
        'n_agents': N_AGENTS,
        'share_trustworthy': SHARE_TRUSTWORTHY,
        'initial_trust_mean': INITIAL_TRUST_MEAN,
        'initial_trust_std': INITIAL_TRUST_STD,
        'sensitivity': SENSITIVITY,
        'mobility': MOBILITY,
        'trust_threshold': TRUST_THRESHOLD
    }
    
    # Run batch
    start = time.time()
    results = run_batch_simulations(
        params, 
        n_runs=n_runs, 
        n_rounds=N_ROUNDS,
        seed=42,
        verbose=1
    )
    elapsed = time.time() - start
    
    print(f"\nTime elapsed: {elapsed:.1f} seconds ({elapsed/n_runs:.2f} sec/run)")
    
    return results


# =============================================================================
# PARAMETER SWEEP EXPERIMENTS
# =============================================================================

def experiment_mobility_sweep(n_runs=100):
    """
    Verify mobility effect (Figure 2, left side).
    
    Test: Does low mobility reduce trust?
    
    Parameters
    ----------
    n_runs : int, optional
        Runs per parameter value (default 100)
    """
    print("\n" + "="*60)
    print("EXPERIMENT: MOBILITY EFFECT")
    print("="*60)
    
    # Base parameters
    base_params = {
        'grid_size': GRID_SIZE,
        'n_agents': N_AGENTS,
        'share_trustworthy': 0.5,
        'initial_trust_mean': 0.6,
        'initial_trust_std': INITIAL_TRUST_STD,
        'sensitivity': 0.05,
        'mobility': 5,  # Will be varied
        'trust_threshold': TRUST_THRESHOLD,
        'seed': RANDOM_SEED
    }
    
    # Mobility values to test
    mobility_values = [1, 2, 5, 10, 20]
    
    # Run sweep
    results = run_parameter_sweep(
        base_params=base_params,
        param_name='mobility',
        param_values=mobility_values,
        n_runs=n_runs,
        n_rounds=N_ROUNDS,
        seed=42,
        verbose=2,
    )
    
    # Print results
    print("\nRESULTS:")
    print("-"*60)
    print(f"{'Mobility':<12} {'Share Trusting':<20} {'Avg Final Trust':<20}")
    print("-"*60)
    for val, share, avg in zip(results['param_values'], 
                                results['share_trusting'],
                                results['avg_final_trust']):
        print(f"{val:<12} {share:<20.3f} {avg:<20.3f}")
    
    return results


def experiment_initial_trust_sweep(n_runs=100):
    """
    Verify initial trust endowment effect (Figure 2, left side).
    
    Test: Does initial trust matter even after 1000 rounds?
    
    Parameters
    ----------
    n_runs : int, optional
        Runs per parameter value (default 100)
    """
    print("\n" + "="*60)
    print("EXPERIMENT: INITIAL TRUST EFFECT")
    print("="*60)
    
    # Base parameters
    base_params = {
        'grid_size': GRID_SIZE,
        'n_agents': N_AGENTS,
        'share_trustworthy': 0.5,
        'initial_trust_mean': 0.6,  # Will be varied
        'initial_trust_std': INITIAL_TRUST_STD,
        'sensitivity': 0.05,
        'mobility': 5,
        'trust_threshold': TRUST_THRESHOLD
    }
    
    # Initial trust values to test
    initial_trust_values = [0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Run sweep
    results = run_parameter_sweep(
        base_params=base_params,
        param_name='initial_trust_mean',
        param_values=initial_trust_values,
        n_runs=n_runs,
        n_rounds=N_ROUNDS,
        seed=42,
        verbose=2
    )
    
    # Print results
    print("\nRESULTS:")
    print("-"*60)
    print(f"{'Init Trust':<12} {'Share Trusting':<20} {'Avg Final Trust':<20}")
    print("-"*60)
    for val, share, avg in zip(results['param_values'], 
                                results['share_trusting'],
                                results['avg_final_trust']):
        print(f"{val:<12.2f} {share:<20.3f} {avg:<20.3f}")
    
    return results


def experiment_sensitivity_sweep(n_runs=100):
    """
    Verify sensitivity effect (Figure 4, left side).
    
    Test: Does high sensitivity reduce trust?
    
    Parameters
    ----------
    n_runs : int, optional
        Runs per parameter value (default 100)
    """
    print("\n" + "="*60)
    print("EXPERIMENT: SENSITIVITY EFFECT")
    print("="*60)
    
    # Base parameters
    base_params = {
        'grid_size': GRID_SIZE,
        'n_agents': N_AGENTS,
        'share_trustworthy': 0.5,
        'initial_trust_mean': 0.6,
        'initial_trust_std': INITIAL_TRUST_STD,
        'sensitivity': 0.05,  # Will be varied
        'mobility': 5,
        'trust_threshold': TRUST_THRESHOLD
    }
    
    # Sensitivity values to test
    sensitivity_values = [0.03, 0.05, 0.07, 0.1]
    
    # Run sweep
    results = run_parameter_sweep(
        base_params=base_params,
        param_name='sensitivity',
        param_values=sensitivity_values,
        n_runs=n_runs,
        n_rounds=N_ROUNDS,
        seed=42,
        verbose=2
    )
    
    # Print results
    print("\nRESULTS:")
    print("-"*60)
    print(f"{'Sensitivity':<12} {'Share Trusting':<20} {'Avg Final Trust':<20}")
    print("-"*60)
    for val, share, avg in zip(results['param_values'], 
                                results['share_trusting'],
                                results['avg_final_trust']):
        print(f"{val:<12.3f} {share:<20.3f} {avg:<20.3f}")
    
    return results


# =============================================================================
# Plotting
# =============================================================================
def plot_last_results():
    """
    Load and plot the most recent results.
    """
    from export import ResultsExporter
    from visualization import TrustVisualizer
    import os
    
    exporter = ResultsExporter()
    viz = TrustVisualizer()
    
    # Find most recent results directory
    results_dir = Path('results')
    if not results_dir.exists():
        print("No results found. Run an experiment first.")
        return
    
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print("No results found.")
        return
    
    # Get most recent
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    print(f"\nLoading results from: {latest}")
    
    # Check if it's a sweep or batch
    if (latest / 'sweep_results.pkl').exists():
        # It's a sweep
        results = exporter.load_results(latest / 'sweep_results.pkl')
        viz.plot_parameter_sweep(results, save=True)
        plt.show()
    
    elif (latest / 'all_results.pkl').exists():
        # It's a batch
        results = exporter.load_results(latest / 'all_results.pkl')
        viz.plot_batch_runs(results, save=True)
        plt.show()
    
    else:
        print("Unknown result format")

# Add these functions to main.py

def verify_figure_2_left(n_runs=100):
    """
    Reproduce Figure 2 LEFT: Share Trusting vs Initial Trust (multiple mobility lines).
    
    This is THE verification figure.
    """
    print("\n" + "="*60)
    print("VERIFY FIGURE 2 LEFT: Initial Trust × Mobility")
    print("="*60)
    
    from core_engine_extended import run_2d_parameter_sweep
    from visualization import TrustVisualizer
    
    base_params = {
        'grid_size': 51,
        'n_agents': 1500,
        'share_trustworthy': 0.5,
        'initial_trust_mean': 0.6,  # Will be varied
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,
        'mobility': 5,  # Will be varied
        'trust_threshold': 0.5
    }
    
    results = run_2d_parameter_sweep(
        base_params=base_params,
        param1_name='initial_trust_mean',
        param1_values=[0.4, 0.5, 0.6, 0.7, 0.8],
        param2_name='mobility',
        param2_values=[1, 2, 5, 10],
        n_runs=n_runs,
        n_rounds=1000,
        seed=42,
        verbose=1
    )
    
    # Save results
    from export import ResultsExporter
    exporter = ResultsExporter()
    exporter.save_parameter_sweep(results, base_params, experiment_name='figure_2_left')
    
    # Plot
    viz = TrustVisualizer()
    viz.plot_2d_sweep_multiline(results, metric='share_trusting', 
                                 filename='figure_2_left_verification.png')
    plt.show()
    
    return results


def verify_figure_2_right(n_runs=100):
    """
    Reproduce Figure 2 RIGHT: Share Trusting vs Share Untrustworthy (multiple mobility lines).
    """
    print("\n" + "="*60)
    print("VERIFY FIGURE 2 RIGHT: Share Untrustworthy × Mobility")
    print("="*60)
    
    from core_engine_extended import run_2d_parameter_sweep
    from visualization import TrustVisualizer
    
    base_params = {
        'grid_size': 51,
        'n_agents': 1500,
        'share_trustworthy': 0.5,  # Will be varied (note: we vary UNtrustworthy)
        'initial_trust_mean': 0.6,
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,
        'mobility': 5,  # Will be varied
        'trust_threshold': 0.5
    }
    
    # Convert share_untrustworthy to share_trustworthy
    untrustworthy_values = [30, 35, 40, 45, 50, 55, 60]
    trustworthy_values = [(100 - u) / 100 for u in untrustworthy_values]
    
    results = run_2d_parameter_sweep(
        base_params=base_params,
        param1_name='share_trustworthy',
        param1_values=trustworthy_values,
        param2_name='mobility',
        param2_values=[1, 2, 5, 10],
        n_runs=n_runs,
        n_rounds=1000,
        seed=42,
        verbose=1
    )
    
    # Convert X-axis labels back to "share untrustworthy" for plotting
    results['param1_values'] = untrustworthy_values
    results['param1_name'] = 'share_untrustworthy'
    
    # Save & plot
    from export import ResultsExporter
    exporter = ResultsExporter()
    exporter.save_parameter_sweep(results, base_params, experiment_name='figure_2_right')
    
    viz = TrustVisualizer()
    viz.plot_2d_sweep_multiline(results, metric='share_trusting',
                                 filename='figure_2_right_verification.png')
    plt.show()
    
    return results


def verify_figure_4_left(n_runs=100):
    """
    Reproduce Figure 4 LEFT: Share Trusting vs Sensitivity (multiple mobility lines).
    """
    print("\n" + "="*60)
    print("VERIFY FIGURE 4 LEFT: Sensitivity × Mobility")
    print("="*60)
    
    from core_engine_extended import run_2d_parameter_sweep
    from visualization import TrustVisualizer
    
    base_params = {
        'grid_size': 51,
        'n_agents': 1500,
        'share_trustworthy': 0.5,
        'initial_trust_mean': 0.6,
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,  # Will be varied
        'mobility': 5,  # Will be varied
        'trust_threshold': 0.5
    }
    
    results = run_2d_parameter_sweep(
        base_params=base_params,
        param1_name='sensitivity',
        param1_values=[0.04, 0.06, 0.08, 0.10],
        param2_name='mobility',
        param2_values=[1, 2, 5, 10],
        n_runs=n_runs,
        n_rounds=1000,
        seed=42,
        verbose=1
    )
    
    # Save & plot
    from export import ResultsExporter
    exporter = ResultsExporter()
    exporter.save_parameter_sweep(results, base_params, experiment_name='figure_4_left')
    
    viz = TrustVisualizer()
    viz.plot_2d_sweep_multiline(results, metric='share_trusting',
                                 filename='figure_4_left_verification.png')
    plt.show()
    
    return results


def verify_figure_4_right(n_runs=100):
    """
    Reproduce Figure 4 RIGHT: Share Trusting vs Relative Social Learning (multiple mobility lines).
    
    This uses the EXTENDED model with social learning factor.
    """
    print("\n" + "="*60)
    print("VERIFY FIGURE 4 RIGHT: Social Learning × Mobility")
    print("="*60)
    
    from core_engine_extended import run_2d_parameter_sweep
    from visualization import TrustVisualizer
    
    base_params = {
        'grid_size': 51,
        'n_agents': 1500,
        'share_trustworthy': 0.5,
        'initial_trust_mean': 0.6,
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,
        'mobility': 5,  # Will be varied
        'trust_threshold': 0.5,
        'social_learning_factor': 1.0  # Will be varied
    }
    
    results = run_2d_parameter_sweep(
        base_params=base_params,
        param1_name='social_learning_factor',
        param1_values=[0.0, 0.5, 1.0, 1.5, 2.0],
        param2_name='mobility',
        param2_values=[1, 2, 5, 10],
        n_runs=n_runs,
        n_rounds=1000,
        seed=42,
        verbose=1,
        use_social_learning=True  # IMPORTANT!
    )
    
    # Save & plot
    from export import ResultsExporter
    exporter = ResultsExporter()
    exporter.save_parameter_sweep(results, base_params, experiment_name='figure_4_right')
    
    viz = TrustVisualizer()
    viz.plot_2d_sweep_multiline(results, metric='share_trusting',
                                 filename='figure_4_right_verification.png')
    plt.show()
    
    return results


def verify_figure_5_left(n_runs=100):
    """
    Reproduce Figure 5 LEFT: Share Trusting vs Share Untrustworthy WITH SHOCKS.
    """
    print("\n" + "="*60)
    print("VERIFY FIGURE 5 LEFT: Shocks × Untrustworthy × Mobility")
    print("="*60)
    
    from core_engine_extended import run_2d_parameter_sweep
    from visualization import TrustVisualizer
    
    base_params = {
        'grid_size': 51,
        'n_agents': 1500,
        'share_trustworthy': 0.5,  # Will be varied
        'initial_trust_mean': 0.6,
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,
        'mobility': 5,  # Will be varied
        'trust_threshold': 0.5
    }
    
    untrustworthy_values = [30, 35, 40, 45, 50, 55, 60]
    trustworthy_values = [(100 - u) / 100 for u in untrustworthy_values]
    
    results = run_2d_parameter_sweep(
        base_params=base_params,
        param1_name='share_trustworthy',
        param1_values=trustworthy_values,
        param2_name='mobility',
        param2_values=[1, 2, 5, 10],
        n_runs=n_runs,
        n_rounds=1000,
        seed=42,
        verbose=2,
        use_shocks=True  # IMPORTANT! Apply shock at round 200
    )
    
    # Convert labels
    results['param1_values'] = untrustworthy_values
    results['param1_name'] = 'share_untrustworthy'
    
    # Save & plot
    from export import ResultsExporter
    exporter = ResultsExporter()
    exporter.save_parameter_sweep(results, base_params, experiment_name='figure_5_left')
    
    viz = TrustVisualizer()
    viz.plot_2d_sweep_multiline(results, metric='share_trusting',
                                 filename='figure_5_left_verification.png')
    plt.show()
    
    return results


# Update main menu
def main():
    """Main menu."""
    
    print("\n" + "="*60)
    print("TRUST ABM - FULL VERIFICATION")
    print("="*60)
    print("\nSelect experiment:")
    print("  1. Single run test (quick)")
    print("  2. Batch run test (10 runs)")
    print("  3. Figure 2 LEFT: Initial Trust × Mobility")
    print("  4. Figure 2 RIGHT: Share Untrustworthy × Mobility")
    print("  5. Figure 4 LEFT: Sensitivity × Mobility")
    print("  6. Figure 4 RIGHT: Social Learning × Mobility")
    print("  7. Figure 5 LEFT: Shocks × Untrustworthy × Mobility")
    print("  8. Run ALL verification (Figures 2, 4, 5)")
    print("  9. Plot last results")
    print("  0. Exit")
    
    choice = input("\nEnter choice: ").strip()
    
    if choice == '1':
        test_single_run()
    
    elif choice == '2':
        test_batch_runs(n_runs=10)
    
    elif choice == '3':
        verify_figure_2_left(n_runs=100)  # Use 100 for final
    
    elif choice == '4':
        verify_figure_2_right(n_runs=20)
    
    elif choice == '5':
        verify_figure_4_left(n_runs=20)
    
    elif choice == '6':
        verify_figure_4_right(n_runs=20)
    
    elif choice == '7':
        verify_figure_5_left(n_runs=20)
    
    elif choice == '8':
        print("\n" + "="*60)
        print("RUNNING ALL VERIFICATION EXPERIMENTS")
        print("="*60)
        print("\nNOTE: Using n_runs=20 for speed.")
        print("For final thesis, edit code to use n_runs=100\n")
        
        verify_figure_2_left(n_runs=20)
        verify_figure_2_right(n_runs=20)
        verify_figure_4_left(n_runs=20)
        verify_figure_4_right(n_runs=20)
        verify_figure_5_left(n_runs=20)
        
        print("\n" + "="*60)
        print("ALL VERIFICATION COMPLETE")
        print("="*60)
    
    elif choice == '9':
        plot_last_results()
    
    elif choice == '0':
        print("Exiting...")
        return
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()