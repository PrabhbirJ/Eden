# experiment_ddem.py (MODIFIED - Part 1/3)
"""
DDEM Experiment: Extended to test multiple initial trust levels

Now tests at initial_trust_mean = 0.5 AND 0.6
Results are stored separately for each condition
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agent import reset_agent_counter
from metrics import compute_share_trusting
from core_engine_extended import run_batch_simulations_ddem

# =============================================================================
# EXPERIMENTAL PARAMETERS
# =============================================================================

def get_base_params(initial_trust_mean):
    """Get base parameters for a given initial trust level."""
    return {
        'grid_size': 51,
        'n_agents': 1500,
        'share_trustworthy': 0.5,
        'initial_trust_mean': initial_trust_mean,  # NOW PARAMETERIZED
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,
        'trust_threshold': 0.5,
        'mobility': 5,
        'ddem_size': 0,
    }

MOBILITIES = [10]
DDEM_SIZES = [10000]

# NEW: Test multiple initial trust levels
INITIAL_TRUST_LEVELS = [0.6,0.7]

N_RUNS = 20
N_ROUNDS = 1000
SEED = 42

# experiment_ddem.py (MODIFIED - Part 2/3)

def run_ddem_experiment_multi_trust(n_runs=N_RUNS, seed=SEED, skip_baseline=False):
    """
    Run DDEM experiment across MULTIPLE initial trust levels.
    
    Returns
    -------
    dict
        Results keyed by initial_trust_mean, each containing a DataFrame
    """
    all_results = {}
    
    for init_trust in INITIAL_TRUST_LEVELS:
        print("\n" + "=" * 70)
        print(f"STARTING EXPERIMENTS FOR INITIAL_TRUST_MEAN = {init_trust}")
        print("=" * 70)
        
        # Run experiment for this trust level
        df = run_ddem_experiment_single_trust(
            init_trust, 
            n_runs=n_runs, 
            seed=seed, 
            skip_baseline=skip_baseline
        )
        
        # Store results
        all_results[init_trust] = df
        
        # Save intermediate results
        filename = f'ddem_results_trust{init_trust}.csv'
        df.to_csv(filename, index=False)
        print(f"\nSaved results for init_trust={init_trust} to {filename}")
    
    return all_results


def run_ddem_experiment_single_trust(initial_trust_mean, n_runs=N_RUNS, 
                                      seed=SEED, skip_baseline=False):
    """
    Run DDEM x Mobility sweep for a SINGLE initial trust level.
    
    Parameters
    ----------
    initial_trust_mean : float
        Initial trust mean (0.5 or 0.6)
    n_runs : int
        Runs per combination
    seed : int
        Base random seed
    skip_baseline : bool
        If True, skip ddem_size=0 (but we won't for now)
    
    Returns
    -------
    pd.DataFrame
        Results for every (mobility, ddem_size) combination
    """
    BASE_PARAMS = get_base_params(initial_trust_mean)
    
    # For this version, let's run ALL conditions including baseline
    ddem_sizes = DDEM_SIZES
    
    results = []
    total = len(MOBILITIES) * len(ddem_sizes)
    current = 0
    
    print(f"\nInitial Trust Mean: {initial_trust_mean}")
    print(f"Combinations: {total} | Runs each: {n_runs} | Total sims: {total * n_runs}")
    print("=" * 70)
    
    experiment_start = time.time()
    
    for mobility in MOBILITIES:
        for ddem_size in ddem_sizes:
            current += 1
            combo_start = time.time()
            
            print(f"\n{'─'*60}")
            print(f"[{current}/{total}] Trust={initial_trust_mean}, "
                  f"Mobility={mobility}, DDEM={ddem_size}")
            print(f"{'─'*60}")
            
            params = BASE_PARAMS.copy()
            params['mobility'] = mobility
            params['ddem_size'] = ddem_size
            
            sim_results = run_batch_simulations_ddem(
                params,
                n_runs=n_runs,
                n_rounds=N_ROUNDS,
                seed=seed + int(initial_trust_mean * 1000),  # Different seed per trust level
                verbose=2
            )
            
            combo_elapsed = time.time() - combo_start
            total_elapsed = time.time() - experiment_start
            remaining = (total_elapsed / current) * (total - current)
            
            # Compute metrics
            share_trusting = compute_share_trusting(sim_results)
            avg_final_trust = np.mean([m.get_final_level_of_trust() for m in sim_results])
            std_final_trust = np.std([m.get_final_level_of_trust() for m in sim_results])
            
            # DDEM-specific metrics
            if ddem_size > 0:
                avg_ddem_occupancy = np.mean([
                    m.history['avg_ddem_size'][-1]
                    for m in sim_results
                    if m.history['avg_ddem_size']
                ]) if sim_results[0].history['avg_ddem_size'] else 0.0
                
                avg_refused = np.mean([
                    m.history['total_refused_interactions'][-1]
                    for m in sim_results
                    if m.history['total_refused_interactions']
                ]) if sim_results[0].history['total_refused_interactions'] else 0.0
            else:
                avg_ddem_occupancy = 0.0
                avg_refused = 0.0
            
            results.append({
                'initial_trust_mean': initial_trust_mean,
                'mobility': mobility,
                'ddem_size': ddem_size,
                'share_trusting': share_trusting,
                'avg_final_trust': avg_final_trust,
                'std_final_trust': std_final_trust,
                'avg_ddem_occupancy': avg_ddem_occupancy,
                'avg_refused_interactions': avg_refused,
            })
            
            print(f"\n  Combo time: {combo_elapsed:.0f}s | "
                  f"Elapsed: {total_elapsed/60:.1f}min | "
                  f"ETA: ~{remaining/60:.1f}min")
    
    df = pd.DataFrame(results)
    df = df.sort_values(['mobility', 'ddem_size']).reset_index(drop=True)
    
    total_time = time.time() - experiment_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE for init_trust={initial_trust_mean} | "
          f"Total time: {total_time/60:.1f} minutes")
    print(f"{'=' * 70}")
    
    return df

# experiment_ddem.py (MODIFIED - Part 3/3)

def plot_ddem_results_multi_trust(all_results, save=True):
    """
    Create comparison plots across different initial trust levels.
    
    Parameters
    ----------
    all_results : dict
        Dict mapping initial_trust_mean → DataFrame
    """
    fig, axes = plt.subplots(len(INITIAL_TRUST_LEVELS), 2, 
                             figsize=(14, 6*len(INITIAL_TRUST_LEVELS)))
    
    if len(INITIAL_TRUST_LEVELS) == 1:
        axes = axes.reshape(1, -1)
    
    colors = {1: '#d62728', 2: '#ff7f0e', 5: '#2ca02c', 10: '#1f77b4'}
    markers = {1: 'o', 2: 's', 5: '^', 10: 'D'}
    
    for row_idx, init_trust in enumerate(INITIAL_TRUST_LEVELS):
        df = all_results[init_trust]
        
        # LEFT: Share Trusting vs DDEM Size
        ax = axes[row_idx, 0]
        for mob in MOBILITIES:
            data = df[df['mobility'] == mob].sort_values('ddem_size')
            ax.plot(data['ddem_size'], data['share_trusting'],
                    color=colors[mob], marker=markers[mob], 
                    linewidth=2, markersize=8,
                    label=f'Mobility={mob}')
        ax.set_xlabel('DDEM Size (0 = Baseline)', fontsize=11)
        ax.set_ylabel('Share of Trusting Simulations', fontsize=11)
        ax.set_title(f'Initial Trust = {init_trust}: Trust Emergence', 
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # RIGHT: Benefit over baseline
        ax = axes[row_idx, 1]
        for mob in MOBILITIES:
            data = df[df['mobility'] == mob].sort_values('ddem_size')
            baseline = data[data['ddem_size'] == 0]['share_trusting'].values[0]
            benefit = data['share_trusting'].values - baseline
            ax.plot(data['ddem_size'].values, benefit,
                    color=colors[mob], marker=markers[mob], 
                    linewidth=2, markersize=8,
                    label=f'Mobility={mob}')
        ax.set_xlabel('DDEM Size', fontsize=11)
        ax.set_ylabel('Delta Share Trusting vs Baseline', fontsize=11)
        ax.set_title(f'Initial Trust = {init_trust}: Memory Benefit', 
                     fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        fig.savefig('ddem_results_multi_trust.png', dpi=300, bbox_inches='tight')
        print("Saved: ddem_results_multi_trust.png")
    
    return fig


def print_summary_multi_trust(all_results):
    """Print formatted summary for all trust levels."""
    for init_trust in INITIAL_TRUST_LEVELS:
        df = all_results[init_trust]
        
        print("\n" + "=" * 70)
        print(f"SUMMARY: Initial Trust = {init_trust}")
        print("=" * 70)
        
        pivot = df.pivot(index='mobility', columns='ddem_size', 
                         values='share_trusting')
        print(f"\n{pivot.to_string(float_format='{:.3f}'.format)}")
        
        print("\n--- KEY FINDINGS ---")
        for mob in MOBILITIES:
            mob_data = df[df['mobility'] == mob]
            baseline = mob_data[mob_data['ddem_size'] == 0]['share_trusting'].values[0]
            ddem_rows = mob_data[mob_data['ddem_size'] > 0]
            
            if len(ddem_rows) > 0:
                best_row = ddem_rows.sort_values('share_trusting', 
                                                  ascending=False).iloc[0]
                benefit = best_row['share_trusting'] - baseline
                direction = ("HELPS" if benefit > 0.05 else 
                            "HURTS" if benefit < -0.05 else "NEUTRAL")
                print(f"  Mobility={mob}: Baseline={baseline:.3f} -> "
                      f"Best DDEM(size={int(best_row['ddem_size'])}): "
                      f"{best_row['share_trusting']:.3f} "
                      f"(delta={benefit:+.3f}) [{direction}]")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Quick test mode
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        print("*** QUICK MODE: 10 runs, 200 rounds ***")
        all_results = run_ddem_experiment_multi_trust(
            n_runs=10, seed=SEED, skip_baseline=False
        )
    else:
        # Full run
        all_results = run_ddem_experiment_multi_trust(
            n_runs=N_RUNS, seed=SEED, skip_baseline=False
        )
    
    # Save combined results
    combined_df = pd.concat([df.assign(initial_trust=trust) 
                             for trust, df in all_results.items()], 
                            ignore_index=True)
    combined_df.to_csv('ddem_results_all_trust_levels.csv', index=False)
    print("\nSaved: ddem_results_all_trust_levels.csv")
    
    # Print summaries
    print_summary_multi_trust(all_results)
    
    # Plot
    fig = plot_ddem_results_multi_trust(all_results)
    plt.show()