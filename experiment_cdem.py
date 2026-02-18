# experiment_cdem.py
"""
CDEM Experiment: Cooperative/trustworthy Agent Memory

Research Question:
Instead of remembering betrayers (DDEM), what if agents remember trustworthy
partners? This could help re-trigger learning even when trust falls.

Key Insight:
- Distrusting agents stop learning (no direct experience)
- But if they remember trustworthy partners, they might trust THEM specifically
- This creates a path back to learning even at low global trust

Design:
- 2D sweep: Mobility x CDEM Size
- At each combination, run N simulations and compute share_trusting
- Compare CDEM conditions against baseline (cdem_size=0)

Parameters varied:
- Mobility: [1, 2, 5, 10]        (lines in plot)
- CDEM Size: [0, 5, 10, 15, 20]  (x-axis, 0 = baseline)
- Initial Trust: [0.5, 0.6]      (separate experiments)

Fixed (from paper, where mobility effect is strongest):
- Share trustworthy: 0.6
- Sensitivity: 0.05
- Grid: 51x51, 1500 agents, 1000 rounds
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from agent import reset_agent_counter, Agent
from metrics import compute_share_trusting, MetricsTracker
from core_engine_extended import ExtendedTrustSimulation
import copy

# =============================================================================
# CDEM AGENT IMPLEMENTATION
# =============================================================================

class AgentWithCDEM(Agent):
    """
    Agent with CDEM (Cooperative/Trustworthy Memory).
    
    Remembers agents who COOPERATED when trusted.
    Will preferentially interact with known cooperators.
    Uses fixed-size deque (FIFO) to store cooperator IDs.
    """
    
    def __init__(self, is_trustworthy, initial_trust, sensitivity, 
                 trust_threshold=0.5, cdem_size=10):
        """
        Parameters
        ----------
        cdem_size : int
            Maximum number of cooperators to remember (deque size)
        """
        super().__init__(is_trustworthy, initial_trust, sensitivity, trust_threshold)
        
        # CDEM: Deque of agent IDs who cooperated with us
        self.cdem = deque(maxlen=cdem_size)  # Oldest auto-removed when full
        
        # Tracking metrics
        self.n_trusted_cooperators = 0  # How many times we trusted known cooperators
        self.n_cdem_success = 0  # How many times CDEM led to positive experience
        
    def should_trust_this_partner(self, partner):
        """
        Enhanced trust decision that considers CDEM.
        
        Returns True if:
        1. Partner is in CDEM (known cooperator), OR
        2. Standard trust decision (trust_expectation >= threshold)
        """
        # Priority 1: Trust known cooperators even if global trust is low
        if partner.agent_id in self.cdem:
            self.n_trusted_cooperators += 1
            return True
        
        # Priority 2: Standard trust decision
        return self.trust_expectation >= self.trust_threshold
    
    def record_cooperation(self, cooperator):
        """
        Add cooperator to CDEM after successful trust.
        
        Parameters
        ----------
        cooperator : Agent
            The agent who cooperated when trusted
        """
        if cooperator.agent_id not in self.cdem:
            self.cdem.append(cooperator.agent_id)
        else:
            # Move to end (most recent)
            self.cdem.remove(cooperator.agent_id)
            self.cdem.append(cooperator.agent_id)
        
        self.n_cdem_success += 1
    
    def get_cdem_usage(self):
        """Return memory usage statistics."""
        return {
            'cdem_size': len(self.cdem),
            'trusted_cooperators': self.n_trusted_cooperators,
            'cdem_successes': self.n_cdem_success
        }


def create_cdem_agent_population(n_agents, share_trustworthy, initial_trust_mean,
                                  initial_trust_std, sensitivity, trust_threshold=0.5,
                                  cdem_size=10, rng=None):
    """
    Create a population of CDEM agents.
    
    Similar to create_ddem_agent_population but uses AgentWithCDEM.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    agents = []
    for i in range(n_agents):
        is_trustworthy = rng.random() < share_trustworthy
        initial_trust = rng.normal(initial_trust_mean, initial_trust_std)
        
        agent = AgentWithCDEM(
            is_trustworthy=is_trustworthy,
            initial_trust=initial_trust,
            sensitivity=sensitivity,
            trust_threshold=trust_threshold,
            cdem_size=cdem_size
        )
        agents.append(agent)
    
    return agents


# =============================================================================
# CDEM SIMULATION ENGINE
# =============================================================================

class CDEMTrustSimulation(ExtendedTrustSimulation):
    """
    Simulation that uses AgentWithCDEM.
    
    Overrides agent creation and trust game to use CDEM memory.
    """
    
    def __init__(self, params, seed=None, verbose=True,
                 social_learning_factor=1.0,
                 shock_round=None, shock_params=None):
        """
        Initialize CDEM simulation.
        
        params must include 'cdem_size' (int).
        """
        from grid import TrustGrid
        
        self.params = params
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        
        self.grid = TrustGrid(size=params['grid_size'])
        
        cdem_size = params.get('cdem_size', 10)
        
        self.agents = create_cdem_agent_population(
            n_agents=params['n_agents'],
            share_trustworthy=params['share_trustworthy'],
            initial_trust_mean=params['initial_trust_mean'],
            initial_trust_std=params['initial_trust_std'],
            sensitivity=params['sensitivity'],
            trust_threshold=params['trust_threshold'],
            cdem_size=cdem_size,
            rng=self.rng
        )
        
        self.grid.place_agents(self.agents, rng=self.rng)
        self.metrics = MetricsTracker()
        self.current_round = 0
        
        # Extended features
        self.social_learning_factor = social_learning_factor
        self.shock_round = shock_round
        self.shock_params = shock_params or {'min_reduction': 0.0, 'max_reduction': 0.5}
        self.shock_applied = False
        self.shock_stats = None
        self.trustworthy_comparison_history = []
        
        if self.verbose:
            print(f"Initialized CDEM simulation: {len(self.agents)} agents, "
                  f"CDEM size={cdem_size}")
    
    def _play_trust_game(self, trustor, trustee):
        """
        Modified trust game with CDEM memory.
        
        Trustor uses CDEM to decide whether to trust.
        On cooperation, trustee is added to CDEM.
        """
        has_cdem = hasattr(trustor, 'should_trust_this_partner')
        
        # CDEM-aware trust decision
        if has_cdem:
            will_trust = trustor.should_trust_this_partner(trustee)
        else:
            will_trust = trustor.decide_to_trust()
        
        if will_trust:
            trustor.n_times_trusted += 1
            cooperated = trustee.as_trustee_respond()
            
            if cooperated:
                trustor.n_times_cooperated += 1
                direct_info = 1.0
                
                # NEW: Record cooperation in CDEM
                if has_cdem:
                    trustor.record_cooperation(trustee)
            else:
                trustor.n_times_abused += 1
                direct_info = 0.0
            
            # Update trustor's belief
            trustor.update_belief(direct_info)
            
            # Social information for trustee
            social_info = 1.0
            trustee.update_belief(social_info)
        else:
            # Trustor chose not to trust
            social_info = 0.0
            trustee.update_belief(social_info)


# =============================================================================
# METRICS TRACKING FOR CDEM
# =============================================================================

# Extend MetricsTracker to track CDEM stats
original_record = MetricsTracker.record

def record_with_cdem(self, agents, round_num):
    """Extended record method that also tracks CDEM stats."""
    # Call original
    original_record(self, agents, round_num)
    
    # Add CDEM tracking
    if hasattr(agents[0], 'get_cdem_usage'):
        # Initialize history keys if they don't exist
        if 'avg_cdem_size' not in self.history:
            self.history['avg_cdem_size'] = []
        if 'total_trusted_cooperators' not in self.history:
            self.history['total_trusted_cooperators'] = []
        if 'cdem_success_rate' not in self.history:
            self.history['cdem_success_rate'] = []
        
        cdem_sizes = [len(a.cdem) for a in agents]
        trusted_coop = sum(a.n_trusted_cooperators for a in agents)
        successes = sum(a.n_cdem_success for a in agents)
        
        self.history['avg_cdem_size'].append(np.mean(cdem_sizes))
        self.history['total_trusted_cooperators'].append(trusted_coop)
        
        # Success rate: cdem successes / total cooperations
        total_coop = sum(a.n_times_cooperated for a in agents)
        success_rate = successes / max(total_coop, 1)
        self.history['cdem_success_rate'].append(success_rate)

MetricsTracker.record = record_with_cdem


# =============================================================================
# BATCH RUNNER FOR CDEM EXPERIMENTS
# =============================================================================

def run_batch_simulations_cdem(params, n_runs=100, n_rounds=1000,
                                record_interval=10, seed=None, verbose=1):
    """
    Run multiple CDEM simulations with same parameters.
    
    Parameters
    ----------
    params : dict
        Must include 'cdem_size'. If cdem_size=0, uses regular TrustSimulation.
    n_runs : int
    n_rounds : int
    record_interval : int
    seed : int, optional
    verbose : int
        0 = silent
        1 = combo summary only (start + end)
        2 = per-run progress + running stats
    
    Returns
    -------
    list of MetricsTracker
    """
    from core_engine import TrustSimulation
    
    results = []
    cdem_size = params.get('cdem_size', 0)
    mode = f"CDEM (size={cdem_size})" if cdem_size > 0 else "Baseline"
    
    if verbose >= 1:
        print(f"  [{mode}] mobility={params['mobility']}, "
              f"init_trust={params['initial_trust_mean']} | {n_runs} runs x {n_rounds} rounds")
    
    for run_num in range(n_runs):
        reset_agent_counter()
        run_seed = None if seed is None else seed + run_num
        
        if cdem_size > 0:
            sim = CDEMTrustSimulation(params, seed=run_seed, verbose=False)
        else:
            sim = TrustSimulation(params, seed=run_seed, verbose=False)
        
        metrics = sim.run(n_rounds, record_interval)
        results.append(metrics)
        
        if verbose >= 2:
            final = metrics.get_final_level_of_trust()
            outcome = "TRUST" if final > 0.75 else "DISTRUST" if final < 0.25 else "MIXED"
            
            # Running tally
            n_done = run_num + 1
            n_trust = sum(1 for m in results if m.get_final_level_of_trust() > 0.75)
            running_share = n_trust / n_done
            
            # CDEM info
            cdem_info = ""
            if cdem_size > 0 and hasattr(metrics.history, 'get') and 'avg_cdem_size' in metrics.history:
                if metrics.history['avg_cdem_size']:
                    cdem_occ = metrics.history['avg_cdem_size'][-1]
                    trusted = metrics.history['total_trusted_cooperators'][-1]
                    cdem_info = f" | CDEM occ={cdem_occ:.1f}, trusted_coop={trusted}"
            
            print(f"    Run {n_done:3d}/{n_runs}: {outcome:8s} (trust={final:.3f}) "
                  f"| running share_trusting={running_share:.2f}{cdem_info}")
        
        elif verbose >= 1 and (run_num + 1) % 25 == 0:
            n_done = run_num + 1
            n_trust = sum(1 for m in results if m.get_final_level_of_trust() > 0.75)
            print(f"    {n_done}/{n_runs} done... share_trusting so far: {n_trust/n_done:.3f}")
    
    # Final summary
    st = compute_share_trusting(results)
    avg = np.mean([m.get_final_level_of_trust() for m in results])
    
    if verbose >= 1:
        print(f"  => Share trusting: {st:.3f} | Avg final trust: {avg:.3f}")
    
    return results


# =============================================================================
# EXPERIMENTAL PARAMETERS
# =============================================================================

def get_base_params(initial_trust_mean):
    """Get base parameters for a given initial trust level."""
    return {
        'grid_size': 51,
        'n_agents': 1500,
        'share_trustworthy': 0.6,
        'initial_trust_mean': initial_trust_mean,
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,
        'trust_threshold': 0.5,
        'mobility': 5,
        'cdem_size': 0,
    }

MOBILITIES = [2, 5, 10]
CDEM_SIZES = [100,200,1000]
INITIAL_TRUST_LEVELS = [0.5]

N_RUNS = 20
N_ROUNDS = 1000
SEED = 42


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_cdem_experiment_single_trust(initial_trust_mean, n_runs=N_RUNS, 
                                      seed=SEED):
    """
    Run CDEM x Mobility sweep for a SINGLE initial trust level.
    """
    BASE_PARAMS = get_base_params(initial_trust_mean)
    
    results = []
    total = len(MOBILITIES) * len(CDEM_SIZES)
    current = 0
    
    print(f"\nInitial Trust Mean: {initial_trust_mean}")
    print(f"Combinations: {total} | Runs each: {n_runs} | Total sims: {total * n_runs}")
    print("=" * 70)
    
    experiment_start = time.time()
    
    for mobility in MOBILITIES:
        for cdem_size in CDEM_SIZES:
            current += 1
            combo_start = time.time()
            
            print(f"\n{'─'*60}")
            print(f"[{current}/{total}] Trust={initial_trust_mean}, "
                  f"Mobility={mobility}, CDEM={cdem_size}")
            print(f"{'─'*60}")
            
            params = BASE_PARAMS.copy()
            params['mobility'] = mobility
            params['cdem_size'] = cdem_size
            
            sim_results = run_batch_simulations_cdem(
                params,
                n_runs=n_runs,
                n_rounds=N_ROUNDS,
                seed=seed + int(initial_trust_mean * 1000),
                verbose=2
            )
            
            combo_elapsed = time.time() - combo_start
            total_elapsed = time.time() - experiment_start
            remaining = (total_elapsed / current) * (total - current)
            
            # Compute metrics
            share_trusting = compute_share_trusting(sim_results)
            avg_final_trust = np.mean([m.get_final_level_of_trust() for m in sim_results])
            std_final_trust = np.std([m.get_final_level_of_trust() for m in sim_results])
            
            # CDEM-specific metrics
            if cdem_size > 0:
                avg_cdem_occupancy = np.mean([
                    m.history.get('avg_cdem_size', [0])[-1]
                    for m in sim_results
                    if 'avg_cdem_size' in m.history and m.history['avg_cdem_size']
                ]) if sim_results else 0.0
                
                avg_trusted_coop = np.mean([
                    m.history.get('total_trusted_cooperators', [0])[-1]
                    for m in sim_results
                    if 'total_trusted_cooperators' in m.history and m.history['total_trusted_cooperators']
                ]) if sim_results else 0.0
            else:
                avg_cdem_occupancy = 0.0
                avg_trusted_coop = 0.0
            
            results.append({
                'initial_trust_mean': initial_trust_mean,
                'mobility': mobility,
                'cdem_size': cdem_size,
                'share_trusting': share_trusting,
                'avg_final_trust': avg_final_trust,
                'std_final_trust': std_final_trust,
                'avg_cdem_occupancy': avg_cdem_occupancy,
                'avg_trusted_cooperators': avg_trusted_coop,
            })
            
            print(f"\n  Combo time: {combo_elapsed:.0f}s | "
                  f"Elapsed: {total_elapsed/60:.1f}min | "
                  f"ETA: ~{remaining/60:.1f}min")
    
    df = pd.DataFrame(results)
    df = df.sort_values(['mobility', 'cdem_size']).reset_index(drop=True)
    
    total_time = time.time() - experiment_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE for init_trust={initial_trust_mean} | "
          f"Total time: {total_time/60:.1f} minutes")
    print(f"{'=' * 70}")
    
    return df


def run_cdem_experiment_multi_trust(n_runs=N_RUNS, seed=SEED):
    """
    Run CDEM experiment across MULTIPLE initial trust levels.
    """
    all_results = {}
    
    for init_trust in INITIAL_TRUST_LEVELS:
        print("\n" + "=" * 70)
        print(f"STARTING CDEM EXPERIMENTS FOR INITIAL_TRUST_MEAN = {init_trust}")
        print("=" * 70)
        
        df = run_cdem_experiment_single_trust(
            init_trust, 
            n_runs=n_runs, 
            seed=seed
        )
        
        all_results[init_trust] = df
        
        filename = f'cdem_results_trust{init_trust}.csv'
        df.to_csv(filename, index=False)
        print(f"\nSaved results for init_trust={init_trust} to {filename}")
    
    return all_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_cdem_results_multi_trust(all_results, save=True):
    """
    Create comparison plots across different initial trust levels.
    """
    fig, axes = plt.subplots(len(INITIAL_TRUST_LEVELS), 2, 
                             figsize=(14, 6*len(INITIAL_TRUST_LEVELS)))
    
    if len(INITIAL_TRUST_LEVELS) == 1:
        axes = axes.reshape(1, -1)
    
    colors = {1: '#d62728', 2: '#ff7f0e', 5: '#2ca02c', 10: '#1f77b4'}
    markers = {1: 'o', 2: 's', 5: '^', 10: 'D'}
    
    for row_idx, init_trust in enumerate(INITIAL_TRUST_LEVELS):
        df = all_results[init_trust]
        
        # LEFT: Share Trusting vs CDEM Size
        ax = axes[row_idx, 0]
        for mob in MOBILITIES:
            data = df[df['mobility'] == mob].sort_values('cdem_size')
            ax.plot(data['cdem_size'], data['share_trusting'],
                    color=colors[mob], marker=markers[mob], 
                    linewidth=2, markersize=8,
                    label=f'Mobility={mob}')
        ax.set_xlabel('CDEM Size (0 = Baseline)', fontsize=11)
        ax.set_ylabel('Share of Trusting Simulations', fontsize=11)
        ax.set_title(f'Initial Trust = {init_trust}: Trust Emergence (CDEM)', 
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # RIGHT: Benefit over baseline
        ax = axes[row_idx, 1]
        for mob in MOBILITIES:
            data = df[df['mobility'] == mob].sort_values('cdem_size')
            baseline = data[data['cdem_size'] == 0]['share_trusting'].values[0]
            benefit = data['share_trusting'].values - baseline
            ax.plot(data['cdem_size'].values, benefit,
                    color=colors[mob], marker=markers[mob], 
                    linewidth=2, markersize=8,
                    label=f'Mobility={mob}')
        ax.set_xlabel('CDEM Size', fontsize=11)
        ax.set_ylabel('Delta Share Trusting vs Baseline', fontsize=11)
        ax.set_title(f'Initial Trust = {init_trust}: CDEM Benefit', 
                     fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        fig.savefig('cdem_results_multi_trust.png', dpi=300, bbox_inches='tight')
        print("Saved: cdem_results_multi_trust.png")
    
    return fig


def print_summary_multi_trust(all_results):
    """Print formatted summary for all trust levels."""
    for init_trust in INITIAL_TRUST_LEVELS:
        df = all_results[init_trust]
        
        print("\n" + "=" * 70)
        print(f"SUMMARY: Initial Trust = {init_trust} (CDEM)")
        print("=" * 70)
        
        pivot = df.pivot(index='mobility', columns='cdem_size', 
                         values='share_trusting')
        print(f"\n{pivot.to_string(float_format='{:.3f}'.format)}")
        
        print("\n--- KEY FINDINGS ---")
        for mob in MOBILITIES:
            mob_data = df[df['mobility'] == mob]
            baseline = mob_data[mob_data['cdem_size'] == 0]['share_trusting'].values[0]
            cdem_rows = mob_data[mob_data['cdem_size'] > 0]
            
            if len(cdem_rows) > 0:
                best_row = cdem_rows.sort_values('share_trusting', 
                                                  ascending=False).iloc[0]
                benefit = best_row['share_trusting'] - baseline
                direction = ("HELPS" if benefit > 0.05 else 
                            "HURTS" if benefit < -0.05 else "NEUTRAL")
                print(f"  Mobility={mob}: Baseline={baseline:.3f} -> "
                      f"Best CDEM(size={int(best_row['cdem_size'])}): "
                      f"{best_row['share_trusting']:.3f} "
                      f"(delta={benefit:+.3f}) [{direction}]")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        print("*** QUICK MODE: 10 runs ***")
        all_results = run_cdem_experiment_multi_trust(n_runs=10, seed=SEED)
    else:
        all_results = run_cdem_experiment_multi_trust(n_runs=N_RUNS, seed=SEED)
    
    # Save combined results
    combined_df = pd.concat([df.assign(initial_trust=trust) 
                             for trust, df in all_results.items()], 
                            ignore_index=True)
    combined_df.to_csv('cdem_results_all_trust_levels.csv', index=False)
    print("\nSaved: cdem_results_all_trust_levels.csv")
    
    # Print summaries
    print_summary_multi_trust(all_results)
    
    # Plot
    fig = plot_cdem_results_multi_trust(all_results)
    plt.show()