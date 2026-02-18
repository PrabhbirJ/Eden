# experiment_ddem_decay.py
"""
DDEM with Decay Experiment: Memory Fading Over Time

Research Question:
The original DDEM never forgets. But what if memories fade?
We implement a rolling window: if a betrayer isn't encountered again,
eventually they "age out" of memory.

Key Mechanism:
- DDEM originally uses deque with maxlen (FIFO when full)
- Decay adds: if not updated, betrayers age out after K rounds
- Implementation: Rolling window that drops oldest entries

Design:
- 2D sweep: Mobility x DDEM Size (with decay)
- Compare against baseline DDEM (no decay)
- At each combination, run N simulations

Parameters varied:
- Mobility: [1, 2, 5, 10]        (lines in plot)
- DDEM Size: [0, 5, 10, 15, 20]  (x-axis, 0 = baseline)
- Initial Trust: [0.5, 0.6]      (separate experiments)

Decay mechanism:
- Every DECAY_INTERVAL rounds, drop entries not recently updated
- Betrayers stay in memory only if encountered within window

Fixed (from paper):
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

# =============================================================================
# DDEM WITH DECAY AGENT IMPLEMENTATION
# =============================================================================

class AgentWithDDEMDecay(Agent):
    """
    Agent with DDEM + Decay (time-limited memory).
    
    Like DDEM but memories fade if not refreshed.
    Uses a rolling window: entries have timestamps, old ones expire.
    """
    
    def __init__(self, is_trustworthy, initial_trust, sensitivity, 
                 trust_threshold=0.5, ddem_size=10, decay_rounds=50):
        """
        Parameters
        ----------
        ddem_size : int
            Maximum number of betrayers to remember
        decay_rounds : int
            How many rounds before a memory expires if not refreshed
        """
        super().__init__(is_trustworthy, initial_trust, sensitivity, trust_threshold)
        
        # DDEM with timestamps: {agent_id: last_betrayal_round}
        self.ddem = {}  # Dict instead of deque for timestamp tracking
        self.ddem_size = ddem_size
        self.decay_rounds = decay_rounds
        self.current_round = 0  # Track current round for decay
        
        # Tracking metrics
        self.n_refused_interactions = 0
        self.n_ddem_hits = 0
        self.n_decayed_entries = 0  # NEW: Track how many entries decayed
        
    def set_current_round(self, round_num):
        """Update current round (called by simulation)."""
        self.current_round = round_num
        self._apply_decay()
    
    def _apply_decay(self):
        """Remove entries older than decay_rounds."""
        cutoff = self.current_round - self.decay_rounds
        
        to_remove = [agent_id for agent_id, last_round in self.ddem.items()
                     if last_round < cutoff]
        
        for agent_id in to_remove:
            del self.ddem[agent_id]
            self.n_decayed_entries += 1
    
    def will_interact_with(self, partner):
        """
        Check if agent will interact with partner.
        
        Returns False if partner is in DDEM (known betrayer).
        """
        if partner.agent_id in self.ddem:
            self.n_refused_interactions += 1
            return False
        return True
    
    def decide_to_trust(self, partner=None):
        """
        Trust decision (only called if will_interact_with returned True).
        """
        return self.trust_expectation >= self.trust_threshold
    
    def record_betrayal(self, betrayer):
        """
        Add/update betrayer in DDEM with current timestamp.
        
        Parameters
        ----------
        betrayer : Agent
            The agent who betrayed our trust
        """
        # Update timestamp (or add new entry)
        self.ddem[betrayer.agent_id] = self.current_round
        self.n_ddem_hits += 1
        
        # Enforce size limit: remove oldest entry if over capacity
        if len(self.ddem) > self.ddem_size:
            # Find oldest entry
            oldest_id = min(self.ddem.keys(), key=lambda k: self.ddem[k])
            del self.ddem[oldest_id]
    
    def get_ddem_usage(self):
        """Return memory usage statistics."""
        return {
            'ddem_size': len(self.ddem),
            'refused_interactions': self.n_refused_interactions,
            'ddem_hits': self.n_ddem_hits,
            'decayed_entries': self.n_decayed_entries
        }


def create_ddem_decay_agent_population(n_agents, share_trustworthy, initial_trust_mean,
                                        initial_trust_std, sensitivity, trust_threshold=0.5,
                                        ddem_size=10, decay_rounds=50, rng=None):
    """
    Create a population of DDEM agents with decay.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    agents = []
    for i in range(n_agents):
        is_trustworthy = rng.random() < share_trustworthy
        initial_trust = rng.normal(initial_trust_mean, initial_trust_std)
        
        agent = AgentWithDDEMDecay(
            is_trustworthy=is_trustworthy,
            initial_trust=initial_trust,
            sensitivity=sensitivity,
            trust_threshold=trust_threshold,
            ddem_size=ddem_size,
            decay_rounds=decay_rounds
        )
        agents.append(agent)
    
    return agents


# =============================================================================
# DDEM DECAY SIMULATION ENGINE
# =============================================================================

class DDEMDecayTrustSimulation(ExtendedTrustSimulation):
    """
    Simulation that uses AgentWithDDEMDecay.
    
    Overrides agent creation and updates agents' round counter for decay.
    """
    
    def __init__(self, params, seed=None, verbose=True,
                 social_learning_factor=1.0,
                 shock_round=None, shock_params=None):
        """
        Initialize DDEM Decay simulation.
        
        params must include 'ddem_size' (int) and 'decay_rounds' (int).
        """
        from grid import TrustGrid
        
        self.params = params
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        
        self.grid = TrustGrid(size=params['grid_size'])
        
        ddem_size = params.get('ddem_size', 10)
        decay_rounds = params.get('decay_rounds', 50)
        
        self.agents = create_ddem_decay_agent_population(
            n_agents=params['n_agents'],
            share_trustworthy=params['share_trustworthy'],
            initial_trust_mean=params['initial_trust_mean'],
            initial_trust_std=params['initial_trust_std'],
            sensitivity=params['sensitivity'],
            trust_threshold=params['trust_threshold'],
            ddem_size=ddem_size,
            decay_rounds=decay_rounds,
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
            print(f"Initialized DDEM Decay simulation: {len(self.agents)} agents, "
                  f"DDEM size={ddem_size}, decay={decay_rounds} rounds")
    
    def run(self, n_rounds, record_interval=10):
        """
        Run simulation with decay updates.
        
        Overrides parent to update round counters for decay.
        """
        # Record initial state
        self.metrics.record(self.agents, round_num=0)
        
        for round_num in range(1, n_rounds + 1):
            self.current_round = round_num
            
            # Update agents' round counter for decay
            for agent in self.agents:
                if hasattr(agent, 'set_current_round'):
                    agent.set_current_round(round_num)
            
            # Phase 1: Interaction
            self._interaction_phase()
            
            # Phase 2: Movement
            self._movement_phase()
            
            # Record metrics
            if round_num % record_interval == 0:
                self.metrics.record(self.agents, round_num)
                
                if self.verbose and round_num % 100 == 0:
                    level = self.metrics.get_final_level_of_trust()
                    print(f"  Round {round_num}/{n_rounds}: Level of trust = {level:.3f}")
        
        # Final recording
        self.metrics.record(self.agents, round_num=n_rounds)
        
        if self.verbose:
            summary = self.metrics.get_summary()
            print(f"\nSimulation complete:")
            print(f"  Final level of trust: {summary['final_level_of_trust']:.3f}")
        
        return self.metrics
    
    def _play_trust_game(self, trustor, trustee):
        """
        Modified trust game with DDEM decay memory.
        """
        has_ddem = hasattr(trustor, 'will_interact_with')
        
        if has_ddem:
            will_interact = trustor.will_interact_with(trustee)
            if not will_interact:
                # Trustor refuses to play
                social_info = 0.0
                trustee.update_belief(social_info)
                return
        
        # Standard trust game
        will_trust = trustor.decide_to_trust(trustee) if has_ddem else trustor.decide_to_trust()
        
        if will_trust:
            trustor.n_times_trusted += 1
            cooperated = trustee.as_trustee_respond()
            
            if cooperated:
                trustor.n_times_cooperated += 1
                direct_info = 1.0
            else:
                trustor.n_times_abused += 1
                direct_info = 0.0
                
                # Record betrayal in DDEM (with timestamp)
                if has_ddem:
                    trustor.record_betrayal(trustee)
            
            trustor.update_belief(direct_info)
            social_info = 1.0
            trustee.update_belief(social_info)
        else:
            social_info = 0.0
            trustee.update_belief(social_info)


# =============================================================================
# METRICS TRACKING FOR DDEM DECAY
# =============================================================================

original_record = MetricsTracker.record

def record_with_ddem_decay(self, agents, round_num):
    """Extended record method that tracks DDEM decay stats."""
    original_record(self, agents, round_num)
    
    if hasattr(agents[0], 'get_ddem_usage'):
        # Initialize history keys if they don't exist
        if 'avg_ddem_size' not in self.history:
            self.history['avg_ddem_size'] = []
        if 'total_refused_interactions' not in self.history:
            self.history['total_refused_interactions'] = []
        if 'total_decayed_entries' not in self.history:
            self.history['total_decayed_entries'] = []
        
        usage = [a.get_ddem_usage() for a in agents]
        
        self.history['avg_ddem_size'].append(np.mean([u['ddem_size'] for u in usage]))
        self.history['total_refused_interactions'].append(sum(u['refused_interactions'] for u in usage))
        self.history['total_decayed_entries'].append(sum(u['decayed_entries'] for u in usage))

MetricsTracker.record = record_with_ddem_decay


# =============================================================================
# BATCH RUNNER FOR DDEM DECAY EXPERIMENTS
# =============================================================================

def run_batch_simulations_ddem_decay(params, n_runs=100, n_rounds=1000,
                                      record_interval=10, seed=None, verbose=1):
    """
    Run multiple DDEM Decay simulations.
    """
    from core_engine import TrustSimulation
    
    results = []
    ddem_size = params.get('ddem_size', 0)
    decay_rounds = params.get('decay_rounds', 50)
    mode = f"DDEM-Decay (size={ddem_size}, decay={decay_rounds})" if ddem_size > 0 else "Baseline"
    
    if verbose >= 1:
        print(f"  [{mode}] mobility={params['mobility']}, "
              f"init_trust={params['initial_trust_mean']} | {n_runs} runs x {n_rounds} rounds")
    
    for run_num in range(n_runs):
        reset_agent_counter()
        run_seed = None if seed is None else seed + run_num
        
        if ddem_size > 0:
            sim = DDEMDecayTrustSimulation(params, seed=run_seed, verbose=False)
        else:
            sim = TrustSimulation(params, seed=run_seed, verbose=False)
        
        metrics = sim.run(n_rounds, record_interval)
        results.append(metrics)
        
        if verbose >= 2:
            final = metrics.get_final_level_of_trust()
            outcome = "TRUST" if final > 0.75 else "DISTRUST" if final < 0.25 else "MIXED"
            
            n_done = run_num + 1
            n_trust = sum(1 for m in results if m.get_final_level_of_trust() > 0.75)
            running_share = n_trust / n_done
            
            # DDEM decay info
            ddem_info = ""
            if ddem_size > 0 and 'avg_ddem_size' in metrics.history:
                if metrics.history['avg_ddem_size']:
                    ddem_occ = metrics.history['avg_ddem_size'][-1]
                    refused = metrics.history['total_refused_interactions'][-1]
                    decayed = metrics.history['total_decayed_entries'][-1]
                    ddem_info = f" | DDEM={ddem_occ:.1f}, refused={refused}, decayed={decayed}"
            
            print(f"    Run {n_done:3d}/{n_runs}: {outcome:8s} (trust={final:.3f}) "
                  f"| running share={running_share:.2f}{ddem_info}")
        
        elif verbose >= 1 and (run_num + 1) % 25 == 0:
            n_done = run_num + 1
            n_trust = sum(1 for m in results if m.get_final_level_of_trust() > 0.75)
            print(f"    {n_done}/{n_runs} done... share_trusting so far: {n_trust/n_done:.3f}")
    
    st = compute_share_trusting(results)
    avg = np.mean([m.get_final_level_of_trust() for m in results])
    
    if verbose >= 1:
        print(f"  => Share trusting: {st:.3f} | Avg final trust: {avg:.3f}")
    
    return results


# =============================================================================
# EXPERIMENTAL PARAMETERS
# =============================================================================

def get_base_params(initial_trust_mean, decay_rounds=50):
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
        'ddem_size': 0,
        'decay_rounds': decay_rounds,
    }

MOBILITIES = [1, ]
DDEM_SIZES = [5, 10, 15]
INITIAL_TRUST_LEVELS = [0.7]
DECAY_ROUNDS = 50  # Memories decay after 50 rounds without refresh

N_RUNS = 20
N_ROUNDS = 1000
SEED = 42


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_ddem_decay_experiment_single_trust(initial_trust_mean, n_runs=N_RUNS, 
                                            seed=SEED, decay_rounds=DECAY_ROUNDS):
    """
    Run DDEM Decay x Mobility sweep for a SINGLE initial trust level.
    """
    BASE_PARAMS = get_base_params(initial_trust_mean, decay_rounds)
    
    results = []
    total = len(MOBILITIES) * len(DDEM_SIZES)
    current = 0
    
    print(f"\nInitial Trust Mean: {initial_trust_mean}, Decay: {decay_rounds} rounds")
    print(f"Combinations: {total} | Runs each: {n_runs} | Total sims: {total * n_runs}")
    print("=" * 70)
    
    experiment_start = time.time()
    
    for mobility in MOBILITIES:
        for ddem_size in DDEM_SIZES:
            current += 1
            combo_start = time.time()
            
            print(f"\n{'─'*60}")
            print(f"[{current}/{total}] Trust={initial_trust_mean}, "
                  f"Mobility={mobility}, DDEM={ddem_size}, Decay={decay_rounds}")
            print(f"{'─'*60}")
            
            params = BASE_PARAMS.copy()
            params['mobility'] = mobility
            params['ddem_size'] = ddem_size
            
            sim_results = run_batch_simulations_ddem_decay(
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
            
            # DDEM decay-specific metrics
            if ddem_size > 0:
                avg_ddem_occupancy = np.mean([
                    m.history['avg_ddem_size'][-1]
                    for m in sim_results
                    if m.history.get('avg_ddem_size')
                ]) if sim_results else 0.0
                
                avg_refused = np.mean([
                    m.history['total_refused_interactions'][-1]
                    for m in sim_results
                    if m.history.get('total_refused_interactions')
                ]) if sim_results else 0.0
                
                avg_decayed = np.mean([
                    m.history['total_decayed_entries'][-1]
                    for m in sim_results
                    if m.history.get('total_decayed_entries')
                ]) if sim_results else 0.0
            else:
                avg_ddem_occupancy = 0.0
                avg_refused = 0.0
                avg_decayed = 0.0
            
            results.append({
                'initial_trust_mean': initial_trust_mean,
                'mobility': mobility,
                'ddem_size': ddem_size,
                'decay_rounds': decay_rounds,
                'share_trusting': share_trusting,
                'avg_final_trust': avg_final_trust,
                'std_final_trust': std_final_trust,
                'avg_ddem_occupancy': avg_ddem_occupancy,
                'avg_refused_interactions': avg_refused,
                'avg_decayed_entries': avg_decayed,
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


def run_ddem_decay_experiment_multi_trust(n_runs=N_RUNS, seed=SEED, 
                                          decay_rounds=DECAY_ROUNDS):
    """
    Run DDEM Decay experiment across MULTIPLE initial trust levels.
    """
    all_results = {}
    
    for init_trust in INITIAL_TRUST_LEVELS:
        print("\n" + "=" * 70)
        print(f"STARTING DDEM DECAY EXPERIMENTS FOR INITIAL_TRUST_MEAN = {init_trust}")
        print("=" * 70)
        
        df = run_ddem_decay_experiment_single_trust(
            init_trust, 
            n_runs=n_runs, 
            seed=seed,
            decay_rounds=decay_rounds
        )
        
        all_results[init_trust] = df
        
        filename = f'ddem_decay_results_trust{init_trust}.csv'
        df.to_csv(filename, index=False)
        print(f"\nSaved results for init_trust={init_trust} to {filename}")
    
    return all_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_ddem_decay_results_multi_trust(all_results, save=True):
    """
    Create comparison plots across different initial trust levels.
    """
    fig, axes = plt.subplots(len(INITIAL_TRUST_LEVELS), 3, 
                             figsize=(18, 6*len(INITIAL_TRUST_LEVELS)))
    
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
        ax.set_title(f'Initial Trust = {init_trust}: Trust Emergence (DDEM w/ Decay)', 
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # MIDDLE: Benefit over baseline
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
        
        # RIGHT: Decay statistics
        ax = axes[row_idx, 2]
        ddem_data = df[df['ddem_size'] > 0]
        for mob in MOBILITIES:
            data = ddem_data[ddem_data['mobility'] == mob].sort_values('ddem_size')
            if len(data) > 0:
                ax.plot(data['ddem_size'], data['avg_decayed_entries'],
                        color=colors[mob], marker=markers[mob], 
                        linewidth=2, markersize=8,
                        label=f'Mobility={mob}')
        ax.set_xlabel('DDEM Capacity', fontsize=11)
        ax.set_ylabel('Avg Decayed Entries', fontsize=11)
        ax.set_title(f'Initial Trust = {init_trust}: Memory Decay', 
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        fig.savefig('ddem_decay_results_multi_trust.png', dpi=300, bbox_inches='tight')
        print("Saved: ddem_decay_results_multi_trust.png")
    
    return fig


def print_summary_multi_trust(all_results):
    """Print formatted summary for all trust levels."""
    for init_trust in INITIAL_TRUST_LEVELS:
        df = all_results[init_trust]
        
        print("\n" + "=" * 70)
        print(f"SUMMARY: Initial Trust = {init_trust} (DDEM w/ Decay)")
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
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        print("*** QUICK MODE: 10 runs ***")
        all_results = run_ddem_decay_experiment_multi_trust(
            n_runs=10, seed=SEED, decay_rounds=DECAY_ROUNDS
        )
    else:
        all_results = run_ddem_decay_experiment_multi_trust(
            n_runs=N_RUNS, seed=SEED, decay_rounds=DECAY_ROUNDS
        )
    
    # Save combined results
    combined_df = pd.concat([df.assign(initial_trust=trust) 
                             for trust, df in all_results.items()], 
                            ignore_index=True)
    combined_df.to_csv('ddem_decay_results_all_trust_levels.csv', index=False)
    print("\nSaved: ddem_decay_results_all_trust_levels.csv")
    
    # Print summaries
    print_summary_multi_trust(all_results)
    
    # Plot
    fig = plot_ddem_decay_results_multi_trust(all_results)
    plt.show()