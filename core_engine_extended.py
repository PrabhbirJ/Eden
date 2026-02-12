"""
Extended simulation engine with advanced features
"""

import numpy as np
from core_engine import TrustSimulation
from advanced_features import AgentWithSocialLearning, apply_negative_shock, compare_trustworthy_vs_untrustworthy


class ExtendedTrustSimulation(TrustSimulation):
    """
    Extended simulation with:
    - Relative social learning
    - External shocks
    - Enhanced tracking
    """
    
    def __init__(self, params, seed=None, verbose=True, 
                 social_learning_factor=1.0,
                 shock_round=None,
                 shock_params=None):
        """
        Initialize extended simulation.
        
        Parameters
        ----------
        params : dict
            Base parameters (same as TrustSimulation)
        seed : int, optional
            Random seed
        verbose : bool, optional
            Print progress
        social_learning_factor : float, optional
            Relative weight on social vs direct info (default 1.0)
            Set to values in [0, 0.5, 1.0, 1.5, 2.0] for Figure 4 right
        shock_round : int, optional
            Round at which to apply shock (default None = no shock)
            Paper uses 200 for Figure 5
        shock_params : dict, optional
            Shock parameters: {'min_reduction': 0.0, 'max_reduction': 0.5}
        """
        # Initialize base simulation
        super().__init__(params, seed, verbose)
        
        # Wrap agents with social learning capability
        if social_learning_factor != 1.0:
            self.agents = [AgentWithSocialLearning(agent, social_learning_factor) 
                          for agent in self.agents]
        
        self.social_learning_factor = social_learning_factor
        self.shock_round = shock_round
        self.shock_params = shock_params or {'min_reduction': 0.0, 'max_reduction': 0.5}
        self.shock_applied = False
        self.shock_stats = None
        
        # Enhanced tracking
        self.trustworthy_comparison_history = []
    
    def run(self, n_rounds, record_interval=10):
        """
        Run simulation with shocks and enhanced tracking.
        
        Parameters
        ----------
        n_rounds : int
            Number of rounds
        record_interval : int
            Record frequency
        
        Returns
        -------
        MetricsTracker
            Results
        """
        # Record initial state
        self.metrics.record(self.agents, round_num=0)
        self._record_trustworthy_comparison(0)
        
        for round_num in range(1, n_rounds + 1):
            self.current_round = round_num
            
            # Apply shock if scheduled
            if self.shock_round is not None and round_num == self.shock_round and not self.shock_applied:
                if self.verbose:
                    print(f"\n  ⚡ SHOCK at round {round_num}")
                
                self.shock_stats = apply_negative_shock(
                    self.agents,
                    min_reduction=self.shock_params['min_reduction'],
                    max_reduction=self.shock_params['max_reduction'],
                    rng=self.rng
                )
                self.shock_applied = True
                
                if self.verbose:
                    print(f"     Level of trust: {self.shock_stats['pre_shock_level_of_trust']:.3f} → {self.shock_stats['post_shock_level_of_trust']:.3f}")
            
            # Phase 1: Interaction
            self._interaction_phase()
            
            # Phase 2: Movement
            self._movement_phase()
            
            # Record metrics
            if round_num % record_interval == 0:
                self.metrics.record(self.agents, round_num)
                self._record_trustworthy_comparison(round_num)
                
                if self.verbose and round_num % 100 == 0:
                    level = self.metrics.get_final_level_of_trust()
                    print(f"  Round {round_num}/{n_rounds}: Level of trust = {level:.3f}")
        
        # Final recording
        self.metrics.record(self.agents, round_num=n_rounds)
        self._record_trustworthy_comparison(n_rounds)
        
        if self.verbose:
            summary = self.metrics.get_summary()
            print(f"\nSimulation complete:")
            print(f"  Final level of trust: {summary['final_level_of_trust']:.3f}")
            print(f"  Converged to: {'TRUST' if summary['converged_to_trust'] else 'DISTRUST' if summary['converged_to_distrust'] else 'MIXED'}")
            
            if self.shock_applied:
                print(f"\n  Shock impact:")
                print(f"    Reduction in level: {self.shock_stats['reduction_in_level']:.3f}")
        
        return self.metrics
    
    # core_engine.py (modification)

    def _play_trust_game(self, trustor, trustee):
        """
        Modified trust game with DDEM memory.
        
        Now includes pre-interaction check:
        - If trustor has betrayer in DDEM, no interaction occurs
        """
        
        # NEW: Check if trustor will interact with this trustee
        has_ddem = hasattr(trustor, 'will_interact_with')
        
        if has_ddem:
            will_interact = trustor.will_interact_with(trustee)
            if not will_interact:
                # Trustor refuses to play with this trustee
                # No payoffs, no learning, no trust game
                # This is equivalent to "no trust" outcome
                
                # Social information: trustee sees trustor didn't trust
                social_info = 0.0
                trustee.update_belief(social_info)
                return
        
        # Standard trust game proceeds
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
                
                # NEW: Record betrayal in DDEM
                if has_ddem:
                    trustor.record_betrayal(trustee)
            
            # Update trustor's belief
            trustor.update_belief(direct_info)
            
            # Social information for trustee
            social_info = 1.0
            trustee.update_belief(social_info)
        else:
            # Trustor chose not to trust
            social_info = 0.0
            trustee.update_belief(social_info)
    
    def _record_trustworthy_comparison(self, round_num):
        """Record comparison between trustworthy and untrustworthy agents."""
        comparison = compare_trustworthy_vs_untrustworthy(self.agents)
        comparison['round'] = round_num
        self.trustworthy_comparison_history.append(comparison)


# =============================================================================
# DDEM-AWARE SIMULATION
# =============================================================================

class DDEMTrustSimulation(ExtendedTrustSimulation):
    """
    Simulation that creates AgentWithDDEM instead of regular agents.
    
    Overrides agent creation to use DDEM agents while keeping
    all the trust game logic from ExtendedTrustSimulation.
    """
    
    def __init__(self, params, seed=None, verbose=True,
                 social_learning_factor=1.0,
                 shock_round=None, shock_params=None):
        """
        Initialize DDEM simulation.
        
        params must include 'ddem_size' (int).
        """
        # Don't call super().__init__ because it creates regular agents.
        # Instead, replicate initialization with DDEM agents.
        from agent import create_ddem_agent_population
        from grid import TrustGrid
        from metrics import MetricsTracker
        
        self.params = params
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        
        self.grid = TrustGrid(size=params['grid_size'])
        
        ddem_size = params.get('ddem_size', 10)
        
        self.agents = create_ddem_agent_population(
            n_agents=params['n_agents'],
            share_trustworthy=params['share_trustworthy'],
            initial_trust_mean=params['initial_trust_mean'],
            initial_trust_std=params['initial_trust_std'],
            sensitivity=params['sensitivity'],
            trust_threshold=params['trust_threshold'],
            ddem_size=ddem_size,
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
            print(f"Initialized DDEM simulation: {len(self.agents)} agents, "
                  f"DDEM size={ddem_size}")


# =============================================================================
# BATCH RUNNER FOR DDEM EXPERIMENTS
# =============================================================================

def run_batch_simulations_ddem(params, n_runs=100, n_rounds=1000,
                                record_interval=10, seed=None, verbose=1):
    """
    Run multiple DDEM simulations with same parameters.
    
    Parameters
    ----------
    params : dict
        Must include 'ddem_size'. If ddem_size=0, uses regular TrustSimulation.
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
    from agent import reset_agent_counter
    from core_engine import TrustSimulation
    
    results = []
    ddem_size = params.get('ddem_size', 0)
    mode = f"DDEM (size={ddem_size})" if ddem_size > 0 else "Baseline"
    
    if verbose >= 1:
        print(f"  [{mode}] mobility={params['mobility']}, "
              f"init_trust={params['initial_trust_mean']} | {n_runs} runs x {n_rounds} rounds")
    
    for run_num in range(n_runs):
        reset_agent_counter()
        run_seed = None if seed is None else seed + run_num
        
        if ddem_size > 0:
            sim = DDEMTrustSimulation(params, seed=run_seed, verbose=False)
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
            
            # DDEM info
            ddem_info = ""
            if ddem_size > 0 and metrics.history['avg_ddem_size']:
                ddem_occ = metrics.history['avg_ddem_size'][-1]
                refused = metrics.history['total_refused_interactions'][-1]
                ddem_info = f" | DDEM occ={ddem_occ:.1f}, refused={refused}"
            
            print(f"    Run {n_done:3d}/{n_runs}: {outcome:8s} (trust={final:.3f}) "
                  f"| running share_trusting={running_share:.2f}{ddem_info}")
            
            # Intra-sim trajectory: show trust at key rounds
            # Pull from recorded history (recorded every record_interval rounds)
            rounds = metrics.history['round']
            trusts = metrics.history['level_of_trust']
            
            # Pick ~6 checkpoints spread across the simulation
            snapshot_rounds = [50, 100, 200, 400, 600, 800]
            snapshots = []
            for target in snapshot_rounds:
                # Find closest recorded round
                best_idx = None
                best_dist = float('inf')
                for idx, r in enumerate(rounds):
                    if abs(r - target) < best_dist:
                        best_dist = abs(r - target)
                        best_idx = idx
                if best_idx is not None and best_dist <= record_interval:
                    snapshots.append(f"r{rounds[best_idx]}={trusts[best_idx]:.2f}")
            
            if snapshots:
                print(f"           trajectory: {' -> '.join(snapshots)} -> final={final:.3f}")
        
        elif verbose >= 1 and (run_num + 1) % 25 == 0:
            n_done = run_num + 1
            n_trust = sum(1 for m in results if m.get_final_level_of_trust() > 0.75)
            print(f"    {n_done}/{n_runs} done... share_trusting so far: {n_trust/n_done:.3f}")
    
    # Final summary
    from metrics import compute_share_trusting
    st = compute_share_trusting(results)
    avg = np.mean([m.get_final_level_of_trust() for m in results])
    
    if verbose >= 1:
        print(f"  => Share trusting: {st:.3f} | Avg final trust: {avg:.3f}")
    
    return results


# =============================================================================
# 2D PARAMETER SWEEP
# =============================================================================

def run_2d_parameter_sweep(base_params, 
                           param1_name, param1_values,
                           param2_name, param2_values,
                           n_runs=100, n_rounds=1000,
                           seed=None, verbose=1,
                           use_social_learning=False,
                           use_shocks=False):
    """
    Run 2D parameter sweep (for multi-line plots).
    
    This is the KEY function for reproducing paper figures.
    
    Example: Figure 2 LEFT
    - param1 = 'initial_trust_mean' (X-axis)
    - param2 = 'mobility' (creates lines)
    
    Parameters
    ----------
    base_params : dict
        Base parameters
    param1_name : str
        First parameter to vary (X-axis)
    param1_values : list
        Values for param1
    param2_name : str
        Second parameter to vary (creates multiple lines)
    param2_values : list
        Values for param2
    n_runs : int, optional
        Runs per parameter combination (default 100)
    n_rounds : int, optional
        Rounds per simulation (default 1000)
    seed : int, optional
        Base random seed
    verbose : int, optional
        Verbosity level
    use_social_learning : bool, optional
        Use social learning variant (default False)
    use_shocks : bool, optional
        Apply shocks at round 200 (default False)
    
    Returns
    -------
    dict
        Results with structure:
        {
            'param1_name': str,
            'param1_values': list,
            'param2_name': str,
            'param2_values': list,
            'share_trusting': 2D array [len(param2), len(param1)],
            'avg_final_trust': 2D array [len(param2), len(param1)],
            'all_results': 2D list of lists of MetricsTrackers
        }
    """
    from metrics import compute_share_trusting
    
    # Initialize result arrays
    n_param1 = len(param1_values)
    n_param2 = len(param2_values)
    
    share_trusting_grid = np.zeros((n_param2, n_param1))
    avg_final_trust_grid = np.zeros((n_param2, n_param1))
    all_results = [[None for _ in range(n_param1)] for _ in range(n_param2)]
    
    total_combinations = n_param1 * n_param2
    current_combination = 0
    
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"2D PARAMETER SWEEP")
        print(f"{'='*60}")
        print(f"Param 1 (X-axis): {param1_name} - {len(param1_values)} values")
        print(f"Param 2 (lines):  {param2_name} - {len(param2_values)} values")
        print(f"Total combinations: {total_combinations}")
        print(f"Runs per combination: {n_runs}")
        print(f"Total simulations: {total_combinations * n_runs:,}")
        print(f"{'='*60}\n")
    
    for j, param2_val in enumerate(param2_values):
        for i, param1_val in enumerate(param1_values):
            current_combination += 1
            
            if verbose >= 1:
                print(f"\n[{current_combination}/{total_combinations}] {param2_name}={param2_val}, {param1_name}={param1_val}")
            
            # Create params for this combination
            params = base_params.copy()
            params[param1_name] = param1_val
            params[param2_name] = param2_val
            
            # Run batch
            batch_results = []
            
            for run_num in range(n_runs):
                # Reset agent counter
                from agent import reset_agent_counter
                reset_agent_counter()
                
                run_seed = None if seed is None else seed + current_combination * 1000 + run_num
                
                # Choose simulation type
                if use_social_learning or use_shocks:
                    social_factor = params.get('social_learning_factor', 1.0)
                    shock_round = 200 if use_shocks else None
                    
                    sim = ExtendedTrustSimulation(
                        params, 
                        seed=run_seed,
                        verbose=True,
                        social_learning_factor=social_factor,
                        shock_round=shock_round
                    )
                else:
                    from core_engine import TrustSimulation
                    sim = TrustSimulation(params, seed=run_seed, verbose=True)
                
                metrics = sim.run(n_rounds, record_interval=10)
                batch_results.append(metrics)
                
                if verbose >= 1 and (run_num + 1) % 10 == 0:
                    print(f"  Run {run_num + 1}/{n_runs}...", end='\r')
            
            # Compute metrics
            share_trusting = compute_share_trusting(batch_results)
            avg_final = np.mean([m.get_final_level_of_trust() for m in batch_results])
            
            # Store
            share_trusting_grid[j, i] = share_trusting
            avg_final_trust_grid[j, i] = avg_final
            all_results[j][i] = batch_results
            
            if verbose >= 1:
                print(f"  → Share trusting: {share_trusting:.3f}, Avg final: {avg_final:.3f}")
    
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"2D SWEEP COMPLETE")
        print(f"{'='*60}\n")
    
    return {
        'param1_name': param1_name,
        'param1_values': param1_values,
        'param2_name': param2_name,
        'param2_values': param2_values,
        'share_trusting': share_trusting_grid,
        'avg_final_trust': avg_final_trust_grid,
        'all_results': all_results
    }