"""
Core simulation engine for Trust ABM
Klein & Marx (2018) implementation

This is the heart of the simulation - coordinates agent interactions,
movement, and learning according to the paper's specification.
"""

import numpy as np
from agent import Agent, create_agent_population, reset_agent_counter
from grid import TrustGrid
from metrics import MetricsTracker


class TrustSimulation:
    """
    Main simulation engine for trust dynamics.
    
    Implements the full simulation loop from Klein & Marx (2018):
    1. Agents interact in trust games (Section 3.4, p. 242)
    2. Agents update beliefs based on outcomes (Equation I, p. 241)
    3. Agents move around the grid (Section 3.4, p. 242)
    4. Repeat for N rounds
    
    Attributes
    ----------
    params : dict
        Simulation parameters
    grid : TrustGrid
        Spatial world
    agents : list of Agent
        All agents
    metrics : MetricsTracker
        Tracks outcomes over time
    rng : np.random.Generator
        Random number generator (for reproducibility)
    """
    
    def __init__(self, params, seed=None, verbose=True):
        """
        Initialize simulation.
        
        Parameters
        ----------
        params : dict
            Must contain:
            - 'grid_size': int
            - 'n_agents': int
            - 'share_trustworthy': float
            - 'initial_trust_mean': float
            - 'initial_trust_std': float
            - 'sensitivity': float
            - 'mobility': int
            - 'trust_threshold': float
        seed : int, optional
            Random seed for reproducibility
        verbose : bool, optional
            Print progress (default True)
        """
        self.params = params
        self.verbose = verbose
        
        # Random number generator
        self.rng = np.random.default_rng(seed)
        
        # Create world
        self.grid = TrustGrid(size=params['grid_size'])
        
        # Create agents
        self.agents = create_agent_population(
            n_agents=params['n_agents'],
            share_trustworthy=params['share_trustworthy'],
            initial_trust_mean=params['initial_trust_mean'],
            initial_trust_std=params['initial_trust_std'],
            sensitivity=params['sensitivity'],
            trust_threshold=params['trust_threshold'],
            rng=self.rng
        )
        
        # Place agents on grid
        self.grid.place_agents(self.agents, rng=self.rng)
        
        # Metrics tracker
        self.metrics = MetricsTracker()
        
        # Current round
        self.current_round = 0
        
        if self.verbose:
            print(f"Initialized simulation: {len(self.agents)} agents on {self.grid.size}x{self.grid.size} grid")
            print(f"  Share trustworthy: {params['share_trustworthy']:.2f}")
            print(f"  Initial trust (mean): {params['initial_trust_mean']:.2f}")
            print(f"  Sensitivity: {params['sensitivity']:.3f}")
            print(f"  Mobility: {params['mobility']}")
    
    def run(self, n_rounds, record_interval=10):
        """
        Run simulation for N rounds.
        
        Each round consists of:
        1. Interaction phase: agents pair up and play trust games
        2. Movement phase: agents move on the grid
        
        Parameters
        ----------
        n_rounds : int
            Number of rounds to simulate
        record_interval : int, optional
            Record metrics every N rounds (default 10)
        
        Returns
        -------
        MetricsTracker
            Metrics from the run
        """
        # Record initial state
        self.metrics.record(self.agents, round_num=0)
        
        for round_num in range(1, n_rounds + 1):
            self.current_round = round_num
            
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
            print(f"  Converged to: {'TRUST' if summary['converged_to_trust'] else 'DISTRUST' if summary['converged_to_distrust'] else 'MIXED'}")
        
        return self.metrics
    
    def _interaction_phase(self):
        """
        Interaction phase: agents pair up and play trust games.
        
        Following paper (Section 3.4, p. 242):
        1. Randomly shuffle agents
        2. Pair agents with available neighbors
        3. Randomly assign roles (trustor/trustee)
        4. Play trust game
        5. Both agents learn from outcome
        """
        # Shuffle agents to randomize pairing order
        unpaired = self.agents.copy()
        self.rng.shuffle(unpaired)
        
        # Track who's already paired this round
        paired_this_round = set()
        
        while len(unpaired) > 0:
            # Pick an agent
            agent1 = unpaired.pop()
            
            if agent1.agent_id in paired_this_round:
                continue
            
            # Get neighbors who aren't paired yet
            neighbors = self.grid.get_neighbors(agent1)
            available_neighbors = [
                n for n in neighbors 
                if n.agent_id not in paired_this_round and n != agent1
            ]
            
            if len(available_neighbors) == 0:
                # No available neighbors, agent stays unpaired this round
                continue
            
            # Pick random neighbor
            agent2 = self.rng.choice(available_neighbors)
            
            # Mark both as paired
            paired_this_round.add(agent1.agent_id)
            paired_this_round.add(agent2.agent_id)
            
            # Remove agent2 from unpaired if still there
            if agent2 in unpaired:
                unpaired.remove(agent2)
            
            # Randomly assign roles
            if self.rng.random() < 0.5:
                trustor, trustee = agent1, agent2
            else:
                trustor, trustee = agent2, agent1
            
            # Play trust game
            self._play_trust_game(trustor, trustee)
    
    def _play_trust_game(self, trustor, trustee):
        """
        Play a single trust game between two agents.
        
        Implements the trust game from Figure 1 (p. 238):
        1. Trustor decides whether to trust
        2. If trust placed:
           - Trustee responds (cooperate or abuse)
           - Trustor gets direct information (I = 1 or 0)
           - Trustee gets social information (I = 1)
        3. If no trust:
           - No interaction
           - Trustee gets social information (I = 0)
        
        Parameters
        ----------
        trustor : Agent
            Agent in trustor role
        trustee : Agent
            Agent in trustee role
        """
        # Trustor decides
        if trustor.decide_to_trust():
            # Trust is placed
            trustor.n_times_trusted += 1
            
            # Trustee responds
            cooperated = trustee.as_trustee_respond()
            
            if cooperated:
                trustor.n_times_cooperated += 1
            else:
                trustor.n_times_abused += 1
            
            # DIRECT INFORMATION for trustor (Channel 1, p. 241)
            # I = 1 if cooperated, I = 0 if abused
            direct_info = 1.0 if cooperated else 0.0
            trustor.update_belief(direct_info)
            
            # SOCIAL INFORMATION for trustee (Channel 2, p. 241)
            # Trustor placed trust → trustee observes trustor is willing to trust
            # This implies trustor thinks others are trustworthy
            # I = 1 (positive signal)
            social_info = 1.0
            trustee.update_belief(social_info)
            
        else:
            # No trust placed
            # SOCIAL INFORMATION for trustee (Channel 2, p. 241)
            # Trustor refused to trust → trustee observes trustor is unwilling
            # This implies trustor thinks others are untrustworthy
            # I = 0 (negative signal)
            social_info = 0.0
            trustee.update_belief(social_info)
            
            # Trustor learns nothing (didn't interact)
    
    def _movement_phase(self):
        """
        Movement phase: agents move on the grid.
        
        Following paper (Section 3.4, p. 242):
        Each agent moves 'mobility' steps in a random direction.
        """
        mobility = self.params['mobility']
        
        for agent in self.agents:
            self.grid.move_agent(agent, distance=mobility, rng=self.rng)


# =============================================================================
# BATCH SIMULATION RUNNER
# =============================================================================

def run_batch_simulations(params, n_runs=100, n_rounds=1000, 
                          record_interval=10, seed=None, verbose=1):
    """
    Run multiple simulations with same parameters.
    
    This is needed to compute "Share trusting" metric (p. 243).
    
    Parameters
    ----------
    params : dict
        Simulation parameters
    n_runs : int, optional
        Number of independent runs (default 100)
    n_rounds : int, optional
        Rounds per simulation (default 1000)
    record_interval : int, optional
        Record metrics every N rounds (default 10)
    seed : int, optional
        Base random seed (each run gets seed + run_num)
    verbose : int, optional
        Verbosity level:
        0 = silent
        1 = progress summary
        2 = detailed per-run output
    
    Returns
    -------
    list of MetricsTracker
        Results from all runs
    """
    results = []
    
    if verbose >= 1:
        print(f"\nRunning {n_runs} simulations with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print()
    
    for run_num in range(n_runs):
        # Reset agent ID counter (so IDs start at 0 each run)
        reset_agent_counter()
        
        # Set seed for this run
        run_seed = None if seed is None else seed + run_num
        
        # Create and run simulation
        show_verbose = (verbose >= 2)
        
        if verbose >= 1 and run_num % 10 == 0:
            print(f"Run {run_num + 1}/{n_runs}...", end='\r')
        
        sim = TrustSimulation(params, seed=run_seed, verbose=show_verbose)
        metrics = sim.run(n_rounds, record_interval)
        
        results.append(metrics)
    
    if verbose >= 1:
        print(f"\nCompleted {n_runs} runs.")
        
        # Compute aggregate statistics
        from metrics import compute_share_trusting
        share_trusting = compute_share_trusting(results)
        
        avg_final_trust = np.mean([m.get_final_level_of_trust() for m in results])
        std_final_trust = np.std([m.get_final_level_of_trust() for m in results])
        
        print(f"\nAggregate Results:")
        print(f"  Share trusting: {share_trusting:.3f}")
        print(f"  Avg final level of trust: {avg_final_trust:.3f} ± {std_final_trust:.3f}")
        from export import ResultsExporter
        exporter = ResultsExporter()
        exporter.save_batch_runs(results, params, experiment_name='batch_auto')
    
    return results


# =============================================================================
# PARAMETER SWEEP RUNNER (for verification experiments)
# =============================================================================

def run_parameter_sweep(base_params, param_name, param_values, 
                       n_runs=100, n_rounds=1000, seed=None, verbose=1):
    """
    Sweep one parameter while holding others constant.
    
    Used for reproducing figures from the paper.
    
    Parameters
    ----------
    base_params : dict
        Base parameters (will be modified for each sweep value)
    param_name : str
        Name of parameter to vary
    param_values : list
        Values to test
    n_runs : int, optional
        Runs per parameter value (default 100)
    n_rounds : int, optional
        Rounds per simulation (default 1000)
    seed : int, optional
        Base random seed
    verbose : int, optional
        Verbosity level
    
    Returns
    -------
    dict
        Results dictionary with keys:
        - 'param_name': str
        - 'param_values': list
        - 'share_trusting': list (one per param value)
        - 'all_results': list of list (all MetricsTrackers)
    """
    from metrics import compute_share_trusting
    
    sweep_results = {
        'param_name': param_name,
        'param_values': param_values,
        'share_trusting': [],
        'avg_final_trust': [],
        'all_results': []
    }
    
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"PARAMETER SWEEP: {param_name}")
        print(f"{'='*60}\n")
    
    for i, value in enumerate(param_values):
        if verbose >= 1:
            print(f"\n[{i+1}/{len(param_values)}] Testing {param_name} = {value}")
        
        # Create params for this sweep value
        params = base_params.copy()
        params[param_name] = value
        
        # Run batch
        results = run_batch_simulations(
            params, 
            n_runs=n_runs, 
            n_rounds=n_rounds,
            seed=seed,
            verbose=max(0, verbose)  # Less verbose for individual runs
        )
        
        # Compute metrics
        share_trusting = compute_share_trusting(results)
        avg_final_trust = np.mean([m.get_final_level_of_trust() for m in results])
        
        # Store
        sweep_results['share_trusting'].append(share_trusting)
        sweep_results['avg_final_trust'].append(avg_final_trust)
        sweep_results['all_results'].append(results)
        
        if verbose >= 1:
            print(f"  → Share trusting: {share_trusting:.3f}")
    
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"SWEEP COMPLETE")
        print(f"{'='*60}\n")
        from export import ResultsExporter
        exporter = ResultsExporter()
        exporter.save_parameter_sweep(sweep_results,base_params)
    
    return sweep_results