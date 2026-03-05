# multi_sim_engine.py
"""
Multi-Simulation Parallel Engine

Uses Python multiprocessing to run multiple simulations simultaneously.
Perfect for:
- Parameter sweeps
- A/B comparisons
- Scenario testing
- Ensemble runs

Each simulation runs on a separate CPU core.
"""

import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
import numpy as np
import time
from collections import deque

from core_engine import TrustSimulation
from agent import reset_agent_counter


def run_simulation_worker(sim_id, params, result_queue, control_dict, seed=None):
    """
    Worker function that runs a single simulation.
    
    Runs in separate process, sends updates back via queue.
    """
    # Initialize simulation
    reset_agent_counter()
    sim = TrustSimulation(params, seed=seed, verbose=False)
    
    round_num = 0
    
    while control_dict['running']:
        # Check if paused
        if control_dict.get(f'paused_{sim_id}', False):
            time.sleep(0.05)
            continue
        
        # Run one step WITH INTERACTION TRACKING
        agents = sim.agents.copy()
        sim.rng.shuffle(agents)
        
        interactions = []  # FRESH list each round - no accumulation!
        interaction_count = 0
        max_interactions_to_track = 15  # Reduced from 20
        
        for trustor in agents:
            neighbors = sim.grid.get_neighbors(trustor)
            if not neighbors:
                continue
            
            trustee = sim.rng.choice(neighbors)
            will_trust = trustor.decide_to_trust()
            
            if will_trust:
                trustor.n_times_trusted += 1
                cooperated = trustee.as_trustee_respond()
                
                if cooperated:
                    trustor.n_times_cooperated += 1
                    direct_info = 1.0
                else:
                    trustor.n_times_abused += 1
                    direct_info = 0.0
                
                # Track ONLY recent interactions - don't accumulate!
                if interaction_count < max_interactions_to_track:
                    if trustor.position and trustee.position:
                        interactions.append({
                            'trustor_pos': list(trustor.position),
                            'trustee_pos': list(trustee.position),
                            'cooperated': bool(cooperated),
                        })
                        interaction_count += 1
                
                trustor.update_belief(direct_info)
                trustee.update_belief(1.0)
            else:
                trustee.update_belief(0.0)
        
        # Movement
        sim._movement_phase()
        round_num += 1
        
        # Send update every 3 rounds (was 5) for more frequent updates
        if round_num % 3 == 0:
            # Collect data - with sampling for performance
            agents_data = []
            
            # Adaptive sampling based on agent count
            if len(sim.agents) > 1000:
                sample_rate = 4
            elif len(sim.agents) > 600:
                sample_rate = 3
            else:
                sample_rate = 2
            
            for i, a in enumerate(sim.agents):
                if i % sample_rate == 0 and a.position:
                    agents_data.append({
                        'x': int(a.position[0]),
                        'y': int(a.position[1]),
                        't': float(a.trust_expectation),
                    })
            
            trusts = [a.trust_expectation for a in sim.agents]
            avg_trust = float(np.mean(trusts))
            
            threshold = params['trust_threshold']
            n_trusting = sum(1 for t in trusts if t >= threshold)
            pct_trusting = 100.0 * n_trusting / len(trusts)
            
            n_coop = sum(a.n_times_cooperated for a in sim.agents)
            n_trust_total = sum(a.n_times_trusted for a in sim.agents)
            coop_rate = n_coop / max(n_trust_total, 1)
            
            # Send update with FRESH interactions only
            update = {
                'sim_id': sim_id,
                'round': round_num,
                'agents': agents_data,
                'interactions': interactions,  # Fresh list, not accumulated
                'stats': {
                    'avg_trust': avg_trust,
                    'pct_trusting': pct_trusting,
                    'coop_rate': coop_rate,
                }
            }
            
            try:
                result_queue.put_nowait(update)
            except:
                pass  # Queue full, skip this update
        
        # Small sleep to prevent CPU saturation
        time.sleep(0.001)


class MultiSimEngine:
    """
    Runs multiple simulations in parallel.
    
    Each simulation gets its own CPU core via multiprocessing.
    Useful for parameter comparison, ensemble runs, etc.
    """
    
    def __init__(self, scenarios, max_workers=4):
        """
        Initialize multi-sim engine.
        
        Parameters
        ----------
        scenarios : list of dict
            Each dict has 'name', 'params', 'seed'
        max_workers : int
            Max number of parallel simulations
        """
        self.scenarios = scenarios[:max_workers]  # Limit to max_workers
        self.max_workers = max_workers
        
        # Multiprocessing setup
        self.manager = Manager()
        self.control_dict = self.manager.dict()
        self.control_dict['running'] = False
        
        self.result_queue = Queue(maxsize=100)
        self.workers = []
        
        # State for each simulation
        self.sim_states = {
            i: {
                'name': scenario['name'],
                'round': 0,
                'agents': [],
                'interactions': [],  # Add interactions
                'trust_history': deque(maxlen=100),
                'coop_history': deque(maxlen=100),
                'stats': {
                    'avg_trust': 0.0,
                    'pct_trusting': 0.0,
                    'coop_rate': 0.0,
                }
            }
            for i, scenario in enumerate(self.scenarios)
        }
    
    def start(self):
        """Start all simulations."""
        if self.control_dict['running']:
            return
        
        self.control_dict['running'] = True
        
        # Start worker processes
        for i, scenario in enumerate(self.scenarios):
            self.control_dict[f'paused_{i}'] = False
            
            p = Process(
                target=run_simulation_worker,
                args=(i, scenario['params'], self.result_queue, 
                      self.control_dict, scenario.get('seed'))
            )
            p.start()
            self.workers.append(p)
        
        print(f"🚀 Started {len(self.workers)} parallel simulations")
    
    def stop(self):
        """Stop all simulations."""
        self.control_dict['running'] = False
        
        for p in self.workers:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        
        self.workers = []
        print("⏹ Stopped all simulations")
    
    def pause(self, sim_id=None):
        """Pause simulation(s)."""
        if sim_id is None:
            # Pause all
            for i in range(len(self.scenarios)):
                self.control_dict[f'paused_{i}'] = True
        else:
            self.control_dict[f'paused_{sim_id}'] = True
    
    def resume(self, sim_id=None):
        """Resume simulation(s)."""
        if sim_id is None:
            # Resume all
            for i in range(len(self.scenarios)):
                self.control_dict[f'paused_{i}'] = False
        else:
            self.control_dict[f'paused_{sim_id}'] = False
    
    def update_states(self):
        """Process updates from worker processes."""
        updates_processed = 0
        
        while not self.result_queue.empty() and updates_processed < 20:
            try:
                update = self.result_queue.get_nowait()
                sim_id = update['sim_id']
                
                if sim_id in self.sim_states:
                    state = self.sim_states[sim_id]
                    state['round'] = update['round']
                    state['agents'] = update['agents']
                    state['interactions'] = update.get('interactions', [])
                    state['stats'] = update['stats']
                    
                    # Update history
                    state['trust_history'].append(update['stats']['avg_trust'])
                    state['coop_history'].append(update['stats']['coop_rate'])
                
                updates_processed += 1
            except:
                break
        
        return updates_processed
    
    def get_all_states(self):
        """Get current state of all simulations."""
        self.update_states()
        
        return {
            sim_id: {
                'name': state['name'],
                'round': state['round'],
                'agents': state['agents'],
                'interactions': state['interactions'],
                'stats': state['stats'],
                'trust_history': list(state['trust_history']),
                'coop_history': list(state['coop_history']),
            }
            for sim_id, state in self.sim_states.items()
        }


# =============================================================================
# PRESET SCENARIOS
# =============================================================================

def get_mobility_comparison():
    """Compare different mobility levels."""
    base_params = {
        'grid_size': 51,
        'n_agents': 600,  # Smaller for performance
        'share_trustworthy': 0.6,
        'initial_trust_mean': 0.5,
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,
        'trust_threshold': 0.5,
    }
    
    return [
        {
            'name': 'Low Mobility (1)',
            'params': {**base_params, 'mobility': 1},
            'seed': 42
        },
        {
            'name': 'Medium Mobility (5)',
            'params': {**base_params, 'mobility': 5},
            'seed': 42
        },
        {
            'name': 'High Mobility (10)',
            'params': {**base_params, 'mobility': 10},
            'seed': 42
        },
        {
            'name': 'Very High Mobility (20)',
            'params': {**base_params, 'mobility': 20},
            'seed': 42
        },
    ]


def get_trust_threshold_comparison():
    """Compare different trust thresholds."""
    base_params = {
        'grid_size': 51,
        'n_agents': 600,
        'share_trustworthy': 0.6,
        'initial_trust_mean': 0.5,
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,
        'mobility': 5,
    }
    
    return [
        {
            'name': 'Threshold 0.3 (Trusting)',
            'params': {**base_params, 'trust_threshold': 0.3},
            'seed': 42
        },
        {
            'name': 'Threshold 0.5 (Moderate)',
            'params': {**base_params, 'trust_threshold': 0.5},
            'seed': 42
        },
        {
            'name': 'Threshold 0.7 (Cautious)',
            'params': {**base_params, 'trust_threshold': 0.7},
            'seed': 42
        },
        {
            'name': 'Threshold 0.9 (Very Cautious)',
            'params': {**base_params, 'trust_threshold': 0.9},
            'seed': 42
        },
    ]


def get_initial_trust_comparison():
    """Compare different initial trust levels."""
    base_params = {
        'grid_size': 51,
        'n_agents': 600,
        'share_trustworthy': 0.6,
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,
        'trust_threshold': 0.5,
        'mobility': 5,
    }
    
    return [
        {
            'name': 'Init Trust 0.3 (Pessimistic)',
            'params': {**base_params, 'initial_trust_mean': 0.3},
            'seed': 42
        },
        {
            'name': 'Init Trust 0.5 (Neutral)',
            'params': {**base_params, 'initial_trust_mean': 0.5},
            'seed': 42
        },
        {
            'name': 'Init Trust 0.7 (Optimistic)',
            'params': {**base_params, 'initial_trust_mean': 0.7},
            'seed': 42
        },
        {
            'name': 'Init Trust 0.9 (Very Optimistic)',
            'params': {**base_params, 'initial_trust_mean': 0.9},
            'seed': 42
        },
    ]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Multi-Sim Engine...")
    print(f"CPU cores available: {mp.cpu_count()}")
    
    # Test with mobility comparison
    scenarios = get_mobility_comparison()
    engine = MultiSimEngine(scenarios, max_workers=4)
    
    print(f"\nRunning {len(scenarios)} scenarios in parallel:")
    for i, s in enumerate(scenarios):
        print(f"  {i+1}. {s['name']}")
    
    engine.start()
    
    try:
        # Run for 30 seconds
        for i in range(30):
            time.sleep(1)
            states = engine.get_all_states()
            
            print(f"\n--- Second {i+1} ---")
            for sim_id, state in states.items():
                print(f"  {state['name']:30s} | "
                      f"Round {state['round']:4d} | "
                      f"Trust {state['stats']['avg_trust']:.3f}")
    
    finally:
        engine.stop()
    
    print("\n✅ Test complete!")