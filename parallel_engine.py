# parallel_engine.py
"""
Multi-threaded Trust Dynamics Engine

Improvements:
1. Background thread for simulation (doesn't block UI)
2. Batch processing (run multiple steps at once)
3. Configurable update frequency
4. Better performance for large populations
"""

import threading
import queue
import time
import numpy as np
from collections import deque

from core_engine import TrustSimulation
from agent import reset_agent_counter


class ParallelTrustEngine:
    """
    Trust simulation engine that runs in background thread.
    
    Main thread handles UI/API requests.
    Worker thread runs simulation continuously.
    Communication via thread-safe queues.
    """
    
    def __init__(self, params, seed=None):
        self.params = params
        self.seed = seed
        
        # Simulation state
        self.sim = None
        self.round = 0
        self.running = False
        self.paused = False
        
        # Thread-safe data structures
        self.state_lock = threading.Lock()
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue(maxsize=1)  # Only keep latest
        
        # Cached state for fast API responses
        self.cached_state = {
            'agents': [],
            'interactions': [],
            'trust_history': deque(maxlen=200),
            'coop_history': deque(maxlen=200),
            'stats': {},
        }
        
        # Worker thread
        self.worker_thread = None
        self.should_stop = threading.Event()
        
        # Performance settings
        self.steps_per_update = 1  # How many steps before updating cache
        self.max_interactions_tracked = 30
        
        # Initialize
        self._initialize_sim()
        
    def _initialize_sim(self):
        """Initialize simulation (thread-safe)."""
        with self.state_lock:
            reset_agent_counter()
            self.sim = TrustSimulation(self.params, seed=self.seed, verbose=False)
            self.round = 0
            self.cached_state['trust_history'].clear()
            self.cached_state['coop_history'].clear()
    
    def start(self):
        """Start the background simulation thread."""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.should_stop.clear()
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("🚀 Parallel engine started")
    
    def stop(self):
        """Stop the background thread."""
        self.should_stop.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        print("⏹ Parallel engine stopped")
    
    def pause(self):
        """Pause simulation."""
        self.paused = True
    
    def resume(self):
        """Resume simulation."""
        self.paused = False
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        return not self.paused  # Return running state
    
    def reset(self):
        """Reset simulation."""
        self._initialize_sim()
        self.paused = False
    
    def update_params(self, new_params):
        """Update parameters (some apply immediately, some on reset)."""
        with self.state_lock:
            for key, value in new_params.items():
                if key in self.params:
                    self.params[key] = value
            
            # Apply immediate changes
            if self.sim:
                if 'mobility' in new_params:
                    self.params['mobility'] = int(new_params['mobility'])
                
                if 'trust_threshold' in new_params:
                    for agent in self.sim.agents:
                        agent.trust_threshold = new_params['trust_threshold']
                
                if 'sensitivity' in new_params:
                    for agent in self.sim.agents:
                        agent.sensitivity = new_params['sensitivity']
    
    def set_speed(self, steps_per_update):
        """Set how many simulation steps to run per update."""
        self.steps_per_update = max(1, int(steps_per_update))
    
    def _worker_loop(self):
        """Main worker thread loop - runs simulation continuously."""
        
        while not self.should_stop.is_set():
            # Check if paused
            if self.paused:
                time.sleep(0.05)
                continue
            
            # Run batch of simulation steps
            for _ in range(self.steps_per_update):
                if self.should_stop.is_set() or self.paused:
                    break
                self._run_single_step()
            
            # Update cached state for API
            self._update_cache()
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.001)
    
    def _run_single_step(self):
        """Run one simulation step (called from worker thread)."""
        with self.state_lock:
            if not self.sim:
                return
            
            # Interaction phase with tracking
            agents = self.sim.agents.copy()
            self.sim.rng.shuffle(agents)
            
            interactions = []
            interaction_count = 0
            
            for trustor in agents:
                neighbors = self.sim.grid.get_neighbors(trustor)
                if not neighbors:
                    continue
                
                trustee = self.sim.rng.choice(neighbors)
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
                    
                    # Track limited interactions
                    if interaction_count < self.max_interactions_tracked:
                        if trustor.position and trustee.position:
                            interactions.append({
                                'trustor_pos': list(trustor.position),
                                'trustee_pos': list(trustee.position),
                                'cooperated': bool(cooperated)
                            })
                            interaction_count += 1
                    
                    trustor.update_belief(direct_info)
                    trustee.update_belief(1.0)
                else:
                    trustee.update_belief(0.0)
            
            # Movement
            self.sim._movement_phase()
            self.round += 1
            
            # Store interactions for this round
            self.cached_state['interactions'] = interactions
    
    def _update_cache(self):
        """Update cached state (called from worker thread)."""
        with self.state_lock:
            if not self.sim:
                return
            
            # Sample agents for rendering
            agents = []
            agent_sample_rate = 1 if len(self.sim.agents) < 1000 else 2
            
            for i, a in enumerate(self.sim.agents):
                if i % agent_sample_rate == 0 and a.position:
                    agents.append({
                        'x': int(a.position[0]),
                        'y': int(a.position[1]),
                        'trust': float(a.trust_expectation),
                        'trustworthy': bool(a.is_trustworthy)
                    })
            
            self.cached_state['agents'] = agents
            
            # Update metrics every 5 rounds
            if self.round % 5 == 0:
                trusts = [a.trust_expectation for a in self.sim.agents]
                avg_trust = float(np.mean(trusts))
                self.cached_state['trust_history'].append(avg_trust)
                
                n_coop = sum(a.n_times_cooperated for a in self.sim.agents)
                n_trust = sum(a.n_times_trusted for a in self.sim.agents)
                coop_rate = float(n_coop / max(n_trust, 1))
                self.cached_state['coop_history'].append(coop_rate)
            
            # Update stats
            trusts = [a.trust_expectation for a in self.sim.agents]
            threshold = self.params['trust_threshold']
            n_trusting = sum(1 for t in trusts if t >= threshold)
            
            n_coop = sum(a.n_times_cooperated for a in self.sim.agents)
            n_trust = sum(a.n_times_trusted for a in self.sim.agents)
            
            self.cached_state['stats'] = {
                'round': self.round,
                'avg_trust': float(np.mean(trusts)),
                'pct_trusting': float(100 * n_trusting / len(self.sim.agents)),
                'coop_rate': float(n_coop / max(n_trust, 1)),
                'n_agents': len(self.sim.agents),
            }
    
    def get_state(self):
        """Get current cached state (thread-safe, fast)."""
        with self.state_lock:
            return {
                'agents': self.cached_state['agents'].copy(),
                'interactions': self.cached_state['interactions'].copy(),
                'stats': self.cached_state['stats'].copy(),
                'trust_history': list(self.cached_state['trust_history']),
                'coop_history': list(self.cached_state['coop_history']),
                'running': not self.paused,
            }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Parallel Engine...")
    
    params = {
        'grid_size': 51,
        'n_agents': 1000,
        'share_trustworthy': 0.6,
        'initial_trust_mean': 0.5,
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,
        'trust_threshold': 0.5,
        'mobility': 5,
    }
    
    engine = ParallelTrustEngine(params, seed=42)
    engine.start()
    
    try:
        # Run for 10 seconds
        for i in range(20):
            time.sleep(0.5)
            state = engine.get_state()
            print(f"Round {state['stats']['round']}: "
                  f"Trust={state['stats']['avg_trust']:.3f}, "
                  f"Agents={len(state['agents'])}")
        
    finally:
        engine.stop()
    
    print("\n✅ Test complete!")