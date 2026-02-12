"""
Metrics tracking and computation for Trust ABM
Implements the output measures from Klein & Marx (2018)
"""

import numpy as np


class MetricsTracker:
    """
    Track simulation metrics over time.
    
    Computes and stores:
    - Level of trust (fraction trusting at each timestep)
    - Share trustworthy (fraction of trustworthy agents)
    - Average trust expectation
    - Clustering metrics
    
    Attributes
    ----------
    history : dict
        Time series of all metrics
    """
    
    def __init__(self):
        """Initialize empty tracker."""
        self.history = {
            'round': [],
            'level_of_trust': [],
            'avg_trust_expectation': [],
            'share_trustworthy': [],
            'n_trusting': [],
            'n_agents': [],
            'avg_ddem_size': [],           # Average DDEM occupancy
            'total_refused_interactions': [], # Total refused interactions
            'ddem_hit_rate': [],           # % betrayals prevented by DDEM
        }
    
    def record(self, agents, round_num):
        """
        Record metrics for current round.
        
        Parameters
        ----------
        agents : list of Agent
            All agents in simulation
        round_num : int
            Current round number
        """
        n_agents = len(agents)
        
        # Count trusting agents (trust_expectation >= 0.5)
        n_trusting = sum(1 for agent in agents if agent.decide_to_trust())
        
        # Level of trust (main metric from paper, p. 243)
        level_of_trust = n_trusting / n_agents if n_agents > 0 else 0
        
        # Average trust expectation
        avg_trust = np.mean([agent.trust_expectation for agent in agents])
        
        # Share trustworthy (should be constant)
        share_trustworthy = np.mean([agent.is_trustworthy for agent in agents])
        
        # Store
        self.history['round'].append(round_num)
        self.history['level_of_trust'].append(level_of_trust)
        self.history['avg_trust_expectation'].append(avg_trust)
        self.history['share_trustworthy'].append(share_trustworthy)
        self.history['n_trusting'].append(n_trusting)
        self.history['n_agents'].append(n_agents)
        if hasattr(agents[0], 'get_ddem_usage'):
            ddem_sizes = [len(a.ddem) for a in agents]
            refused = sum(a.n_refused_interactions for a in agents)
            hits = sum(a.n_ddem_hits for a in agents)
            
            self.history['avg_ddem_size'].append(np.mean(ddem_sizes))
            self.history['total_refused_interactions'].append(refused)
            
            # Hit rate: prevented betrayals / total betrayals
            total_abused = sum(a.n_times_abused for a in agents)
            hit_rate = hits / max(total_abused + hits, 1)
            self.history['ddem_hit_rate'].append(hit_rate)
    
    def get_final_level_of_trust(self):
        """
        Get final level of trust (main outcome measure).
        
        Returns
        -------
        float
            Level of trust at last recorded round
        """
        if len(self.history['level_of_trust']) == 0:
            return 0.0
        return self.history['level_of_trust'][-1]
    
    def converged_to_trust(self, threshold=0.75):
        """
        Check if simulation converged to universal trust.
        
        Paper finds simulations converge to extremes (0 or 1).
        We call it "trust" if final level > threshold.
        
        Parameters
        ----------
        threshold : float, optional
            Threshold for "converged to trust" (default 0.75)
        
        Returns
        -------
        bool
            True if converged to trust, False otherwise
        """
        return self.get_final_level_of_trust() > threshold
    
    def converged_to_distrust(self, threshold=0.25):
        """
        Check if simulation converged to universal distrust.
        
        Parameters
        ----------
        threshold : float, optional
            Threshold for "converged to distrust" (default 0.25)
        
        Returns
        -------
        bool
            True if converged to distrust, False otherwise
        """
        return self.get_final_level_of_trust() < threshold
    
    def get_summary(self):
        """
        Get summary statistics.
        
        Returns
        -------
        dict
            Summary statistics
        """
        return {
            'final_level_of_trust': self.get_final_level_of_trust(),
            'converged_to_trust': self.converged_to_trust(),
            'converged_to_distrust': self.converged_to_distrust(),
            'final_avg_trust_expectation': self.history['avg_trust_expectation'][-1] if self.history['avg_trust_expectation'] else 0,
            'share_trustworthy': self.history['share_trustworthy'][-1] if self.history['share_trustworthy'] else 0,
        }


def compute_share_trusting(results):
    """
    Compute "Share trusting" metric from paper (p. 243).
    
    This is the fraction of simulation runs that converged to trust.
    
    Parameters
    ----------
    results : list of MetricsTracker
        Results from multiple simulation runs
    
    Returns
    -------
    float
        Share trusting (0 to 1)
    """
    if len(results) == 0:
        return 0.0
    
    n_trusting = sum(1 for tracker in results if tracker.converged_to_trust())
    return n_trusting / len(results)