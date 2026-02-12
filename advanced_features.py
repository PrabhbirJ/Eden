"""
Advanced features for Trust ABM
- Relative social learning
- External shocks
- Enhanced tracking
"""

import numpy as np


# =============================================================================
# RELATIVE SOCIAL LEARNING
# =============================================================================

class AgentWithSocialLearning:
    """
    Extended agent with separate weights for direct vs social information.
    
    This implements the social learning variant from Figure 4 (right).
    
    Learning equation becomes:
    trust_new = (1 - s*f) * trust_old + s*f * I
    
    where:
    - s = sensitivity (base learning rate)
    - f = relative social learning factor
    - I = information (weighted by f if social, by 1 if direct)
    """
    
    def __init__(self, agent, social_learning_factor=1.0):
        """
        Wrap an existing agent with social learning capability.
        
        Parameters
        ----------
        agent : Agent
            Base agent
        social_learning_factor : float
            Relative weight on social vs direct info
            - f=0: ignore social info
            - f=1: treat social and direct equally
            - f=2: social info weighted 2× more than direct
        """
        self.agent = agent
        self.social_learning_factor = social_learning_factor
    
    def update_belief_with_type(self, observation, is_social=False):
        """
        Update belief with different weights for social vs direct.
        
        Parameters
        ----------
        observation : float
            New information (0 or 1)
        is_social : bool
            Is this social information (vs direct)?
        """
        # Determine effective weight
        if is_social:
            effective_weight = self.agent.sensitivity * self.social_learning_factor
        else:
            # Direct information uses base sensitivity
            effective_weight = self.agent.sensitivity
        
        # Clip weight to [0, 1]
        effective_weight = np.clip(effective_weight, 0.0, 1.0)
        
        # Update (modified Equation I)
        self.agent.trust_expectation = (
            (1 - effective_weight) * self.agent.trust_expectation + 
            effective_weight * observation
        )
        
        # Clip to valid range
        self.agent.trust_expectation = np.clip(self.agent.trust_expectation, 0.0, 1.0)
        
        # Track history
        self.agent.history_trust_expectation.append(self.agent.trust_expectation)
    
    def __getattr__(self, name):
        """Delegate all other attributes to wrapped agent."""
        return getattr(self.agent, name)


# =============================================================================
# SHOCK MECHANISMS
# =============================================================================

def apply_negative_shock(agents, min_reduction=0.0, max_reduction=0.5, rng=None):
    """
    Apply negative shock to all agents' trust expectations.
    
    This simulates external events like:
    - Fraud scandal in the news
    - Political crisis
    - Economic downturn
    
    From paper (p. 252):
    "After 200 rounds of simulation, we let an external shock
    diminish each agent's trust expectation by a random amount 
    between 0 and 0.5 points."
    
    Parameters
    ----------
    agents : list of Agent
        Agents to shock
    min_reduction : float, optional
        Minimum trust reduction (default 0.0)
    max_reduction : float, optional
        Maximum trust reduction (default 0.5)
    rng : np.random.Generator, optional
        Random number generator
    
    Returns
    -------
    dict
        Statistics about shock impact
    """
    if rng is None:
        rng = np.random.default_rng()
    
    pre_shock_trust = [agent.trust_expectation for agent in agents]
    pre_shock_level = sum(1 for agent in agents if agent.trust_expectation >= 0.5) / len(agents)
    
    # Apply random reduction to each agent
    for agent in agents:
        reduction = rng.uniform(min_reduction, max_reduction)
        agent.trust_expectation = max(0.0, agent.trust_expectation - reduction)
    
    post_shock_trust = [agent.trust_expectation for agent in agents]
    post_shock_level = sum(1 for agent in agents if agent.trust_expectation >= 0.5) / len(agents)
    
    shock_stats = {
        'pre_shock_level_of_trust': pre_shock_level,
        'post_shock_level_of_trust': post_shock_level,
        'reduction_in_level': pre_shock_level - post_shock_level,
        'avg_reduction_per_agent': np.mean([pre - post for pre, post in zip(pre_shock_trust, post_shock_trust)])
    }
    
    return shock_stats


def apply_positive_shock(agents, min_increase=0.0, max_increase=0.5, rng=None):
    """
    Apply positive shock (e.g., trust-building campaign).
    
    NOT in the paper, but useful for extensions.
    
    Parameters
    ----------
    agents : list of Agent
        Agents to affect
    min_increase : float
        Minimum trust increase
    max_increase : float
        Maximum trust increase
    rng : np.random.Generator, optional
        Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()
    
    for agent in agents:
        increase = rng.uniform(min_increase, max_increase)
        agent.trust_expectation = min(1.0, agent.trust_expectation + increase)


# =============================================================================
# ENHANCED TRACKING
# =============================================================================

def compare_trustworthy_vs_untrustworthy(agents):
    """
    Compare trust levels between trustworthy and untrustworthy agents.
    
    This is for Figure 6 (p. 254).
    
    Paper finds: In 60-75% of simulations, trustworthy agents
    have HIGHER trust expectations than untrustworthy agents.
    
    Parameters
    ----------
    agents : list of Agent
        All agents
    
    Returns
    -------
    dict
        Comparison statistics
    """
    trustworthy = [a for a in agents if a.is_trustworthy]
    untrustworthy = [a for a in agents if not a.is_trustworthy]
    
    if len(trustworthy) == 0 or len(untrustworthy) == 0:
        return {
            'trustworthy_avg': 0,
            'untrustworthy_avg': 0,
            'trustworthy_higher': False,
            'difference': 0
        }
    
    trustworthy_avg = np.mean([a.trust_expectation for a in trustworthy])
    untrustworthy_avg = np.mean([a.trust_expectation for a in untrustworthy])
    
    return {
        'trustworthy_avg': trustworthy_avg,
        'untrustworthy_avg': untrustworthy_avg,
        'trustworthy_higher': trustworthy_avg > untrustworthy_avg,
        'difference': trustworthy_avg - untrustworthy_avg
    }


def track_convergence_time(metrics):
    """
    Estimate when simulation converged to final state.
    
    Useful for understanding dynamics.
    
    Parameters
    ----------
    metrics : MetricsTracker
        Metrics from a run
    
    Returns
    -------
    int or None
        Round at which convergence occurred, or None if didn't converge
    """
    levels = metrics.history['level_of_trust']
    
    if len(levels) < 10:
        return None
    
    # Check for convergence to trust (>0.9)
    for i in range(len(levels) - 10):
        if all(l > 0.9 for l in levels[i:i+10]):
            return metrics.history['round'][i]
    
    # Check for convergence to distrust (<0.1)
    for i in range(len(levels) - 10):
        if all(l < 0.1 for l in levels[i:i+10]):
            return metrics.history['round'][i]
    
    return None