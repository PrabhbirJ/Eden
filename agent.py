"""
Agent class for Trust ABM
Implements trustor and trustee behavior with learning
"""

import numpy as np


class Agent:
    """
    An agent in the trust game.
    
    Each agent has:
    - A fixed trustee type (trustworthy or not)
    - A learned trust expectation (belief about others' trustworthiness)
    - A position on the grid
    
    Attributes
    ----------
    agent_id : int
        Unique identifier
    trust_expectation : float
        Belief that others are trustworthy (0 to 1)
    is_trustworthy : bool
        Fixed type: will this agent cooperate when trusted?
    position : tuple
        (x, y) coordinates on grid
    sensitivity : float
        Learning rate (weight on new information)
    
    Methods
    -------
    decide_to_trust()
        Decision rule: trust if trust_expectation >= threshold
    update_belief(observation)
        1(weighted average)
    """
    
    # Class variable to track total agents created (for unique IDs)
    id_counter = 0
    
    def __init__(self, 
                 is_trustworthy,
                 initial_trust,
                 sensitivity,
                 trust_threshold=0.5):
        """
        Initialize an agent.
        
        Parameters
        ----------
        is_trustworthy : bool
            Agent's fixed type as trustee
        initial_trust : float
            Initial trust expectation (0 to 1)
        sensitivity : float
            Learning rate (0 to 1)
        trust_threshold : float, optional
            Threshold for deciding to trust (default 0.5)
        """
        self.agent_id = Agent.id_counter
        Agent.id_counter += 1
        
        # Fixed traits
        self.is_trustworthy = is_trustworthy
        self.sensitivity = sensitivity
        self.trust_threshold = trust_threshold
        
        # Learned belief (this changes over time)
        self.trust_expectation = np.clip(initial_trust, 0.0, 1.0)
        
        # Position (set by grid when agent is placed)
        self.position = None
        
        # Tracking (for analysis)
        self.n_times_trusted = 0
        self.n_times_cooperated = 0
        self.n_times_abused = 0
        self.history_trust_expectation = [self.trust_expectation]
    
    def decide_to_trust(self):
        """
        Trust if trust_expectation >= threshold.
        Returns
        -------
        bool
            True if agent decides to place trust, False otherwise
        """
        return self.trust_expectation >= self.trust_threshold
    
    def update_belief(self, observation):
        """
        Update trust expectation using weighted average:
        trust_new = (1 - s) * trust_old + s * I
        
        where:
        - s = sensitivity (learning rate)
        - I = observation (0 or 1)
        
        Parameters
        ----------
        observation : float
            New information (0 = negative, 1 = positive)
        """
        # Equation (I) from paper
        self.trust_expectation = (
            (1 - self.sensitivity) * self.trust_expectation + 
            self.sensitivity * observation
        )
        
        # Clip to valid range [0, 1]
        self.trust_expectation = np.clip(self.trust_expectation, 0.0, 1.0)
        
        # Track history
        self.history_trust_expectation.append(self.trust_expectation)
    
    def as_trustee_respond(self):
        """
        Behavior as trustee in trust game.
        
        Fixed by agent type (doesn't change over simulation).
        
        Returns
        -------
        bool
            True if cooperate, False if abuse trust
        """
        return self.is_trustworthy
    
    def __repr__(self):
        """String representation for debugging."""
        trustee_type = "Trustworthy" if self.is_trustworthy else "Untrustworthy"
        willing = "Yes" if self.decide_to_trust() else "No"
        return (f"Agent({self.agent_id}: {trustee_type}, "
                f"trust={self.trust_expectation:.2f}, willing={willing})")


# =============================================================================
# UTILITY FUNCTIONS FOR AGENT CREATION
# =============================================================================

def create_agent_population(n_agents, 
                            share_trustworthy,
                            initial_trust_mean,
                            initial_trust_std,
                            sensitivity,
                            trust_threshold=0.5,
                            rng=None):
    """
    Create a population of agents with specified parameters.
    
    This is a factory function to create many agents at once.
    
    Parameters
    ----------
    n_agents : int
        Number of agents to create
    share_trustworthy : float
        Proportion of trustworthy agents (0 to 1)
    initial_trust_mean : float
        Mean of initial trust distribution
    initial_trust_std : float
        Std dev of initial trust distribution
    sensitivity : float
        Learning rate for all agents
    trust_threshold : float, optional
        Trust decision threshold (default 0.5)
    rng : np.random.Generator, optional
        Random number generator (for reproducibility)
    
    Returns
    -------
    list of Agent
        Population of initialized agents
    """
    if rng is None:
        rng = np.random.default_rng()
    
    agents = []
    
    for i in range(n_agents):
        # Determine trustee type
        is_trustworthy = rng.random() < share_trustworthy
        
        # Draw initial trust from normal distribution
        initial_trust = rng.normal(initial_trust_mean, initial_trust_std)
        
        # Create agent
        agent = Agent(
            is_trustworthy=is_trustworthy,
            initial_trust=initial_trust,
            sensitivity=sensitivity,
            trust_threshold=trust_threshold
        )
        
        agents.append(agent)
    
    return agents


# =============================================================================
# DDEM EXTENSION (Deque of Denied Engagement Memory)
# =============================================================================
from collections import deque

class AgentWithDDEM(Agent):
    """
    Agent with DDEM (Deque of Denied Engagement Memory).
    
    Remembers agents who abused trust and refuses to interact with them.
    Uses a fixed-size deque (FIFO) to store betrayer IDs.
    """
    
    def __init__(self, is_trustworthy, initial_trust, sensitivity, 
                 trust_threshold=0.5, ddem_size=10):
        """
        Parameters
        ----------
        ddem_size : int
            Maximum number of betrayers to remember (deque size)
        """
        super().__init__(is_trustworthy, initial_trust, sensitivity, trust_threshold)
        
        # DDEM: Deque of agent IDs who betrayed us
        self.ddem = deque(maxlen=ddem_size)  # Oldest auto-removed when full
        
        # Tracking metrics
        self.n_refused_interactions = 0  # How many times we refused to play
        self.n_ddem_hits = 0  # How many times DDEM prevented betrayal
        
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
        # Standard Klein & Marx decision
        return self.trust_expectation >= self.trust_threshold
    
    def record_betrayal(self, betrayer):
        """
        Add betrayer to DDEM after being abused.
        
        Parameters
        ----------
        betrayer : Agent
            The agent who betrayed our trust
        """
        if betrayer.agent_id not in self.ddem:
            self.ddem.append(betrayer.agent_id)
            self.n_ddem_hits += 1
        # If already in deque, do nothing (or could move to end)
    
    def get_ddem_usage(self):
        """Return memory usage statistics."""
        return {
            'ddem_size': len(self.ddem),
            'refused_interactions': self.n_refused_interactions,
            'ddem_hits': self.n_ddem_hits
        }
    
def reset_agent_counter():
    """Reset the global agent ID counter (call between simulation runs)."""
    Agent.id_counter = 0


def create_ddem_agent_population(n_agents, share_trustworthy, initial_trust_mean,
                                  initial_trust_std, sensitivity, trust_threshold=0.5,
                                  ddem_size=10, rng=None):
    """
    Create a population of DDEM agents.
    
    Same as create_agent_population but uses AgentWithDDEM.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    agents = []
    for i in range(n_agents):
        is_trustworthy = rng.random() < share_trustworthy
        initial_trust = rng.normal(initial_trust_mean, initial_trust_std)
        
        agent = AgentWithDDEM(
            is_trustworthy=is_trustworthy,
            initial_trust=initial_trust,
            sensitivity=sensitivity,
            trust_threshold=trust_threshold,
            ddem_size=ddem_size
        )
        agents.append(agent)
    
    return agents