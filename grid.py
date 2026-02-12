"""
Grid/World class for Trust ABM
Handles spatial structure and agent movement
"""

import numpy as np


class TrustGrid:
    """
    2D toroidal grid for trust game simulation.
    
    Agents live on a 2D grid with periodic boundary conditions (torus).
    Each cell can contain at most one agent.
    Agents interact with von Neumann neighbors (N, S, E, W).
    
    Attributes
    ----------
    size : int
        Grid dimension (size x size)
    grid : np.ndarray
        2D array storing agent references
    agents : list
        All agents in the simulation
    
    Methods
    -------
    place_agents(agents)
        Randomly place agents on empty cells
    get_neighbors(agent)
        Get von Neumann neighbors of an agent
    move_agent(agent, distance)
        Move agent random direction for distance steps
    """
    
    def __init__(self, size):
        """
        Initialize empty grid.
        
        Parameters
        ----------
        size : int
            Grid dimension (creates size x size grid)
        """
        self.size = size
        self.grid = np.full((size, size), None)  # None = empty cell
        self.agents = []
        
        # Track agent positions for fast lookup
        self.agent_positions = {}  # {agent_id: (x, y)}
    
    def place_agents(self, agents, rng=None):
        """
        Place agents randomly on grid.
        
        Each agent gets a unique position.
        
        Parameters
        ----------
        agents : list of Agent
            Agents to place
        rng : np.random.Generator, optional
            Random number generator
        """
        if rng is None:
            rng = np.random.default_rng()
        
        self.agents = agents
        
        # Get all possible positions
        all_positions = [(i, j) for i in range(self.size) 
                                 for j in range(self.size)]
        
        # Sample without replacement
        if len(agents) > len(all_positions):
            raise ValueError(f"Too many agents ({len(agents)}) for grid ({self.size}x{self.size})")
        
        positions = rng.choice(len(all_positions), size=len(agents), replace=False)
        
        for agent, pos_idx in zip(agents, positions):
            pos = all_positions[pos_idx]
            self._place_agent_at(agent, pos)
    
    def _place_agent_at(self, agent, position):
        """
        Place agent at specific position.
        
        Parameters
        ----------
        agent : Agent
            Agent to place
        position : tuple
            (x, y) coordinates
        """
        x, y = position
        
        # Remove from old position if exists
        if agent.position is not None:
            old_x, old_y = agent.position
            self.grid[old_x, old_y] = None
        
        # Place at new position
        self.grid[x, y] = agent
        agent.position = position
        self.agent_positions[agent.agent_id] = position
    
    def get_neighbors(self, agent):
        """
        Get von Neumann neighbors (N, S, E, W) of agent.
        
        Uses periodic boundary conditions (torus topology).
        
        Parameters
        ----------
        agent : Agent
            Agent whose neighbors to find
        
        Returns
        -------
        list of Agent
            Neighboring agents (length 0-4)
        """
        if agent.position is None:
            return []
        
        x, y = agent.position
        
        # Von Neumann neighborhood (4 cardinal directions)
        neighbor_positions = [
            ((x - 1) % self.size, y),      # North
            ((x + 1) % self.size, y),      # South
            (x, (y - 1) % self.size),      # West
            (x, (y + 1) % self.size)       # East
        ]
        
        # Get agents at those positions (filter out None/empty cells)
        neighbors = []
        for pos in neighbor_positions:
            neighbor = self.grid[pos]
            if neighbor is not None:
                neighbors.append(neighbor)
        
        return neighbors
    
    def move_agent(self, agent, distance, rng=None):
        """
        Move agent a certain distance in a random direction.
        
        Movement is random walk:
        - Pick random direction (N, S, E, W)
        - Move one step
        - Repeat for 'distance' steps
        
        If target cell is occupied, keep trying until find empty cell.
        
        Parameters
        ----------
        agent : Agent
            Agent to move
        distance : int
            Number of steps to move
        rng : np.random.Generator, optional
            Random number generator
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if agent.position is None:
            return
        
        x, y = agent.position
        
        # Random walk
        for step in range(distance):
            # Pick random direction
            direction = rng.choice(['N', 'S', 'E', 'W'])
            
            # Calculate target position
            if direction == 'N':
                target_x, target_y = (x - 1) % self.size, y
            elif direction == 'S':
                target_x, target_y = (x + 1) % self.size, y
            elif direction == 'W':
                target_x, target_y = x, (y - 1) % self.size
            else:  # E
                target_x, target_y = x, (y + 1) % self.size
            
            # If target is empty, move there
            if self.grid[target_x, target_y] is None:
                self._place_agent_at(agent, (target_x, target_y))
                x, y = target_x, target_y
            # Otherwise, try again from current position
            # (this prevents agents from getting stuck)
    
    def get_empty_cells(self):
        """
        Get list of empty cell coordinates.
        
        Returns
        -------
        list of tuple
            List of (x, y) coordinates where grid is None
        """
        empty = []
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] is None:
                    empty.append((i, j))
        return empty
    
    def get_density(self):
        """
        Calculate grid density (fraction of cells occupied).
        
        Returns
        -------
        float
            Density from 0 (empty) to 1 (full)
        """
        occupied = sum(1 for i in range(self.size) 
                         for j in range(self.size) 
                         if self.grid[i, j] is not None)
        return occupied / (self.size * self.size)
    
    def __repr__(self):
        """String representation."""
        return f"TrustGrid({self.size}x{self.size}, {len(self.agents)} agents, density={self.get_density():.2f})"