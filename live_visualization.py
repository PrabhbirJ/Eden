# live_viz_interactive.py
"""
Enhanced live visualization with CLEAR interaction visualization.

Shows:
- Dots = agents (color by trust)
- Lines connecting agents = interactions
  - GREEN line = cooperation
  - RED line = betrayal
  - Thickness shows strength
- Lines fade out over time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

from core_engine import TrustSimulation
from agent import reset_agent_counter


class InteractiveTrustViz:
    """
    Enhanced visualization showing WHO interacts with WHOM.
    """
    
    def __init__(self, params, seed=None):
        self.params = params
        
        # Initialize simulation
        reset_agent_counter()
        self.sim = TrustSimulation(params, seed=seed, verbose=False)
        
        # Setup figure - 2x2 grid
        self.fig = plt.figure(figsize=(16, 12))
        
        # Agent grid (left, large)
        self.ax_grid = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=2)
        
        # Trust histogram (top right)
        self.ax_hist = plt.subplot2grid((2, 2), (0, 1))
        
        # Time series (bottom right)
        self.ax_series = plt.subplot2grid((2, 2), (1, 1))
        
        self._setup_plots()
        
        # State
        self.round = 0
        self.trust_history = []
        self.coop_history = []
        self.scatter = None
        
        # Interaction tracking - store who interacted with whom
        self.recent_interactions = []  # List of {trustor, trustee, cooperated, round}
        self.interaction_lines = []  # Line objects to remove
        
    def _setup_plots(self):
        """Setup all plots."""
        gs = self.params['grid_size']
        
        # Grid
        self.ax_grid.set_xlim(-1, gs)
        self.ax_grid.set_ylim(-1, gs)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.set_title('Trust Dynamics (Lines = Interactions)', 
                               fontsize=14, fontweight='bold')
        self.ax_grid.grid(True, alpha=0.2)
        
        # Add legend for interactions
        legend_elements = [
            Line2D([0], [0], color='green', linewidth=3, label='Cooperation'),
            Line2D([0], [0], color='red', linewidth=3, label='Betrayal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='High Trust', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='Low Trust', markeredgecolor='black'),
        ]
        self.ax_grid.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Histogram
        self.ax_hist.set_title('Trust Distribution', fontsize=12)
        self.ax_hist.set_xlabel('Trust Level')
        self.ax_hist.set_ylabel('Count')
        self.ax_hist.set_xlim(0, 1)
        
        # Time series
        self.ax_series.set_title('Metrics Over Time', fontsize=12)
        self.ax_series.set_xlabel('Round')
        self.ax_series.set_ylabel('Level')
        self.ax_series.set_ylim(0, 1)
        self.ax_series.grid(True, alpha=0.3)
    
    def _run_interaction_phase_tracked(self):
        """
        Run interaction phase while tracking WHO interacts with WHOM.
        
        This is a copy of TrustSimulation._interaction_phase but with tracking.
        """
        # Shuffle for random order
        agents_shuffled = self.sim.agents.copy()
        self.sim.rng.shuffle(agents_shuffled)
        
        round_interactions = []
        
        for trustor in agents_shuffled:
            # Get neighbors
            neighbors = self.sim.grid.get_neighbors(trustor)
            
            if not neighbors:
                continue
            
            # Pick random neighbor as trustee
            trustee = self.sim.rng.choice(neighbors)
            
            # Play trust game
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
                
                # TRACK THIS INTERACTION
                if trustor.position is not None and trustee.position is not None:
                    round_interactions.append({
                        'trustor': trustor,
                        'trustee': trustee,
                        'trustor_pos': trustor.position,
                        'trustee_pos': trustee.position,
                        'cooperated': cooperated,
                        'round': self.round
                    })
                
                # Update beliefs
                trustor.update_belief(direct_info)
                trustee.update_belief(1.0)  # Social info
            else:
                # No trust
                trustee.update_belief(0.0)
        
        return round_interactions
    
    def update(self, frame):
        """Run one round and update all plots."""
        # === RUN SIMULATION ===
        
        # Run interaction phase WITH tracking
        new_interactions = self._run_interaction_phase_tracked()
        self.recent_interactions.extend(new_interactions)
        
        # Remove old interactions (keep last 3 rounds)
        self.recent_interactions = [
            i for i in self.recent_interactions 
            if self.round - i['round'] < 3
        ]
        
        # Run movement phase
        self.sim._movement_phase()
        self.round += 1
        
        # === UPDATE VISUALIZATIONS ===
        
        self._update_grid()
        self._update_histogram()
        self._update_series()
        
        return []
    
    def _update_grid(self):
        """Update agent grid with current positions and trust."""
        # Clear old interaction lines
        for line in self.interaction_lines:
            line.remove()
        self.interaction_lines = []
        
        # Get positions and trust levels
        positions = []
        trust_levels = []
        sizes = []
        
        for agent in self.sim.agents:
            if agent.position is not None:
                positions.append(agent.position)
                trust_levels.append(agent.trust_expectation)
                # Larger for trustworthy
                sizes.append(80 if agent.is_trustworthy else 50)
        
        if len(positions) == 0:
            return
        
        positions = np.array(positions)
        trust_levels = np.array(trust_levels)
        sizes = np.array(sizes)
        
        # Remove old scatter
        if self.scatter is not None:
            self.scatter.remove()
        
        # Plot agents
        colors = plt.cm.RdYlGn(trust_levels)
        self.scatter = self.ax_grid.scatter(
            positions[:, 0], positions[:, 1],
            c=colors, s=sizes,
            alpha=0.8, edgecolors='black', linewidth=1
        )
        
        # Draw interaction lines BETWEEN agents
        for interaction in self.recent_interactions:
            age = self.round - interaction['round']
            alpha = 1.0 - (age / 3.0)
            
            # Get CURRENT positions (agents may have moved)
            trustor = interaction['trustor']
            trustee = interaction['trustee']
            
            if trustor.position is None or trustee.position is None:
                continue
            
            trustor_pos = trustor.position
            trustee_pos = trustee.position
            
            # Color and style based on outcome
            if interaction['cooperated']:
                color = 'green'
                linewidth = 3
                linestyle = '-'
            else:
                color = 'red'
                linewidth = 2
                linestyle = '--'
            
            # Draw line from trustor to trustee
            line = self.ax_grid.plot(
                [trustor_pos[0], trustee_pos[0]],
                [trustor_pos[1], trustee_pos[1]],
                color=color, alpha=alpha, linewidth=linewidth, 
                linestyle=linestyle, zorder=1
            )[0]
            
            self.interaction_lines.append(line)
        
        # Update title
        avg_trust = np.mean(trust_levels)
        n_trusting = sum(1 for t in trust_levels if t >= self.params['trust_threshold'])
        pct = 100 * n_trusting / len(trust_levels)
        n_interactions = len([i for i in self.recent_interactions if self.round - i['round'] == 0])
        
        self.ax_grid.set_title(
            f'Round {self.round} | Avg Trust: {avg_trust:.3f} | Trusting: {pct:.1f}% | '
            f'Interactions this round: {n_interactions}',
            fontsize=13, fontweight='bold'
        )
    
    def _update_histogram(self):
        """Update trust distribution histogram."""
        self.ax_hist.clear()
        
        trust_levels = [a.trust_expectation for a in self.sim.agents]
        
        self.ax_hist.hist(trust_levels, bins=20, range=(0, 1),
                         color='steelblue', alpha=0.7, edgecolor='black')
        self.ax_hist.axvline(self.params['trust_threshold'], 
                            color='red', linestyle='--', linewidth=2,
                            label=f"Threshold")
        
        self.ax_hist.set_title('Trust Distribution', fontsize=12)
        self.ax_hist.set_xlabel('Trust Level')
        self.ax_hist.set_ylabel('Count')
        self.ax_hist.set_xlim(0, 1)
        self.ax_hist.legend(fontsize=9)
    
    def _update_series(self):
        """Update time series plot."""
        # Compute stats
        trust_levels = [a.trust_expectation for a in self.sim.agents]
        avg_trust = np.mean(trust_levels)
        
        n_cooperated = sum(a.n_times_cooperated for a in self.sim.agents)
        n_trusted = sum(a.n_times_trusted for a in self.sim.agents)
        coop_rate = n_cooperated / max(n_trusted, 1)
        
        self.trust_history.append(avg_trust)
        self.coop_history.append(coop_rate)
        
        # Plot
        self.ax_series.clear()
        
        rounds = list(range(len(self.trust_history)))
        self.ax_series.plot(rounds, self.trust_history, 'b-', 
                           linewidth=2, label='Avg Trust')
        self.ax_series.plot(rounds, self.coop_history, 'g-', 
                           linewidth=2, label='Coop Rate')
        
        self.ax_series.set_title('Metrics Over Time', fontsize=12)
        self.ax_series.set_xlabel('Round')
        self.ax_series.set_ylabel('Level')
        self.ax_series.set_ylim(0, 1)
        self.ax_series.grid(True, alpha=0.3)
        self.ax_series.legend(fontsize=10)
        
        # Stats box
        stats = f"""Round: {self.round}
Mobility: {self.params['mobility']}
Init Trust: {self.params['initial_trust_mean']}
Trustworthy %: {self.params['share_trustworthy']*100:.0f}%

Total Interactions: {n_trusted}
Total Cooperations: {n_cooperated}
Cooperation Rate: {coop_rate:.3f}
        """.strip()
        
        self.ax_series.text(0.98, 0.02, stats,
                           transform=self.ax_series.transAxes,
                           fontsize=9, verticalalignment='bottom',
                           horizontalalignment='right', family='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def run(self, n_rounds=500, interval=100):
        """
        Run visualization.
        
        Parameters
        ----------
        n_rounds : int
            Number of rounds
        interval : int
            Milliseconds between frames (100 = slower, clearer)
        """
        print(f"\n{'='*70}")
        print(f"INTERACTIVE TRUST DYNAMICS VISUALIZATION")
        print(f"{'='*70}")
        print(f"\nParameters:")
        print(f"  Mobility: {self.params['mobility']}")
        print(f"  Initial Trust: {self.params['initial_trust_mean']}")
        print(f"  Agents: {self.params['n_agents']}")
        print(f"  Share Trustworthy: {self.params['share_trustworthy']}")
        print(f"\nWhat you'll see:")
        print(f"  • Colored dots = agents (red=distrust, green=trust)")
        print(f"  • GREEN lines = cooperation happened")
        print(f"  • RED dashed lines = betrayal happened")
        print(f"  • Lines connect trustor → trustee")
        print(f"  • Lines fade after 3 rounds")
        print(f"\nRunning {n_rounds} rounds...")
        print(f"Close window to stop.\n")
        
        anim = animation.FuncAnimation(
            self.fig, self.update,
            frames=n_rounds,
            interval=interval,
            blit=False,
            repeat=False
        )
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS:")
        print(f"{'='*70}")
        print(f"Final average trust: {self.trust_history[-1]:.3f}")
        print(f"Final cooperation rate: {self.coop_history[-1]:.3f}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def demo(mobility=2, initial_trust=0.5, n_rounds=500):
    """Run demo visualization."""
    
    params = {
        'grid_size': 51,
        'n_agents': 100,
        'share_trustworthy': 0.6,
        'initial_trust_mean': initial_trust,
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,
        'trust_threshold': 0.5,
        'mobility': mobility,
    }
    
    viz = InteractiveTrustViz(params, seed=42)
    viz.run(n_rounds=n_rounds, interval=100)


def compare_scenarios():
    """Compare different scenarios."""
    
    scenarios = [
        {'name': 'Low Mobility (Trust Often Collapses)', 
         'mobility': 1, 'trust': 0.5},
        
        {'name': 'Medium Mobility (Mixed Outcomes)', 
         'mobility': 5, 'trust': 0.5},
        
        {'name': 'High Mobility (More Trust Rescue)', 
         'mobility': 10, 'trust': 0.5},
    ]
    
    for scenario in scenarios:
        print(f"\n\n{'#'*70}")
        print(f"# {scenario['name']}")
        print(f"{'#'*70}\n")
        
        demo(mobility=scenario['mobility'], 
             initial_trust=scenario['trust'],
             n_rounds=300)
        
        input("\n>>> Press Enter for next scenario...")


def quick_test():
    """Quick test with fewer agents."""
    params = {
        'grid_size': 51,
        'n_agents': 500,  # Fewer for speed
        'share_trustworthy': 0.6,
        'initial_trust_mean': 0.5,
        'initial_trust_std': 0.2,
        'sensitivity': 0.05,
        'trust_threshold': 0.5,
        'mobility': 5,
    }
    
    viz = InteractiveTrustViz(params, seed=42)
    viz.run(n_rounds=200, interval=50)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == 'compare':
            compare_scenarios()
        elif cmd == 'test':
            quick_test()
        else:
            mobility = int(sys.argv[1])
            initial_trust = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
            demo(mobility=mobility, initial_trust=initial_trust)
    else:
        # Default
        demo()