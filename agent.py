import random
import numpy as np
from cell_type import CellType  # Assuming CellType is defined in cell_type.py

class Agent:
    def __init__(self, x, y,max_energy=100,energy=75):
        self.energy = energy
        self.x = x
        self.y = y
        self.max_energy = max_energy
        self.alive = True
        self.dna = {
            "move_bias": random.uniform(0.3, 0.9),  # Probability of moving in a tick
            "reproduce_threshold": random.randint(40, 100), #Energy required to reproduce
            "mutation_rate": random.uniform(0.01, 0.2),  # Chance of mutation on reproduction
            "view_range": random.randint(1, 3),  # How far the agent can "see" resources
            "greed": random.uniform(0.1, 0.9),  # How much the agent prioritizes resources over random movement
            "color": [random.random(), random.random(), random.random()],
            "aggression": random.uniform(0.0, 1.0),  # How aggressive the agent is towards other agents
            "cooperation": random.uniform(0.0, 1.0),  # How cooperative the agent is with others
            "memory": set()
        }

    def move(self, world):
        if not self.alive:
            return
        if( random.random() > self.dna["move_bias"]):
            return
        visible_resources = [
        res for res in world.resources
        if abs(res.x - self.x) <= self.dna["view_range"]
        and abs(res.y - self.y) <= self.dna["view_range"]
        ]
        
        target = None
        if visible_resources and random.random() < self.dna["greed"]:
            # Go toward the closest visible resource
            target = min(visible_resources, key=lambda res: abs(res.x - self.x) + abs(res.y - self.y))
            dx = np.sign(target.x - self.x)
            dy = np.sign(target.y - self.y)
            move_options = [(dx, 0), (0, dy), (dx, dy)]
        else:
            # Random wandering
            move_options = [(0,1), (1,0), (0,-1), (-1,0)]
            random.shuffle(move_options)

        # Try each move until successful
        for dx, dy in move_options:
            nx,ny = self.x + dx, self.y + dy
            if 0 <= nx < world.width and 0 <= ny < world.height:
                target_cell = world.grid[ny, nx]
                if target_cell == CellType.EMPTY:
                    world.grid[self.y, self.x] = CellType.EMPTY
                    self.x, self.y = nx, ny
                    world.grid[ny, nx] = CellType.AGENT
                elif target_cell == CellType.RESOURCE:
                    for res in world.resources:
                        if res.x == nx and res.y == ny:
                            self.increase_energy(res.energy) # Increase energy for collecting resources
                            world.resources.remove(res)
                    world.grid[self.y, self.x] = CellType.EMPTY
                    self.x, self.y = nx, ny
                    world.grid[ny, nx] = CellType.AGENT
                self.decrease_energy(1)  # Decrease energy for moving
                if self.energy <= 0:
                    self.alive = False
                    world.grid[self.y, self.x] = CellType.EMPTY
                else:
                    world.grid[self.y, self.x] = CellType.AGENT

    def increase_energy(self, amount):
        self.energy = min(self.max_energy, self.energy + amount)

    def decrease_energy(self, amount):
        self.energy = max(0, self.energy - amount)
        if self.energy <= 0:
            self.alive = False

    def attack(self,other_agent):
        if not self.alive or not other_agent.alive:
            return
        damage = random.randint(5, 15)
        other_agent.decrease_energy(damage)
        
     
              

    def try_reproduce(self, world):
        if self.energy < self.dna['reproduce_threshold'] or random.random() > 0.5:
            return
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < world.width and 0 <= ny < world.height and world.grid[ny, nx] == CellType.EMPTY:
                baby = Agent(nx, ny, max_energy=self.max_energy, energy=self.energy // 2)
                baby.dna = self.mutate_dna()
                self.energy //= 2
                world.agents.append(baby)
                world.grid[ny, nx] = CellType.AGENT
                break

    def mutate_dna(self):
        new_dna = self.dna.copy()
        
        for key in new_dna:
            chance = random.random()
            if key == "color":
                if  chance< self.dna["mutation_rate"]:
                    new_dna["color"] = [
                        min(1.0, max(0.0, c + random.uniform(-0.05, 0.05)))
                        for c in new_dna["color"]
                    ]
            elif chance < self.dna["mutation_rate"]:
                val = new_dna[key]
                if isinstance(val, float):
                    new_dna[key] = max(0.0, min(1.0, val + random.uniform(-0.1, 0.1)))
                elif isinstance(val, int):
                    new_dna[key] = max(1, val + random.randint(-5, 5))
        return new_dna
    
    def __repr__(self):
        return f"Agent(x={self.x}, y={self.y}, energy={self.energy})"

            
        