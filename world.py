import numpy as np
import matplotlib.pyplot as plt
import random
from cell_type import CellType  
from agent import Agent  
from resources import Resource
from hazard import Hazard 
#from hazard import Hazard
class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.full((height, width), CellType.EMPTY, dtype=object)
        self.agents = []
        self.resources = []
        self.hazards = []
    

    def add_agent(self, num_agents):
        for _ in range(num_agents):
            x,y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            while self.grid[y, x] != CellType.EMPTY:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            agent = Agent(x, y)
            self.agents.append(agent)
            self.grid[y, x] = CellType.AGENT


    def populate_resources(self, num_resources):
        for _ in range(num_resources):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            while self.grid[y, x] != CellType.EMPTY:
                x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            resource = Resource(x, y)
            self.resources.append(resource)
            self.grid[y, x] = CellType.RESOURCE

    def populate_hazards(self, num_hazards):
        for _ in range(num_hazards):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            while self.grid[y, x] != CellType.EMPTY:
                x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            hazard = Hazard(x, y)
            self.hazards.append(hazard)
            self.grid[y, x] = CellType.HAZARD


    def step(self):
        for agent in self.agents:
            if agent.alive:
                agent.move(self)
                agent.try_reproduce(self)
                for hazard in self.hazards:
                    if agent.x == hazard.x and agent.y == hazard.y:
                        hazard.affect_agent(agent)
        
        # Let each resource decide if it wants to reproduce
        for resource in list(self.resources):  # Use a copy in case of appends
            resource.try_reproduce(self)

        for agent in self.agents:
            if agent.alive:
                self.grid[agent.y, agent.x] = CellType.AGENT

        
        alive_resources = []
        for res in self.resources:
            if res.energy > 0:
                alive_resources.append(res)
            else:
                self.grid[res.y, res.x] = CellType.EMPTY
        self.resources = alive_resources
        self.agents = [agent for agent in self.agents if agent.alive]
        self.resources = [r for r in self.resources if r.energy > 0]


    def get_neighbors(self, x, y):
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append(self.grid[ny, nx])
        return neighbors

    def render(self):
        color_map = {
            CellType.EMPTY: 0,
            CellType.RESOURCE: 1,
            CellType.HAZARD: 2,
            CellType.AGENT: 3
        }
        int_grid = np.vectorize(lambda cell: color_map[cell])(self.grid)
        cm = plt.cm.get_cmap('viridis', 4)  # Use a colormap with 4 distinct colors
        plt.imshow(int_grid, cmap=cm)

        for agent in self.agents:
            if agent.alive:
                color = agent.dna.get("color", [1, 0, 0])  # fallback to red if missing
                plt.plot(agent.x, agent.y, 'o', color=color, markersize=4)
        for hazard in self.hazards:
            plt.plot(hazard.x, hazard.y, 'X', color='black', markersize=6)  # Big black X for death

        plt.title(f"🌱 Digital Eden — Agents: {len(self.agents)}")
        plt.axis('off')
        plt.pause(0.1)
        
        # Only clear the figure if in interactive mode
        if plt.isinteractive():
            plt.clf()
    def log_stats(world, history):
        if not world.agents:
            return
        traits = {
            "move_bias": [],
            "greed": [],
            "view_range": [],
            "mutation_rate": [],
            "reproduce_threshold": []
        }
        for agent in world.agents:
            for key in traits:
                traits[key].append(agent.dna[key])

        for key in traits:
            history[key].append(np.mean(traits[key]))



if __name__ == "__main__":
    world = World(width=200, height=200)
    world.populate_resources(300)
    world.populate_hazards(50)
    #world.add_agent(20)
    plt.figure(figsize=(200, 200))
    plt.ion()
    for i in range(10000):  # 100 ticks
        if(i%1000==0):
            world.render()
            world.step()
            print(f"Tick {i+1}: {len(world.agents)} agents alive.")  # Optional debug

    plt.ioff()
    world.render()  # Final render, don't clear this one
    plt.show()
