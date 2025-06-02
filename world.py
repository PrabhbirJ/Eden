from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import random
class CellType(Enum):
    EMPTY = 0
    RESOURCE = 1
    AGENT = 2
    HAZARD = 3
    
class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.full((height, width), CellType.EMPTY, dtype=object)

    def populate_resources(self,num_resources):
        for _ in range(num_resources):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            self.grid[y, x] = CellType.RESOURCE
    def step(self):
        new_grid = self.grid.copy()

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y, x]

                if cell == CellType.EMPTY:
                    # If surrounded by 4+ resources, grow a new one (10% chance)
                    neighbors = self.get_neighbors(x, y)
                    resource_count = sum(1 for n in neighbors if n == CellType.RESOURCE)
                    if resource_count >= 4 and random.random() < 0.1:
                        new_grid[y, x] = CellType.RESOURCE

        self.grid = new_grid

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
        }
        int_grid = np.vectorize(lambda cell: color_map[cell])(self.grid)
        plt.imshow(int_grid, cmap='viridis')
        plt.title("🌱 Digital Eden")
        plt.axis('off')
        plt.pause(0.1)
        plt.clf()



if __name__ == "__main__":
    world = World(width=50, height=50)
    world.populate_resources(300)

    plt.ion()
    for _ in range(100):  # Run 100 ticks
        world.render()
        world.step()
    plt.ioff()
    plt.show()
