import random
from resource_type import ResourceType
from cell_type import CellType  
class Resource:
    def __init__(self, x, y, resource_type=ResourceType.FOOD,max_life=10):
        self.x = x
        self.y = y
        self.type = resource_type
        self.energy = resource_type.energy
        self.life = max_life

    def decrease_life(self):
        self.life -= 1
        if self.life <= 0:
            self.energy = 0
    
    def try_reproduce(self, world):
        if random.random() > 0.1:
            return

        # Spread to a neighboring EMPTY cell
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < world.width and 0 <= ny < world.height and world.grid[ny, nx] == CellType.EMPTY:
                new_resource = Resource(nx, ny, self.type)
                world.resources.append(new_resource)
                world.grid[ny, nx] = CellType.RESOURCE
                break


    def __repr__(self):
        return f"Resource(x={self.x}, y={self.y}, type={self.type.name}, amount={self.amount})"
