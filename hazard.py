import random
from cell_type import CellType

class Hazard:
    def __init__(self, x, y, damage=10):
        self.x = x
        self.y = y
        self.damage = damage  # how much energy it removes on contact

    def affect_agent(self, agent):
        agent.decrease_energy(self.damage)
