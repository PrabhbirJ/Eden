from enum import Enum

class ResourceType(Enum):
    FOOD = {"amount": 10}
    WATER = {"amount": 5}
    MEDICINE = {"amount": 20}

    def __init__(self, properties):
        self.energy = properties["amount"]
