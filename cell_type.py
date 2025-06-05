from enum import Enum

class CellType(Enum):
    EMPTY = 0
    RESOURCE = 1
    AGENT = 2
    HAZARD = 3