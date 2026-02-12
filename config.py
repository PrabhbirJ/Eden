'''
Config File using given values from the paper and other values assumed
'''
# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
GRID_SIZE = 51
N_AGENTS = 1500
N_ROUNDS = 1000 #paper uses 1000 rounds which is roughly 83 years if each round is a month
RECORD_INTERVAL = 10 #how often to record metrics(ever N rounds)

# =============================================================================
# AGENT PARAMETERS (these will be varied in experiments(assumed within ranges))
# =============================================================================

SHARE_TRUSTWORTHY = 0.5 #how many trustworthy agents(range: 0.4 to 0.7)
SENSITIVITY = 0.05#Sensitivity to new information (range: 0.03 to 0.1)
MOBILITY = 5 # Mobility: number of steps agents move each round (range: 1 to 20)

INITIAL_TRUST_MEAN = 0.6 # Initial trust endowment (mean, range: 0.4 to 0.8 in paper)
INITIAL_TRUST_STD = 0.2  # Paper uses σ = 0.2


TRUST_THRESHOLD = 0.5 # Decision threshold for trusting (paper uses 0.5)

# =============================================================================
# EXPERIMENTAL PARAMETERS
# =============================================================================


N_RUNS = 100 # Number of simulation runs per parameter combination
RANDOM_SEED = 1310

SAVE_RESULTS = True
OUTPUT_DIR = 'results/'

VERBOSE = 1