"""
Test if shocks are actually working
"""

from core_engine_extended import ExtendedTrustSimulation
from agent import reset_agent_counter

# Test WITHOUT shock
print("="*60)
print("TEST 1: NO SHOCK")
print("="*60)

reset_agent_counter()

params = {
    'grid_size': 51,
    'n_agents': 1500,
    'share_trustworthy': 0.6,  # 40% untrustworthy
    'initial_trust_mean': 0.6,
    'initial_trust_std': 0.2,
    'sensitivity': 0.05,
    'mobility': 5,
    'trust_threshold': 0.5
}

sim_no_shock = ExtendedTrustSimulation(
    params,
    seed=42,
    verbose=True,
    shock_round=None  # NO SHOCK
)

metrics_no_shock = sim_no_shock.run(1000)
print(f"Final trust (no shock): {metrics_no_shock.get_final_level_of_trust():.3f}")
print(f"Converged to trust: {metrics_no_shock.converged_to_trust()}")

# Test WITH shock
print("\n" + "="*60)
print("TEST 2: WITH SHOCK AT ROUND 200")
print("="*60)

reset_agent_counter()

sim_with_shock = ExtendedTrustSimulation(
    params,
    seed=42,
    verbose=True,  # Show shock message
    shock_round=200  # SHOCK AT 200
)

metrics_with_shock = sim_with_shock.run(1000)
print(f"\nFinal trust (with shock): {metrics_with_shock.get_final_level_of_trust():.3f}")
print(f"Converged to trust: {metrics_with_shock.converged_to_trust()}")
print(f"Shock was applied: {sim_with_shock.shock_applied}")

# Compare
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"No shock:   {metrics_no_shock.get_final_level_of_trust():.3f}")
print(f"With shock: {metrics_with_shock.get_final_level_of_trust():.3f}")
print(f"Difference: {metrics_no_shock.get_final_level_of_trust() - metrics_with_shock.get_final_level_of_trust():.3f}")

if metrics_no_shock.get_final_level_of_trust() == metrics_with_shock.get_final_level_of_trust():
    print("\n⚠️  WARNING: Results are IDENTICAL - shock had no effect")
else:
    print("\n✅ Results are DIFFERENT - shock is working")