import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms

# --- System Constants ---
MU_MAX = 0.5        # Maximum specific growth rate (h^-1)
K_S = 0.1           # Monod saturation constant (g/L)
Y_XS = 0.5          # Biomass yield from substrate (g/g)
M_X = 0.01          # Maintenance coefficient 
S_IN = 20.0         # Substrate concentration in feed (g/L)
I_IN = 10.0         # IPTG concentration in feed (g/L)
K_I = 1.0           # Induction Hill constant
HILL_N = 2.1        # Hill coefficient
# CRITICAL: When you finish your docking in VdockS and get your real numbers, 
# you will change this k_cat variable to match your high, mid, and low affinity antibiotics.
K_CAT_S = 380       # Catalytic speed (s^-1)
K_ACT_H = K_CAT_S * 3600  # Converted to h^-1 (CRITICAL conversion)
F_OUT = 0.0         # Assuming no outflow for standard fed-batch

def get_F_in(t, genes):
    """Return the feed rate for the discrete time intervals based on DEAP genes"""
    if t < 4:
        return genes[0]
    elif t < 8:
        return genes[1]
    elif t < 12:
        return genes[2]
    elif t < 16:
        return genes[3]
    else: # t <= 24
        return genes[4]

def bioreactor_odes(t, state, genes):
    """
    state = [XV, SV, V, EV, IV]
    XV = Biomass mass
    SV = Substrate mass
    V  = Volume
    EV = Enzyme Product mass
    IV = IPTG/Inducer mass
    """
    XV, SV, V, EV, IV = state
    
    # Prevent numerical instability for negative masses
    XV = max(0.0, XV)
    SV = max(0.0, SV)
    V = max(0.1, V)
    EV = max(0.0, EV)
    IV = max(0.0, IV)
    
    # Calculate concentrations (g/L)
    X = XV / V
    S = SV / V
    E = EV / V
    I = IV / V
    
    # Current feed rate (L/h)
    F_in = get_F_in(t, genes)
    
    # Kinetic rates
    mu = MU_MAX * S / (K_S + S)
    
    # mRNA/repressor intermediary states based on a Hill coefficient
    P = (I**HILL_N) / (K_I**HILL_N + I**HILL_N) if (K_I**HILL_N + I**HILL_N) > 0 else 0
    
    # --- ODE Mass Balances ---
    # Biomass (X): d(XV)/dt = mu * X * V - F_out * X
    dXV_dt = mu * X * V - F_OUT * X
    
    # Substrate (S): d(SV)/dt = F_in * S_in - (1/Y_XS) * mu * X * V - m_X * S * V * X - F_out * S
    dSV_dt = F_in * S_IN - (1/Y_XS) * mu * X * V - M_X * S * V * X - F_OUT * S
    
    # Volume (V): dV/dt = F_in(t) - F_out
    dV_dt = F_in - F_OUT
    
    # Enzyme Product (E): d(EV)/dt = k_act * P * V - mu * E * V
    dEV_dt = K_ACT_H * P * V - mu * E * V
    
    # Inducer/IPTG: tracking mass to find concentration P
    dIV_dt = F_in * I_IN - F_OUT * I
    
    return [dXV_dt, dSV_dt, dV_dt, dEV_dt, dIV_dt]

def simulate(genes):
    """Runs a 24-hour simulation for a specific feeding schedule"""
    # Initial states [XV, SV, V, EV, IV]
    # V0 = 1.0 L, X0 = 0.5 g/L, S0 = 2.0 g/L (lowered to prevent massive surplus), E0 = 0.0, I0 = 0.0
    y0 = [0.5 * 1.0, 2.0 * 1.0, 1.0, 0.0, 0.0] 
    
    # time points from 0 to 24 hours
    t_span = [0, 24]
    t_eval = np.linspace(0, 24, 200)
    
    # Solve ODE using LSODA due to non-smooth logic
    sol = solve_ivp(bioreactor_odes, t_span, y0, args=(genes,), t_eval=t_eval, method='LSODA', max_step=0.5)
    return sol

def evaluate(individual):
    """Evaluates the fitness of a specific feeding schedule (genes)"""
    genes = list(individual)
    
    sol = simulate(genes)
    if not sol.success:
        return (0.0,)
        
    SV = sol.y[1]
    V = sol.y[2]
    EV = sol.y[3]
    
    # Target enzyme mass at exactly t=24 hours
    final_enzyme_mass = EV[-1]
    
    fitness = final_enzyme_mass
    penalty = 0.0
    
    # Constraints 1: Volume must not exceed 5L (Initial V = 1L)
    max_V = np.max(V)
    if max_V > 5.0:
        return (0.0,) # "If Volume > 5 at the end of the run, make the score zero."
        
    # Constraint 2: Substrate dropping to 0 (Starvation)
    # Check if S goes below a very small number (e.g. 0.005 g/L) anywhere
    S = SV / V
    min_S = np.min(S)
    if min_S < 0.005:
        # Starvation occurring! Penalize inversely to how badly it starved 
        penalty += (0.005 - min_S) * 1e5
        
    final_score = max(0.0, fitness - penalty)
    
    return (final_score,)

def checkBounds(min_val, max_val):
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max_val:         child[i] = max_val
                    elif child[i] < min_val:       child[i] = min_val
            return offspring
        return wrapper
    return decorator

def setup_ga():
    """Sets up the DEAP GA environment"""
    # Prevent creator redefinition warnings if setup_ga is called multiple times
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Each gene: float representing F_in (L/h). Setting bounds 0.0 to 0.4
    toolbox.register("attr_float", random.uniform, 0.0, 0.2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Keep attributes within physical boundaries (F_in >= 0) and avoiding extremely high feeds (<= 0.5)
    toolbox.decorate("mate", checkBounds(0.0, 0.5))
    toolbox.decorate("mutate", checkBounds(0.0, 0.5))
    
    return toolbox

def optimize():
    print("Initializing Bioreactor GA Optimization...")
    toolbox = setup_ga()
    
    # Run specs: Population = 50, Generations = 20
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print("Running Genetic Algorithm Evolution...")
    # Run the genetic algorithm
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, 
                                       ngen=20, stats=stats, halloffame=hof, verbose=True)
                                       
    best_schedule = hof[0]
    best_fitness = best_schedule.fitness.values[0]
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Mathematically Optimal Feeding Schedule (L/h):")
    print(f"Hours  0-4 : {best_schedule[0]:.4f}")
    print(f"Hours  4-8 : {best_schedule[1]:.4f}")
    print(f"Hours  8-12: {best_schedule[2]:.4f}")
    print(f"Hours 12-16: {best_schedule[3]:.4f}")
    print(f"Hours 16-24: {best_schedule[4]:.4f}")
    print(f"Maximized Enzyme Mass at t=24h: {best_fitness:.2f} g")
    print("="*50 + "\n")
    
    # Plotting the winning schedule
    plot_results(best_schedule)

def plot_results(genes):
    sol = simulate(genes)
    t = sol.t
    
    XV = sol.y[0]
    SV = sol.y[1]
    V = sol.y[2]
    EV = sol.y[3]
    
    X = XV / V
    S = SV / V
    E = EV / V
    
    # Generate F_in across time
    F_in_t = [get_F_in(ti, genes) for ti in t]
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Substrate & Feed
    plt.subplot(2, 2, 1)
    plt.plot(t, S, 'y-', linewidth=2, label='Substrate [S] (g/L)')
    plt.plot(t, F_in_t, 'k--', label='Feed F_in (L/h)')
    plt.axhline(0.005, color='r', linestyle=':', label='Starvation Limit')
    plt.xlabel('Time (h)')
    plt.ylabel('S & Feed')
    plt.title('Substrate and Feeding rate')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Volume 
    plt.subplot(2, 2, 2)
    plt.plot(t, V, 'b-', linewidth=2, label='Volume (L)')
    plt.axhline(5.0, color='r', linestyle='--', label='Max Volume Limit (5L)')
    plt.xlabel('Time (h)')
    plt.ylabel('Volume (L)')
    plt.title('Bioreactor Volume [V]')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Biomass
    plt.subplot(2, 2, 3)
    plt.plot(t, X, 'g-', linewidth=2, label='Biomass [X] (g/L)')
    plt.xlabel('Time (h)')
    plt.ylabel('Biomass [X] (g/L)')
    plt.title('Biomass Concentration')
    plt.grid(True)
    plt.legend()
    
    # Plot 4: Enzyme Product
    plt.subplot(2, 2, 4)
    plt.plot(t, E, 'm-', linewidth=2, label='Enzyme [E] (g/L)')
    plt.xlabel('Time (h)')
    plt.ylabel('Enzyme (g/L)')
    plt.title('Target Enzyme Concentration [E]')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    optimize()
