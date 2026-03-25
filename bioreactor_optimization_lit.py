import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
import os
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

# --- System Constants ---
MU_MAX = 0.5
K_S = 0.1
Y_XS = 0.5
M_X = 0.01
S_IN = 20.0
I_IN = 10.0
K_I = 1.0
HILL_N = 2.1
F_OUT = 0.0

K_CAT_LITERATURE = {
    'Penicillin_G': 50,
    'Ampicillin': 25,
    'Amoxicillin': 20,
    'Cephalothin': 14,
    'Nitrocefin': 125,
    'Cefazolin': 7.5,
    'Cefuroxime': 1.25,
    'Cefotaxime': 0.05
}

def get_F_in(t, genes):
    if t < 4:
        return genes[0]
    elif t < 8:
        return genes[1]
    elif t < 12:
        return genes[2]
    elif t < 16:
        return genes[3]
    else:
        return genes[4]

def bioreactor_odes(t, state, genes, substrate='Nitrocefin'):
    XV, SV, V, EV, IV = state
    XV = max(0.0, XV)
    SV = max(0.0, SV)
    V = max(0.1, V)
    EV = max(0.0, EV)
    IV = max(0.0, IV)

    X = XV / V
    S = SV / V
    E = EV / V
    I = IV / V

    F_in = get_F_in(t, genes)
    mu = MU_MAX * S / (K_S + S) if (K_S + S) > 0 else 0.0
    P = (I**HILL_N) / (K_I**HILL_N + I**HILL_N) if (K_I**HILL_N + I**HILL_N) > 0 else 0.0

    K_CAT_S = K_CAT_LITERATURE.get(substrate, 125)
    K_ACT_H = K_CAT_S * 3600

    dXV_dt = mu * X * V - F_OUT * X
    dSV_dt = F_in * S_IN - (1 / Y_XS) * mu * X * V - M_X * S * V * X - F_OUT * S
    dV_dt = F_in - F_OUT
    dEV_dt = K_ACT_H * P * V - mu * E * V
    dIV_dt = F_in * I_IN - F_OUT * I

    return [dXV_dt, dSV_dt, dV_dt, dEV_dt, dIV_dt]

def simulate(genes, substrate='Nitrocefin'):
    y0 = [0.5, 2.0, 1.0, 0.0, 0.0]
    t_span = [0, 24]
    t_eval = np.linspace(0, 24, 200)
    sol = solve_ivp(
        bioreactor_odes,
        t_span,
        y0,
        args=(genes, substrate),
        t_eval=t_eval,
        method='LSODA',
        max_step=0.5
    )
    return sol

def evaluate(individual, substrate='Nitrocefin'):
    genes = list(individual)
    sol = simulate(genes, substrate=substrate)
    if not sol.success:
        return (0.0,)

    SV = sol.y[1]
    V = sol.y[2]
    EV = sol.y[3]

    final_enzyme_mass = EV[-1]
    penalty = 0.0

    if np.max(V) > 5.0:
        return (0.0,)

    S = SV / V
    min_S = np.min(S)
    if min_S < 0.005:
        penalty += (0.005 - min_S) * 1e5

    return (max(0.0, final_enzyme_mass - penalty),)

def checkBounds(min_val, max_val):
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max_val:
                        child[i] = max_val
                    elif child[i] < min_val:
                        child[i] = min_val
            return offspring
        return wrapper
    return decorator

def optimize_ga(substrate='Nitrocefin'):
    if not DEAP_AVAILABLE:
        return [0.1, 0.1, 0.1, 0.1, 0.1], None

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.0, 0.2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: evaluate(ind, substrate=substrate))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.decorate("mate", checkBounds(0.0, 0.5))
    toolbox.decorate("mutate", checkBounds(0.0, 0.5))

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.7, mutpb=0.2,
        ngen=20, stats=stats,
        halloffame=hof, verbose=False
    )

    return list(hof[0]), hof[0].fitness.values[0]

def plot_results(genes, substrate='Nitrocefin', save_path=None):
    sol = simulate(genes, substrate=substrate)
    t = sol.t
    XV, SV, V, EV = sol.y[0], sol.y[1], sol.y[2], sol.y[3]

    X = XV / V
    S = SV / V
    E = EV / V
    F_in_t = [get_F_in(ti, genes) for ti in t]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(t, S, 'y-', linewidth=2, label='Substrate [S] (g/L)')
    plt.plot(t, F_in_t, 'k--', label='Feed F_in (L/h)')
    plt.axhline(0.005, color='r', linestyle=':', label='Starvation Limit')
    plt.xlabel('Time (h)')
    plt.ylabel('S & Feed')
    plt.title(f'Substrate and Feeding rate: {substrate}')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t, V, 'b-', linewidth=2, label='Volume (L)')
    plt.axhline(5.0, color='r', linestyle='--', label='Max Volume Limit (5L)')
    plt.xlabel('Time (h)')
    plt.ylabel('Volume (L)')
    plt.title('Bioreactor Volume [V]')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(t, X, 'g-', linewidth=2, label='Biomass [X] (g/L)')
    plt.xlabel('Time (h)')
    plt.ylabel('Biomass [X] (g/L)')
    plt.title('Biomass Concentration')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(t, E, 'm-', linewidth=2, label='Enzyme [E] (g/L)')
    plt.xlabel('Time (h)')
    plt.ylabel('Enzyme (g/L)')
    plt.title('Target Enzyme Concentration [E]')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_all_substrates(save_dir='.'):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    for substrate, kcat in K_CAT_LITERATURE.items():
        best_schedule, best_fit = optimize_ga(substrate)
        sol = simulate(best_schedule, substrate=substrate)
        final_titer = sol.y[3][-1] / sol.y[2][-1]

        results.append({
            'substrate': substrate,
            'k_cat': kcat,
            'best_schedule': best_schedule,
            'best_fitness': best_fit,
            'final_titer': final_titer
        })

        plot_results(
            best_schedule,
            substrate=substrate,
            save_path=os.path.join(save_dir, f'{substrate}_timecourses.png')
        )

        print(f"{substrate}: k_cat={kcat}, final_titer={final_titer:.4f} g/L")

    return results

def plot_comparative_analysis(results, save_dir='.'):
    substrates = [r['substrate'] for r in results]
    kcats = [r['k_cat'] for r in results]
    titers = [r['final_titer'] for r in results]

    order = np.argsort(kcats)
    substrates = [substrates[i] for i in order]
    kcats = [kcats[i] for i in order]
    titers = [titers[i] for i in order]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(substrates)))

    plt.figure(figsize=(12, 6))
    bars = plt.bar(substrates, titers, color=colors)
    plt.ylabel('Final Enzyme Titer (g/L)')
    plt.xlabel('Substrate')
    plt.title('Comparative Analysis of Substrates in the Bioreactor Model')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    for bar, kcat in zip(bars, kcats):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'k_cat={kcat:g}',
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'substrate_comparative_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path

if __name__ == '__main__':
    results = run_all_substrates('.')
    comp_path = plot_comparative_analysis(results, '.')
    print(f"Comparative plot saved to: {comp_path}")
