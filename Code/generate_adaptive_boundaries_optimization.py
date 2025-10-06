import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import random
from deap import base, creator, tools
from deap import algorithms
import matplotlib.pyplot as plt
from sklearn.model_selection import  KFold
from scipy.integrate import simps
from scipy.signal import savgol_filter





# Function to train the model
def train_model(X,y, method):
    if method == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=50, 
            criterion = "absolute_error", 
            min_samples_split=10, 
            random_state=42)
        model.fit(X, y)
    elif method == "nearest_neighbors":
        model = KNeighborsRegressor(
            n_neighbors=10, 
            weights='uniform', 
            p=3)  
        model.fit(X, y)
    return model

# Function to calculate the loss
def calculate_loss(model, X_val, y_val):
    y_predict = model.predict(X_val)
    loss = np.sqrt(mean_squared_error(y_val, y_predict))
    return loss

# calculate spectra features in terms of wavelength, energy or energy -second order
def calculate_spectra_features_generic(filename, start_interval, end_interval, spectra_axis,  order):
    df_spectra = pd.read_csv(filename, header=None)
    spectra_xaxis = df_spectra[0].values
    if spectra_axis == 'energy':
        # Convert wavelength to energy (in eV)
        spectra_xaxis = 1239.8 / spectra_xaxis
    intensity = df_spectra[1].values
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())  
    if order == 0:
        intensity = savgol_filter(intensity, 25, 3)  # Apply Savitzky-Golay filter for smoothing
    else:
        intensity = savgol_filter(intensity, window_length = 51, polyorder=2, deriv=order)
    # Area under F4TCNQ neutral 
    mask = (spectra_xaxis >= start_interval) & (spectra_xaxis <= end_interval)
    xaxis_feature = spectra_xaxis[mask]
    intensity_feature = intensity[mask]
    if len(xaxis_feature) < 2:
        return 0.0  # Return 0 if there are fewer than two points in the selected range
    area_feature = simps(intensity_feature,xaxis_feature) 
    area_feature = abs(area_feature)   
    return area_feature

# Function to calculate area under the cureve for each boundary for the spectra
def calculate_features_for_boundaries(boundaries, measurement_files, spectra_axis,  order):
    measurement_spectra_features = []

    # Loop through each file path and experiment ID in the tuple
    for file_path, experiment_id in measurement_files:
        features = [experiment_id]  # Start with the experiment ID (foldername)
        for i in range(len(boundaries)-1):
            start_interval = boundaries[i]
            end_interval = boundaries[i+1]
            
            # Calculate the feature for this boundary range
            feature = calculate_spectra_features_generic(
                file_path, start_interval, end_interval, spectra_axis,  order)
            features.append(feature)
        
        measurement_spectra_features.append(features)

    columns = ['exp_id'] + [f'feature_{i+1}' for i in range(len(boundaries)-1)]
    df_measurement_features = pd.DataFrame(
        measurement_spectra_features, columns=columns)
    return df_measurement_features

# Function to generate boundaries in ascending order and within limits
def generate_ordered_boundaries(boundaries_limit, n_boundaries, spectra_axis ):
    lower, upper = boundaries_limit
    total_range = upper - lower
    min_dist = 5
    if spectra_axis == 'energy':
        min_dist = 0.004 
    free_space = total_range - (min_dist * (n_boundaries - 1))
    if free_space <= 0:
        raise ValueError("Not enough space to generate boundaries with the specified minimum distance.")
    random_numbers = sorted(random.uniform(lower, free_space) for _ in range(n_boundaries))
    boundaries = [ random_number +  i*min_dist for i, random_number in enumerate(random_numbers) ]
    boundaries = repair_boundaries(boundaries, boundaries_limit, spectra_axis)
    return boundaries

# Function to repair boundaries to ensure they are within limits and properly spaced    
def repair_boundaries(boundaries, boundaries_limit, spectra_axis):
    min_dist = 5
    if spectra_axis == 'energy':
        min_dist = 0.004
    lower, upper = boundaries_limit
    boundaries = sorted(np.clip(boundaries, lower, upper))

    for _ in range(10):
        modified = False
        for i in range(1, len(boundaries)):
            if boundaries[i] - boundaries[i - 1] < min_dist:
                new_val = boundaries[i-1] + min_dist
                if new_val > upper:
                    new_val = upper
                if new_val != boundaries[i]:
                    boundaries[i] = new_val
                    modified = True
        
        for i in range(len(boundaries) -1, 0, -1):
            if boundaries[i] - boundaries[i -1] < min_dist:
                new_val = boundaries[i] - min_dist
                if new_val < lower:
                    new_val = lower
                if new_val != boundaries[i-1]:
                    boundaries[i-1] = new_val
                    modified = True
        
        if not modified:
            break

    boundaries = sorted(np.clip(boundaries, lower, upper))  # Ensure boundaries are within limits after modification
    return boundaries

# Function to mutate boundaries
def mutate_boundaries(boundaries, boundaries_limit, spectra_axis):
    mutated = []
    for i in range(len(boundaries)):
        base_mutation = random.uniform(-200, 200)
        if spectra_axis == 'energy':
            base_mutation = random.uniform(-200/1239.8, 200/1239.8)  
        mutated_boundary = boundaries[i] + base_mutation
        mutated.append(mutated_boundary)
    mutated = repair_boundaries(mutated, boundaries_limit, spectra_axis)  
    mutated = creator.Boundaries(mutated)  
    return mutated, 

# Function to perform crossover between two boundaries
def crossover_boundaries(boundary1, boundary2, boundaries_limit, spectra_axis):
    cxpoint = random.randint(1, len(boundary1) - 1)
    offspring1 = boundary1[:cxpoint] + boundary2[cxpoint:]
    offspring2 = boundary2[:cxpoint] + boundary1[cxpoint:]
    offspring1 = repair_boundaries(offspring1, boundaries_limit, spectra_axis)  
    offspring2 = repair_boundaries(offspring2, boundaries_limit, spectra_axis)  
    offspring1 = creator.Boundaries(offspring1)  
    offspring2 = creator.Boundaries(offspring2)  
    return offspring1, offspring2

# Objective function to evaluate the fitness of a set of boundaries
def objective_function(boundaries, df, measurement_files, spectra_axis,  order, method):
    boundaries = np.array(boundaries)
    boundaries = np.sort(boundaries)
    df_measurement_features_derivative = calculate_features_for_boundaries(boundaries, measurement_files, spectra_axis,  order=2)
    df_measurement_features_orignal = calculate_features_for_boundaries(boundaries, measurement_files, spectra_axis,  order=0)
    df_measurement_features = pd.merge(df_measurement_features_derivative, df_measurement_features_orignal, on='exp_id', how='inner')
    df_features = pd.merge(df_measurement_features, df[['exp_id', 'CB', 'DCB', 'annealing_temperature', 'conductivity']], on='exp_id', how='inner')
    X = df_features.drop(columns=['exp_id', 'conductivity']).values  # All columns except exp_id and conductivity
    y = df_features['conductivity'].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_fitness = []

    for train_index, validation_index in kf.split(X):
        X_train, X_val = X[train_index], X[validation_index]
        y_train, y_val = y[train_index], y[validation_index]

        model = train_model(X_train, y_train, method)
        loss = calculate_loss(model, X_val, y_val)
        fold_fitness.append(loss)

    average_loss = np.mean(fold_fitness)
    loss_progress.append(average_loss)
    return average_loss, 

def custom_selection(boundaries, k, elite_rate= 0.3, tournsize=3):
    num_elite = int(k * elite_rate)
    elite = tools.selBest(boundaries, num_elite)  # Select elite individuals
    rest = tools.selTournament(boundaries, k - num_elite, tournsize=tournsize)  # Select the rest using tournament selection
    return elite + rest
# crossover and mutation
def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]  
    for i in range (1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], 
                                                        offspring[i])  
            del offspring[i-1].fitness.values , offspring[i].fitness.values 

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values 
    return offspring

def custom_ea_next_gen(population, toolbox, n , elitep , varp  , cxpb , mutpb):

    # generate 5 % from elite individuals
    elite = tools.selBest(population, max(1, int(len(population) * elitep)))  

    # generate 45 % using crossover and mutation from the rest of the population
    non_elite = [ind for ind in population if ind not in elite]  
    selected = tools.selTournament(non_elite, int(varp*len(population)), tournsize = 3)
    offspring = varAnd(selected, toolbox, cxpb=cxpb, mutpb=mutpb)  # Apply crossover and mutation to selected individuals

    #  generate 50 % from scratch
    new_individuals = [toolbox.boundaries() for _ in range(int(n - ((elitep + varp) * len(population))))]
    
    new_population = elite + offspring + new_individuals  
    return new_population

def custom_ea(population, toolbox, n , ngen, elitep, varp, cxpb, mutpb, 
              stats=None, halloffame=None, verbose=__debug__ ):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate individuals with invalid fitness
    invalid = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid)) 
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid), **record)
    if verbose:
        print(logbook.stream)
    
    for gen in range(1, ngen + 1):

        # custom offspring generation
        population = custom_ea_next_gen(population, toolbox, n, 
                                        elitep=elitep,
                                        varp=varp,
                                        cxpb=cxpb,
                                        mutpb=mutpb)
        
        invalid = [ind for ind in population if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid)) 
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)
        
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def run_ga(measurement_files, df,  boundaries_limit, n_boundaries, 
           spectra_axis = 'wavelength',  order = 2,  method="random_forest", 
           n = 100, ngen =50, cxpb = 0.7, mutpb = 0.3, 
           elitep = 0.05, varp = 0.45):

    
    """
    The run_ga function runs a genetic algorithm (GA) to optimize boundaries for a given problem.
    
    Parameters:
    - measurement_files: Files used to evaluate the fitness of the boundaries.
    - df: DataFrame containing relevant data for evaluation.
    - boundaries_limit: The min and max limits for boundary values.
    - n_boundaries: The number of boundaries in each individual solution.
    - spectra_axis: The axis type for spectra ('wavelength' or 'energy', default is 'wavelength').
    - order: The order of the derivative to use for feature calculation (default is 2).
    - method: The method to use for evaluation (default is "random_forest").
    - n: The population size for the GA (default is 100).
    - ngen: The number of generations for the GA (default is 50).
    - cxpb: The probability of mating two individuals (default is 0.7).
    - mutpb: The probability of mutating an individual (default is 0.1).
    - elitep: The proportion of elite individuals to carry over to the next generation (default is 0.05).
    - varp: The proportion of individuals generated through crossover and mutation (default is 0.45).
    
    Returns:
    - optimal_boundaries: The best set of boundaries found by the GA.
    - (population, logbook): The final population and logbook containing GA statistics.
    - loss_progress: A list of the best fitness (loss) over generations.
    """

    global loss_progress 
    loss_progress = []  # Initialize loss progress for this run
    # Create the DEAP framework
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize the fitness function
    creator.create("Boundaries", list, fitness=creator.FitnessMin)  # Create a Boundaries class inheriting from list


    toolbox = base.Toolbox()
    toolbox.register("boundaries", tools.initIterate, creator.Boundaries, 
                     lambda: generate_ordered_boundaries(boundaries_limit, n_boundaries, spectra_axis)) # Generate initial boundaries for each individual
    toolbox.register("population", tools.initRepeat, list, toolbox.boundaries)
    toolbox.register("mate", crossover_boundaries, 
                     boundaries_limit=boundaries_limit, spectra_axis=spectra_axis)  # Crossover function for boundaries
    toolbox.register("mutate", mutate_boundaries, 
                     boundaries_limit=boundaries_limit, spectra_axis=spectra_axis)  # Mutation function for boundaries
    toolbox.register("evaluate", objective_function,df=df, 
                     measurement_files=measurement_files, spectra_axis = spectra_axis, 
                     order = order,  method=method)

    # Create population
    population = toolbox.population(n=n)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)


    # GA parameters
    ngen = ngen
    cxpb = cxpb
    mutpb = mutpb

    
    population , logbook = custom_ea(population, toolbox, n=n, ngen=ngen, 
                                     elitep=elitep, varp=varp, cxpb=cxpb, 
                                     mutpb=mutpb, stats=stats, halloffame=hof, verbose=True)
                                        
   
    best_fitness = [logbook.select("min")[i] for i in range(ngen)]

    # Plot the best fitness (loss) over generations after the GA finishes
    plt.plot(range(ngen), best_fitness)
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value (Loss)')
    plt.show()

    optimal_boundaries = hof[0]
    if spectra_axis == 'energy':
        optimal_boundaries = np.sort(np.round(optimal_boundaries, 4))
    else:
        optimal_boundaries = np.sort(np.round(optimal_boundaries, 2).astype(int))
    
    print(f"Optimal Boundaries: {optimal_boundaries}")
    return optimal_boundaries, (population, logbook), loss_progress

