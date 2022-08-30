import pandas as pd
import time
import datetime
import numpy as np
import random
import skillmatrix
import beschikbaarheid
import schedule
import EA
from tqdm import tqdm

df_skills, skills, all_persons = skillmatrix.get_matrix()
df_b = beschikbaarheid.get_beschikbaarheid()
available_persons = beschikbaarheid.get_all_available_persons(all_persons)

sched = schedule.Schedule(begin_date='2021-11-01', end_date='2021-12-26')
empty_schedule = sched.generate_empty()
df_final_availability = beschikbaarheid.fine_tuning_availability(empty_schedule, df_b)

# Initialize population

ea = EA.EA()
population = []
generations = 5
population_size = 20
k = 5

for i in range(population_size):
    sched_df_filled = sched.generate_random(empty_schedule, df_final_availability)
    population.append(sched_df_filled)


best_individual = None
best_fit = 0

for g in range(generations):
    fit_pop = ea.fitness(population, skills, sched)
    fit_pop = np.array(fit_pop)
    indices = np.argsort(fit_pop)

    # arg sort not working as expected ..... still work in progress
    assert (fit_pop[indices] == np.sort(fit_pop)).all()

    population = [population[i] for i in indices]
    pop = []
    for i in indices:
        pop.append(population[i])

    fit_pop = ea.fitness(pop, skills, sched)
    print('Gen: %s'%g,fit_pop)

    k_best, k_best_fitness =  ea.get_elite(population, fit_pop, k, skills, sched)
    recombinated_pop = ea.mutation(pop, df_final_availability, g)
    population = pop[-5:] + recombinated_pop[:15]
    #
    fit_offspring = ea.fitness(population, skills, sched)

    selected_pop, fit_selected_pop = ea.selection(population, fit_offspring, population_size - k)
    # Add elite to selected pop for next generation
    k_best.extend(recombinated_pop[:15])
    population = k_best
    # print(len(population))
    # print(ea.fitness(population, skills, sched))
    # print(len(population))
    all_fit_pop = (np.concatenate((fit_selected_pop, k_best_fitness))).tolist()
    # print(k_best_fitness)

    max_fit = max(all_fit_pop)
    max_index = all_fit_pop.index(max_fit)
    best = population[max_index]
    if max_fit > best_fit:
        best_fit = max_fit
        best_individual = best

max_value = best_fit
print('Highest possible score: 440')
print('max fitness achieved', max_value)
print('Accuracy: %.2f procent' % ((max_value / 440) * 100))
print(best_individual)
