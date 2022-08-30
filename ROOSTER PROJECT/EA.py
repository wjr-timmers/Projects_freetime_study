import pandas as pd
import time
import datetime
import numpy as np
import random

class EA:
    def __init__(self):
        pass

    def fitness_function(self, schedule, skill, sched):
        positive = 0
        negative = 0
        total = 0
        TASKS = ['WN-Balie', 'HG-Balie1', 'HG-Balie2',
                 'OCC', 'CSC', 'SD1', 'SD2', 'SD3', 'SD4', 'SD5', 'SD6']

        for idx in range(len(schedule)):
            for t in TASKS:
                total += 1
                # Double check if there are duplicates
                plan = schedule.loc[idx, ['WN-Balie', 'HG-Balie1', 'HG-Balie2', 'OCC', 'CSC', 'SD1',
                                          'SD2', 'SD3', 'SD4', 'SD5', 'SD6']]
                if sched.check_duplicates(plan.tolist()) == True:
                    print(plan)
                    raise ValueError('There is a double scheduling')

                person = schedule.at[idx, t]

                if t == 'WN-Balie' and (person in skill['wn']) and person != 'None':
                    positive += 1
                if (t == 'HG-Balie1' or  t == 'HG-Balie2' ) and (person in skill['hg']) and person != 'None':
                    positive += 1
                if t == 'OCC' and (person in skill['occ']) and person != 'None':
                    positive += 1
                if t == 'CSC' and (person in skill['csc']) and person != 'None':
                    positive += 1
                if (t == 'SD1' or  t == 'SD2' or t == 'SD3' or t == 'SD4'
                    or t == 'SD5'  or t == 'SD6'  ) and (person in skill['sd']) and person != 'None':
                    positive += 1
                if person == 'None':
                    positive += 1
                else:
                    negative -= 1

        fitness = positive  # + negative
        return fitness

    def fitness(self, population_f, skill, sched):
        fitness_list = []
        for pf in population_f:
            fitness_list.append(int(self.fitness_function(pf, skill, sched)))
        return fitness_list

    def mutate_part_afternoon(self, idx, whole_day, afternoon_day, morning_day, p, availability):
        if len(afternoon_day) == 0 or afternoon_day == 'None':
            return p
        elif len(afternoon_day) == 1:
            go_to_work = afternoon_day[0]
        else:
            go_to_work = random.choice(afternoon_day)

        afternoon_day.remove(go_to_work)
        if go_to_work in whole_day:
            whole_day.remove(go_to_work)
        if go_to_work in morning_day:
            morning_day.remove(go_to_work)
        p.at[idx, "Rest_Whole_day"] = whole_day
        p.at[idx, "Rest_Afternoon"] = afternoon_day
        p.at[idx, "Rest_Morning"] = morning_day

        job_to_do = 'HG-Balie2'
        removed_person = p.at[idx, job_to_do]
        p.at[idx, job_to_do] = go_to_work
        if removed_person in availability.at[idx, "Whole Day A"]:
            p.at[idx, "Rest_Whole_day"].append(removed_person)
        if removed_person in availability.at[idx, "Morning A"]:
            p.at[idx, "Rest_Morning"].append(removed_person)
        if removed_person in availability.at[idx, "Afternoon A"]:
            p.at[idx, "Rest_Afternoon"].append(removed_person)
        return p


    def mutate_part_morning(self, idx, whole_day, afternoon_day, morning_day, p, availability):
        if len(morning_day) == 0 or morning_day == 'None':
            return p
        elif len(morning_day) == 1:
            go_to_work = morning_day[0]
        else:
            go_to_work = random.choice(morning_day)

        morning_day.remove(go_to_work)
        if go_to_work in whole_day:
            whole_day.remove(go_to_work)
        if go_to_work in afternoon_day:
            afternoon_day.remove(go_to_work)
        p.at[idx, "Rest_Whole_day"] = whole_day
        p.at[idx, "Rest_Afternoon"] = afternoon_day
        p.at[idx, "Rest_Morning"] = morning_day

        job_to_do = 'HG-Balie1'
        removed_person = p.at[idx, job_to_do]
        p.at[idx, job_to_do] = go_to_work
        if removed_person in availability.at[idx, "Whole Day A"]:
            p.at[idx, "Rest_Whole_day"].append(removed_person)
        if removed_person in availability.at[idx, "Morning A"]:
            p.at[idx, "Rest_Morning"].append(removed_person)
        if removed_person in availability.at[idx, "Afternoon A"]:
            p.at[idx, "Rest_Afternoon"].append(removed_person)
        return p

    def mutate_part_whole_day(self, idx, whole_day, afternoon_day, morning_day, p, availability):
        if len(whole_day) == 0 or whole_day == 'None':
            return p
        elif len(whole_day) == 1:
            go_to_work = whole_day[0]
        else:
            go_to_work = random.choice(whole_day)
        whole_day.remove(go_to_work)
        if go_to_work in morning_day:
            morning_day.remove(go_to_work)
        if go_to_work in afternoon_day:
            afternoon_day.remove(go_to_work)
        p.at[idx, "Rest_Whole_day"] = whole_day
        p.at[idx, "Rest_Afternoon"] = afternoon_day
        p.at[idx, "Rest_Morning"] = morning_day

        whole_day_tasks = ['WN-Balie', 'HG-Balie1', 'HG-Balie2',
                           'OCC', 'CSC', 'SD1', 'SD2', 'SD3', 'SD4', 'SD5', 'SD6']
        job_to_do = random.choice(whole_day_tasks)
        removed_person = p.at[idx, job_to_do]
        p.at[idx, job_to_do] = go_to_work
        if removed_person in availability.at[idx, "Whole Day A"]:
            p.at[idx, "Rest_Whole_day"].append(removed_person)
        if removed_person in availability.at[idx, "Morning A"]:
            p.at[idx, "Rest_Morning"].append(removed_person)
        if removed_person in availability.at[idx, "Afternoon A"]:
            p.at[idx, "Rest_Afternoon"].append(removed_person)
        return p

    def cross_shifts(self, idx, p):
        # Still excludes HG
        whole_day_tasks = ['WN-Balie',
                           'OCC', 'CSC', 'SD1', 'SD2', 'SD3', 'SD4', 'SD5', 'SD6']

        job1 = random.choice(whole_day_tasks)
        whole_day_tasks.remove(job1)
        job2 = random.choice(whole_day_tasks)
        p.loc[idx ,[job1 ,job2]] = p.loc[idx ,[job2 ,job1]].values
        return p

    def mutation(self, population_i, availability, g):
        TASKS_B = ['WN-Balie', 'HG-Balie1', 'HG-Balie2',
                   'OCC', 'CSC', 'SD1', 'SD2', 'SD3', 'SD4', 'SD5', 'SD6',
                   'Rest_Whole_day', 'Rest_Morning', 'Rest_Afternoon']

        for p in population_i:
            for idx in range(len(p)):
                if g <= 20:
                    p = self.cross_shifts(idx, p)
                    p = self.cross_shifts(idx, p)
                else:
                    p = self.mutate_part_whole_day(idx, p.at[idx, "Rest_Whole_day"],
                                                   p.at[idx, "Rest_Afternoon"],
                                                   p.at[idx, "Rest_Morning"],
                                                   p, availability)
                    p = self.mutate_part_morning(idx, p.at[idx, "Rest_Whole_day"],
                                                 p.at[idx, "Rest_Afternoon"],
                                                 p.at[idx, "Rest_Morning"],
                                                 p, availability)
                    p = self.mutate_part_afternoon(idx, p.at[idx, "Rest_Whole_day"],
                                                   p.at[idx, "Rest_Afternoon"],
                                                   p.at[idx, "Rest_Morning"],
                                                   p, availability)
                    p = self.cross_shifts(idx, p)
        return population_i

    def get_elite(self, pop, fit_pop, k, skill, sched):
        sort_indices = np.argsort(fit_pop)
        best_indices = sort_indices[-k:]
        selected_pop = []
        for b in best_indices:
            selected_pop.append(pop[b])
        fit_selected_pop = np.take(fit_pop, best_indices).tolist()
        return selected_pop, fit_selected_pop

    def selection(self, pop, fit_pop, npop):
        # selection
        sort_indices = np.argsort(fit_pop)
        best_indices = sort_indices[-npop:]
        selected_pop = []
        for b in best_indices:
            selected_pop.append(pop[b])
        fit_selected_pop = np.take(fit_pop, best_indices).tolist()

        return selected_pop, fit_selected_pop

    def recombination(self, pop):
        samples = random.sample(range(1, len(pop)), 4)
        idx1 = samples[0]
        idx2 = samples[1]
        idx3 = samples[2]
        idx4 = samples[3]
        recomb_operator = random.randint(2, 11)

        for idx in range(len(pop[0])):
            if idx % recomb_operator == 0:
                change_var = pop[idx1].loc[idx]
                pop[idx1].loc[idx] = pop[idx2].loc[idx]
                pop[idx2].loc[idx] = change_var

        for idx in range(len(pop[0])):
            if idx % recomb_operator == 0:
                change_var = pop[idx3].loc[idx]
                pop[idx3].loc[idx] = pop[idx4].loc[idx]
                pop[idx4].loc[idx] = change_var

        return pop