import random
import pandas as pd
import numpy as np
import math
import pickle
import seaborn as sns
import statistics as st
import os
import matplotlib.pyplot as plt
import multiprocessing   # used to improve time efficiency
import parameters as par

from tqdm import tqdm   # used for time management
from scipy import stats
from scipy.stats import distributions as dist
from numpy.random import choice

fig = plt.figure(figsize=(6,4), dpi=300)

# Set seed for reproducibility
np.random.seed(420)

def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1
    return n

def importance_sampling(a,a2):
    # Create alternative beta dist A -- g(x)
    g_xx = stats.beta(a, 1)
    g_yy = stats.beta(a2, 1)

    # generate samples from this beta dist
    g_x = stats.beta.rvs(a, 1, size=par.NO_SAMPLES)
    g_y = stats.beta.rvs(a2, 1, size=par.NO_SAMPLES)

    # Create random samples from uniform dist B --  f(x)
    f_x = np.random.uniform(low=0, high=1, size=par.NO_SAMPLES)
    f_y = np.random.uniform(low=0, high=1, size=par.NO_SAMPLES)

    weights_x = g_x / f_x
    weights_y = g_y / f_y

    # Evaluate samples from dist B, on dist A
    evaluate_x = g_xx.pdf(f_x)
    evaluate_y = g_yy.pdf(f_y)

    # Determine importance
    importance_x = evaluate_x / weights_x
    importance_x_dist = importance_x / sum(importance_x)
    importance_y = evaluate_y / weights_y
    importance_y_dist = importance_y / sum(importance_y)

    # Draw new samples from importance sampled dist
    new_x = choice(g_x, par.NO_SAMPLES,
                   p=importance_x_dist)
    new_y = choice(g_y, par.NO_SAMPLES,
                   p=importance_y_dist)

    return new_x, new_y


def pure_random(max_iter):
    sample_area_count = 0
    for i in range(par.NO_SAMPLES):

        # Draw random numbers from uniform distribution
        x = np.random.uniform(low=par.RE_START, high=par.RE_END)
        y = np.random.uniform(low=par.IM_START, high=par.IM_END)

        # Convert coordinates to complex number and compute the mandelbrot iterations
        c = complex(x,y)
        # Compute the number of iterations
        m = mandelbrot(c, max_iter)
        # The color depends on the number of iterations
        color = 255 - int(m * 255 / max_iter)
        if color == 0:
            sample_area_count += 1

    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


# importance sampled pure random
def imp_pure_random(max_iter):
    sample_area_count = 0

    new_x, new_y = importance_sampling(par.a, par.a2)

    for value_x, value_y in zip(new_x,new_y):
        x = value_x
        y = value_y
        # Convert coordinates to complex number and compute the mandelbrot iterations
        c = complex(par.RE_START + x * (par.RE_END - par.RE_START),
                    par.IM_START + y * (par.IM_END - par.IM_START))
        # Compute the number of iterations
        m = mandelbrot(c, max_iter)
        # The color depends on the number of iterations
        color = 255 - int(m * 255 / max_iter)
        if color == 0:
            sample_area_count += 1

    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


def orth_sampling(max_iter):
    major = int(math.sqrt(par.NO_SAMPLES))
    sample_area_count = 0
    m = 0
    xlist = np.empty((major, major))
    ylist = np.empty((major, major))
    scale_x = (abs(par.RE_START) + abs(par.RE_END)) / par.NO_SAMPLES
    scale_y = (abs(par.IM_START) + abs(par.IM_END)) / par.NO_SAMPLES

    for i in range(0, major):
        for j in range(0, major):
            xlist[i][j] = ylist[i][j] = m
            m += 1

    np.random.shuffle(xlist)
    np.random.shuffle(ylist)

    for i in range(0, major):
        for j in range(0, major):
            x = par.RE_START + (scale_x * (xlist[i][j]+ np.random.uniform(0,1)))
            y = par.IM_START + (scale_y * (ylist[j][i]+ np.random.uniform(0,1)))

            c = complex(x, y)
            # Compute the number of iterations
            m = mandelbrot(c, max_iter)
            # The color depends on the number of iterations
            color = 255 - int(m * 255 / max_iter)
            if color == 0:
                sample_area_count += 1
                #plt.plot(x, y, 'ro', color='green')
            #else:
                #plt.plot(x, y, 'ro', color='red')
            #plt.axvline(x=(par.RE_START + (scale_x * (xlist[i][j]))), color='black')
            #plt.axhline(y= (par.IM_START + (scale_y * (ylist[j][i]))), color='black')

    #plt.savefig('exampleplotorthsamp.jpg')
    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


# importance sampled orth sampling
def imp_orth_sampling(max_iter):
    major = int(math.sqrt(par.NO_SAMPLES))
    sample_area_count = 0
    m = 0
    xlist = np.empty((major, major))
    ylist = np.empty((major, major))
    scale_x = (abs(par.RE_START) + abs(par.RE_END)) / par.NO_SAMPLES
    scale_y = (abs(par.IM_START) + abs(par.IM_END)) / par.NO_SAMPLES

    for i in range(0, major):
        for j in range(0, major):
            xlist[i][j] = ylist[i][j] = m
            m += 1

    np.random.shuffle(xlist)
    np.random.shuffle(ylist)

    new_x, new_y = importance_sampling(par.a, par.a2)
    noise_count = 0

    for i in range(0, major):
        for j in range(0, major):
            x = par.RE_START + (scale_x * (xlist[i][j]+ new_x[noise_count]))
            y = par.IM_START + (scale_y * (ylist[j][i]+ new_y[noise_count]))
            noise_count += 1

            c = complex(x, y)
            # Compute the number of iterations
            m = mandelbrot(c, max_iter)
            # The color depends on the number of iterations
            color = 255 - int(m * 255 / max_iter)
            if color == 0:
                sample_area_count += 1
                #plt.plot(x, y, 'ro', color='green')
            #else:
                #plt.plot(x, y, 'ro', color='red')
            #plt.axvline(x=(par.RE_START + (scale_x * (xlist[i][j]))), color='black')
            #plt.axhline(y= (par.IM_START + (scale_y * (ylist[j][i]))), color='black')

    #plt.savefig('exampleplotorthsamp.jpg')
    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


def lhs(max_iter):
    sample_area_count = 0
    m = 0
    xlist = np.zeros(par.NO_SAMPLES)
    ylist = np.zeros(par.NO_SAMPLES)
    scale_x = (abs(par.RE_START) + abs(par.RE_END)) / par.NO_SAMPLES
    scale_y = (abs(par.IM_START) + abs(par.IM_END)) / par.NO_SAMPLES

    for i in range(0, par.NO_SAMPLES):
        xlist[i] = ylist[i] = m
        m += 1

    np.random.shuffle(xlist)
    np.random.shuffle(ylist)

    for i in xlist:
        x = par.RE_START + scale_x * (xlist[int(i)] + np.random.uniform(0, 1))
        y_index = int(np.random.randint(0, len(ylist)))
        y = par.IM_START + scale_y * (ylist[y_index] + np.random.uniform(0, 1))
        #plt.axvline(x=(par.RE_START + (scale_x * (xlist[int(i)]))), color='black')
        #plt.axhline(y=(par.IM_START + (scale_y * (ylist[y_index]))), color='black')
        ylist = np.delete(ylist, y_index)
        c = complex(x, y)
        # Compute the number of iterations
        m = mandelbrot(c, max_iter)
        # The color depends on the number of iterations
        color = 255 - int(m * 255 / max_iter)
        if color == 0:
            sample_area_count += 1
            #plt.plot(x, y, 'ro', color='green', ms=72. / fig.dpi)
        #else:
            #plt.plot(x, y, 'ro', color='red', ms=72. / fig.dpi)

    #plt.show()
    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


# importance sampled, latin hypercube
def imp_lhs(max_iter):
    sample_area_count = 0
    m = 0
    xlist = np.zeros(par.NO_SAMPLES)
    ylist = np.zeros(par.NO_SAMPLES)
    scale_x = (abs(par.RE_START) + abs(par.RE_END)) / par.NO_SAMPLES
    scale_y = (abs(par.IM_START) + abs(par.IM_END)) / par.NO_SAMPLES

    for i in range(0, par.NO_SAMPLES):
        xlist[i] = ylist[i] = m
        m += 1

    np.random.shuffle(xlist)
    np.random.shuffle(ylist)

    noise_x, noise_y = importance_sampling(par.a, par.a2)
    noise_count = 0
    for i in xlist:
        x = par.RE_START + scale_x * (xlist[int(i)] + noise_x[noise_count])
        y_index = int(np.random.randint(0, len(ylist)))
        y = par.IM_START + scale_y * (ylist[y_index] + noise_y[noise_count])
        noise_count += 1
        #plt.axvline(x=(par.RE_START + (scale_x * (xlist[int(i)]))), color='black')
        #plt.axhline(y=(par.IM_START + (scale_y * (ylist[y_index]))), color='black')
        ylist = np.delete(ylist, y_index)
        c = complex(x, y)
        # Compute the number of iterations
        m = mandelbrot(c, max_iter)
        # The color depends on the number of iterations
        color = 255 - int(m * 255 / max_iter)
        if color == 0:
            sample_area_count += 1
            #plt.plot(x, y, 'ro', color='green', ms=72. / fig.dpi)
        #else:
            #plt.plot(x, y, 'ro', color='red', ms=72. / fig.dpi)

    #plt.show()
    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


# Function used to calculate the differences A_is - A_js , indicating convergence
def cpu_spread(process_range, function):
    areas_js = []
    areas_diff = []
    area_is = function(par.MAX_ITER)
    for j in tqdm(process_range): # van 1 tot 200 max iteraties
        area_js = function(j)
        areas_js.append(area_js)
        area_diff = abs(area_js - area_is)
        areas_diff.append(area_diff)
    return areas_js, areas_diff


# Function used to calculate the statistics for every method, iterations mandelbrot kept fixed
def cpu_spread_taking_maxiter(process_range, function):
    areas = []
    for j in tqdm(process_range):
        area = function(par.MAX_ITER)
        areas.append(area)

    avg_area = st.mean(areas)
    std_area = np.std(areas)
    ci_upper = stats.t.interval(alpha=par.alpha, df=len(areas)-1, loc=avg_area, scale=np.std(areas))[1]
    ci_down = stats.t.interval(alpha=par.alpha, df=len(areas)-1, loc=avg_area, scale=np.std(areas))[0]

    return avg_area, std_area, areas, ci_upper, ci_down


def plot_area_convergence(result_dict):
    results = ["pure_random", "orth_sampling", "lhs"]
    results2 = ["imp_pure_random", "imp_orth_sampling", "imp_lhs"]
    # plot total area
    for idx, i in enumerate(results):
        plt.plot(result_dict["iterations"], result_dict[i]["area_js"], label="%s"%i)
    plt.xlabel('Max iterations Mandelbrot', fontsize=12)
    plt.ylabel('Area approximation', fontsize=12)
    plt.legend()
    plt.savefig('A_js_%s_%s.jpg'%(par.NO_SAMPLES, par.MAX_ITER))
    plt.clf()

    for idx, i in enumerate(results2):
        plt.plot(result_dict["iterations"], result_dict[i]["area_js"], label="%s"%i)
    plt.xlabel('Max iterations Mandelbrot', fontsize=12)
    plt.ylabel('Area approximation', fontsize=12)
    plt.legend()
    plt.savefig('A_js_%s_%s_imp.jpg'%(par.NO_SAMPLES, par.MAX_ITER))
    plt.clf()

    # plot diff area
    for idx, i in enumerate(results):
        plt.plot(result_dict["iterations"], result_dict[i]["area_diff"], label="%s"%i)
    plt.xlabel('Max iterations Mandelbrot', fontsize=12)
    plt.ylabel('A_js - A_is', fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig('A_diff_%s_%s.jpg' % (par.NO_SAMPLES, par.MAX_ITER))
    plt.clf()
    for idx, i in enumerate(results2):
        plt.plot(result_dict["iterations"], result_dict[i]["area_diff"], label="%s"%i)
    plt.xlabel('Max iterations Mandelbrot', fontsize=12)
    plt.ylabel('A_js - A_is', fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig('A_diff_%s_%s_imp.jpg' % (par.NO_SAMPLES, par.MAX_ITER))
    plt.clf()

    for idx, i in enumerate(results):
        plt.plot(result_dict["iterations"], result_dict[i]["area_diff"], label="%s"%i)
    plt.xlim(600, 1000)
    plt.ylim(-0.01, 0.1)
    plt.xlabel('Max iterations Mandelbrot', fontsize=12)
    plt.ylabel('A_js - A_is', fontsize=12)
    plt.savefig('A_diff_%s_%s_last200.jpg' % (par.NO_SAMPLES, par.MAX_ITER))
    plt.clf()

    for idx, i in enumerate(results2):
        plt.plot(result_dict["iterations"], result_dict[i]["area_diff"], label="%s"%i)
    plt.xlim(600, 1000)
    plt.ylim(-0.01, 0.1)
    plt.xlabel('Max iterations Mandelbrot', fontsize=12)
    plt.ylabel('A_js - A_is', fontsize=12)
    plt.savefig('A_diff_%s_%s_last200_imp.jpg' % (par.NO_SAMPLES, par.MAX_ITER))
    plt.clf()
    return


def perform_stat_analysis(stats_dict):
    # plot the distributions - based on this make some assumptions for testing
    sns.distplot(stats_dict["pure_random"]["areas"], label='pure_random')
    sns.distplot(stats_dict["orth_sampling"]["areas"], label='orth_sampling')
    sns.distplot(stats_dict["lhs"]["areas"], label='lhs')
    plt.xlabel('Mandelbrot Area')
    plt.ylabel('No. Samples')
    plt.legend()
    plt.savefig('distplot.jpg')
    plt.clf()
    sns.distplot(stats_dict["imp_pure_random"]["areas"], label='imp_pure_random')
    sns.distplot(stats_dict["imp_orth_sampling"]["areas"], label='imp_orth_sampling')
    sns.distplot(stats_dict["imp_lhs"]["areas"], label='imp_lhs')
    plt.xlabel('Mandelbrot Area')
    plt.ylabel('No. Samples')
    plt.legend()
    plt.savefig('distplot2.jpg')

    methods = ["pure_random", "orth_sampling", "lhs", "imp_pure_random", "imp_orth_sampling", "imp_lhs"]
    # get all possible pairs for F-testing
    all_pairs = [(methods[i], methods[j]) for i in range(len(methods)) for j in range(i + 1, len(methods))]

    # Independent t tests - Welsh
    for pair in all_pairs:
        t_test = stats.ttest_ind(stats_dict["%s"%pair[0]]["areas"], stats_dict["%s"%pair[1]]["areas"])
        print('t test for %s and %s is %s' %(pair[0], pair[1], t_test))

    # Get results table
    df = pd.DataFrame.from_dict(stats_dict)
    df.drop(df.tail(1).index, inplace=True)# Skip last row with all areas
    df.to_excel("output.xlsx")
    print(df)
    return


# Function used to approximate a distribution for importance sampling
# Attempt has been made to find this distribution by analyzing different beta dists
def rand_sample_approx_function():
    v = {
        "a": [],
        "b": [],
        "area": []
    }
    for a in np.arange(1, 4, 0.2):
        for a2 in np.arange(1, 4, 0.2):
            par.a = a
            par.a2 = a2
            v["area"].append(abs(imp_pure_random(par.MAX_ITER)- 1.506793))  # 1.506... based on Mitchell K., 2001
            v["a"].append(a) ; v["b"].append(b)

    index_min = np.argmin(v["area"])
    return v["a"][index_min], v["b"][index_min], v["area"][index_min]


run_ajs_convergence = True
run_stats_max_iter = True
run_ajs_conv_plotting = True
run_statistical_analysis = True


results = {
    "iterations" : [],
    "pure_random": [],
    "orth_sampling": [],
    "lhs" : [],
    "imp_pure_random": [],
    "imp_orth_sampling": [],
    "imp_lhs": []}

stats_dict ={"pure_random": [],
    "orth_sampling": [],
    "lhs" : [],
    "imp_pure_random": [],
    "imp_orth_sampling": [],
    "imp_lhs": [],
    "stat_sample_size": 0
}


if __name__ == '__main__':
    methods = [imp_pure_random, imp_orth_sampling, imp_lhs, pure_random, orth_sampling, lhs]
    pool = multiprocessing.Pool(os.cpu_count())
    if run_ajs_convergence:
        iterations = list(range(1, par.MAX_ITER + 1, 1))
        for method in methods:
            print(str(method))
            areas_js, areas_diff = pool.apply(cpu_spread, (iterations, method, ))
            label = str(method.__name__)
            results[label] = {"area_js": areas_js,
                              "area_diff": areas_diff}
        results["iterations"] = iterations

        # Save data to pickle file so we can plot without running the algo's again
        file_to_write = open("results.pickle", "wb")
        pickle.dump(results, file_to_write)
        file_to_write.close()

    if run_stats_max_iter:
        stat_sample_size = par.STAT_SAMPLE_SIZE # 100 sample runs
        iterations = list(range(1, stat_sample_size + 1, 1))
        for method in methods:
            print(str(method))
            pool = multiprocessing.Pool(os.cpu_count())
            avg_area, std_area, areas, ci_up, ci_down = pool.apply(cpu_spread_taking_maxiter, (iterations, method,))
            label = str(method.__name__)
            stats_dict[label] = {"avg_area": avg_area,
                              "std_area": std_area,
                                 "ci_upper": ci_up,
                                 "ci_down": ci_down,
                                 "areas": areas}
        stats_dict["stat_sample_size"] = stat_sample_size
        file_to_write = open("stats.pickle", "wb")
        pickle.dump(stats_dict, file_to_write)
        file_to_write.close()
        table = pd.DataFrame.from_dict(stats_dict)
        print(table)
    pool.close()

    if run_ajs_conv_plotting:
        infile = open("results.pickle", 'rb')
        new_dict_results = pickle.load(infile)
        plot_area_convergence(new_dict_results)

    if run_statistical_analysis:
        infile = open("stats.pickle", 'rb')
        new_dict_stats = pickle.load(infile)
        perform_stat_analysis(new_dict_stats)




