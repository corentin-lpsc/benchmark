"""
utils.py

This module provides utility functions for benchmarking and memory profiling.

Functions:
----------
init_benchmark_results():
    Returns a dictionary to store the benchmarking results for different matrix operations.

format_memory(mem_usage):
    Takes a memory usage value in bytes and converts it to a more readable format (KB, MB, GB).

timer(operation, times, memories=None):
    A timer function that measures the execution time and memory usage of a given operation.

print_benchmark_results(benchmark_results, sizes, show_memory_usage=True):
    Prints the benchmarking results in human readable way.

plot_CPU_time_benchmark(sizes, benchmark_results, title='Time Comparison', xlabel='Matrix Size', ylabel='Time (s)'):
    Plots the CPU time benchmarking results.
"""

import timeit
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
from typing import List, Callable

# Structure de données suggérée pour conserver les résultats. On aurait pu passer par une classe.
def init_benchmark_results():
    """
    Returns a dictionary with the following structure"""
    benchmark_results = {
        'coo': {
            'times': [],
            'memories': []
        },
        'numpy': {
            'times': [],
            'memories': []
        },
        'scipy': {
            'times': [],
            'memories': []
        }
    }
    return benchmark_results

def format_memory(mem_usage):
    units = ['bytes', 'KB', 'MB', 'GB']
    unit = 0
    while mem_usage > 1024 and unit < len(units) - 1:
        mem_usage /= 1024
        unit += 1
    return f"{mem_usage:.2f} {units[unit]}"

def timer(operation: 'Callable', times: 'List', memories: 'List'=None) -> None:
    """
    An advanced timer. Note that for very fast operations, the timer's overhead might be significant compared to the computation time. In such cases, feel free to retest using %timeit or modify the function.

    Use case:
    memories = None: memory is not measured, avoiding unnecessary overhead.
    memories = []: memory is measured, and the results are stored in the memories list. => low computation times measured may not be representative of reality.

    Arguments:
    ----------
    operation: the function to be called (without parentheses)
        You can use a lambda function to pass arguments to the function. For example: lambda: my_function(arg1, arg2)

    times: list
        List in which computation times are stored. We add np.nan if the computation time exceeds 1 second.

    memories: list or None
        List in which used memory is stored. We add np.nan if the computation time exceeds 1 second.
        If None, memory is not measured.

    Returns:
    --------
    None

    Exemple:
    --------
    >>> times=[]
    >>> timer(lambda : max([1, 20, 100]), times)
    >>> print(times)
    [1.6709964256733656e-06]
    """
    if not times or times[-1]<1:
        operation()
        start = timeit.default_timer()
        if memories is not None:
            mem=memory_usage(operation, interval=0.1)
            memories.append(max(mem)-min(mem))
        else:
            operation()
        times.append(timeit.default_timer() - start)
    else : 
        times.append(np.nan)
        if memories is not None:memories.append(np.nan)
    return 

def print_benchmark_results(benchmark_results, sizes, show_memory_usage=True):
    # Print the results
    print("Matrix Size | COO Pure Time | Numpy Time | Scipy Time")
    print("------------|---------------|------------|-----------")
    for size, time_coo, time_numpy, time_scipy in zip(sizes, benchmark_results['coo']['times'], benchmark_results['numpy']['times'], benchmark_results['scipy']['times']):
        print(f"{size:^12}|{time_coo:^15.4f}|{time_numpy:^12.4f}|{time_scipy:^11.4f}")
    if show_memory_usage:
        # Print memory usage results
        print("\nMemory Usage")
        print("------------|---------------|------------|-----------")
        for size, mem_coo, mem_numpy, mem_scipy in zip(sizes, benchmark_results['coo']['memories'], benchmark_results['numpy']['memories'], benchmark_results['scipy']['memories']):
            print(f"{size:^12}|{format_memory(mem_coo):^15}|{format_memory(mem_numpy):^12}|{format_memory(mem_scipy):^11}")

def plot_CPU_time_benchmark(sizes, 
                            benchmark_results, 
                            title='Time Comparison',
                            xlabel="Matrix Size", 
                            ylabel='Time (s)',
                            xlog=False,
                            output=None):
    plt.figure(figsize=(6, 4),dpi=100)
    line_styles = {"numpy":'+-k', "coo":'o:k',  "scipy":'*--k'}

    for _, (key, value) in enumerate(benchmark_results.items()):
        # Use a different line style for each data series
        if xlog: #handling matrix size manually to increase graph readability
            xval=np.array(sizes)/1000
        else:
            xval=sizes 
        plt.plot(xval, value['times'], 
                 line_styles[key], label=key, linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')  # Set the y-axis to a logarithmic scale
    if xlog:
        plt.xscale("log")
        # Set the major formatter for the x-axis using plt
        from matplotlib.ticker import FuncFormatter
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda value, tick_number: f"{int(value)}k"))
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add a grid
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if output:
        plt.savefig(output)
    plt.show()