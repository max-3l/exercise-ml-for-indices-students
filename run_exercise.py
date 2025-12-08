import random
import time
import argparse
import numpy as np
from tabulate import tabulate

from learned_index_exercise.utils import coords_to_z_order
from learned_index_exercise.searches import search_full_scan, search_binary, search_exponential
from learned_index_exercise.learned_index import PyTorchLinearModel, PyTorchMLPModel, DecisionTreeModel
from learned_index_exercise.data_loader import load_and_process_data

def run_exercise(dataset_size: int):
    """
    Loads real data, runs the searches, and prints the benchmark results for multiple models,
    averaging results over multiple search queries.
    """
    print("--- Learned Index Exercise with Real POI Data ---")

    # Configuration
    coord_max = 4294967295   # Max value for quantized coordinates (fits in 32 bits: 2^32 - 1)
    num_runs = 10       # Number of random search queries to average over
    target_data_size = dataset_size # Target size for the extended dataset

    # 1. Load and process data from the CSV file
    print(f"\n1. Loading and processing POI data...")
    quantized_points = load_and_process_data(coord_max, target_size=target_data_size)
    
    if not quantized_points:
        print("Could not load data points. Exiting.")
        return

    # 2. Convert to 1D Z-order values and sort them
    print("2. Converting to Z-order values and sorting...")
    z_values = sorted([coords_to_z_order(x, y) for x, y in quantized_points])

    # 3. Define and train models to test (training happens once)
    models = {
        "PyTorch Linear": PyTorchLinearModel(show_progress=True),
        "PyTorch MLP": PyTorchMLPModel(show_progress=True),
        "Decision Tree": DecisionTreeModel(),
    }

    print("\n3. Training Learned Index Models...")
    for name, model in models.items():
        print(f"--- Training {name} ---")
        start_time_train = time.perf_counter()
        model.train(z_values)
        end_time_train = time.perf_counter()
        train_time = (end_time_train - start_time_train) * 1e3 # in milliseconds
        print(f"Training time for {name}: {train_time:.2f} ms")


    # Store results for averaging
    all_results = {
        "Full Scan": {'times': [], 'comps': [], 'successes': []},
        "Binary Search": {'times': [], 'comps': [], 'successes': []},
        "Exponential Search": {'times': [], 'comps': [], 'successes': []}
    }
    for name in models.keys():
        all_results[name] = {'times': [], 'comps': [], 'successes': []}

    print(f"\n--- Running Benchmarks over {num_runs} random queries ---")

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}...")
        # 4. Pick a random value to search for
        search_query = random.choice(z_values)

        # Baseline: Full Scan
        start_time = time.perf_counter()
        found_idx_fs, comps_fs = search_full_scan(z_values, search_query)
        end_time = time.perf_counter()
        all_results["Full Scan"]['times'].append((end_time - start_time) * 1e6)
        all_results["Full Scan"]['comps'].append(comps_fs)
        all_results["Full Scan"]['successes'].append(True)

        # Baseline: Binary Search
        start_time = time.perf_counter()
        found_idx_bs, comps_bs = search_binary(z_values, search_query)
        end_time = time.perf_counter()
        all_results["Binary Search"]['times'].append((end_time - start_time) * 1e6)
        all_results["Binary Search"]['comps'].append(comps_bs)
        all_results["Binary Search"]['successes'].append(True)

        # Baseline: Exponential Search
        start_time = time.perf_counter()
        found_idx_es, comps_es = search_exponential(z_values, search_query)
        end_time = time.perf_counter()
        all_results["Exponential Search"]['times'].append((end_time - start_time) * 1e6)
        all_results["Exponential Search"]['comps'].append(comps_es)
        all_results["Exponential Search"]['successes'].append(True)

        if found_idx_bs == -1:
            print(f"Warning: Binary search failed to find query {search_query}. Skipping this run.")
            continue
        
        # Benchmark learned models
        for name, model in models.items():
            start_time_search = time.perf_counter()
            found_idx_li, comps_li = model.search(z_values, search_query)
            end_time_search = time.perf_counter()
            
            success = (found_idx_li == found_idx_bs)
            all_results[name]['times'].append((end_time_search - start_time_search) * 1e6)
            all_results[name]['comps'].append(comps_li)
            all_results[name]['successes'].append(success)

    # 6. Calculate averages and print results using tabulate
    print("\n\n--- Final Averaged Benchmark Results ---")
        
    headers = ["Search Method", "Avg Time (µs)", "Avg Comparisons", "Success Rate"]
    table_data = []

    for name, data in all_results.items():
        avg_time = np.mean(data['times']) if data['times'] else 0
        avg_comps = np.mean(data['comps']) if data['comps'] else 0
        success_rate = np.mean(data['successes']) * 100 if data['successes'] else 0

        method_name_display = name
        if name in models.keys() and success_rate < 100:
            method_name_display = f"{name} ({success_rate:.1f}% suc.)"
        
        table_data.append([method_name_display, f"{avg_time:.2f}", f"{avg_comps:.2f}", f"{success_rate:.1f}%"])

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print("\nAnalysis:")
    print(f"Averaged over {num_runs} random search queries on a dataset of {len(z_values)} points.")
    print(f"Baseline Binary Search averages {np.mean(all_results['Binary Search']['comps']):.2f} comparisons.")
    print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the learned index exercise with a configurable dataset size.")
    parser.add_argument("--dataset-size", type=int, default=0,
                        help="Target size of the dataset. If 0 or less, uses the original dataset size. "
                             "Otherwise, extends the dataset to the specified size using KDE.")
    args = parser.parse_args()
    
    run_exercise(args.dataset_size)
