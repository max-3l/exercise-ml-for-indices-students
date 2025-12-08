# Learned Index Structure for Point Queries Exercise

This exercise provides a hands-on introduction to the core concepts behind learned index structures, designed for a lecture on Spatial Databases. The goal is to build a simple "learned" index for spatially-ordered data using a Z-order curve.

The exercise uses a small, real-world dataset of Points of Interest (POIs), which is included in the `data/` directory.

## Core Concepts

- **Real-World Data:** We use a sample of real POI data with latitude and longitude coordinates. Real-world coordinates (floats) are converted into integers to be used by the Z-order curve. The `data_loader.py` module handles this by normalizing the coordinates to a unit square and then quantizing them into an integer grid. A z-curve is applied to map coordinates into a 1D array. The dataset is resampled to match a target dataset size.
- **Learned Indexing:** Using a model to predict the position of a key in the sorted Z-order dataset.
- **Benchmarking:** Comparing the performance of the learned indices against standard baseline algorithms.

## Project Structure

The project is organized as a Python package for clarity:

```
.
├── data/
│   └── poi_data.csv        # Sample POI dataset
├── learned_index_exercise/
│   ├── __init__.py           # Makes the directory a Python package
│   ├── data_loader.py        # Handles loading and processing of the data
│   ├── learned_index.py      # <-- YOUR CODE GOES HERE
│   ├── searches.py           # Contains baseline search algorithms for comparison
│   └── utils.py              # Contains the Z-order curve implementation
├── dashboard.py              # Interactive Streamlit dashboard for experimentation
├── run_exercise.py           # The main script to execute the benchmark
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Your Task

Your assignment is to complete the implementation of the `LearnedIndex` abstract base class located in the `learned_index_exercise/learned_index.py` file.

You will need to fill in the logic for three methods marked with `TODO` comments:

1. ` predict(z_value: int) -> int`  
Implement the prediction of the array position given a key in the classes `PyTorchModel` and `DecisionTreeModel`.
Take a look at the implementations and API of the used libraries (PyTorch for MLPs and Scikit-Learn for decision tree).

Hint: The training code might help you.

2. `error_bound(sorted_z_values: list[int]) -> float`  
Compute the maximum absolute prediction error across all training data points. This error bound will be used to limit the search range during queries.

3. `search(sorted_z_values: list[int], query: int) -> tuple[int, int]`  
Perform a search using the learned model's prediction. The method should:
- Predict the position of the query value
- Ensure the predicted position is within valid bounds
- Search around the predicted position using the error bounds or perform exponential search starting from the predicted position
- Count and return the number of comparisons made
- Return a tuple of the final position (or -1 if not found), and the total number of comparisons

## Setup

It is recommended to set up a virtual environment to manage dependencies.

1.  **Create a virtual environment (if you don't have one):**
    ```bash
    python -m venv venv
    ```
2.  **Activate the virtual environment:**
    -   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    -   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Exercise

1.  Navigate to the root directory of the project in your terminal.
2.  Run the main script:
    ```bash
    python run_exercise.py
    ```

The script will automatically use your implementation in `learned_index.py`, run the benchmarks, and print a report comparing the performance of your learned index against the baselines. Your goal is to successfully find the element with fewer comparisons than a full scan and, ideally, to be competitive with binary search.

### Experiment with the Interactive Dashboard

After completing your implementation, you can explore the behavior of learned indices interactively using the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The dashboard allows you to:

#### Spatial POI Data Mode
- **Configure dataset size and resolution**: Experiment with different dataset sizes (10K to 2M points).
- **Compare multiple models**: Run benchmarks with different combinations of baseline methods (Full Scan, Binary Search, Exponential Search) and learned index models (PyTorch Linear, PyTorch MLP, Decision Tree)
- **Tune hyperparameters**: Adjust training epochs, learning rates, batch sizes for PyTorch models, or tree depth for Decision Trees
- **Visualize results**: See performance metrics including average query time, number of comparisons, success rate, model parameters, memory consumption, and error bounds

#### Synthetic 1D Data Mode
- **Draw custom distributions**: Use your mouse to draw custom data distributions on a canvas to see how learned indices perform on different data patterns

**Key Insights to Explore:**
- How does the error bound relate to the number of comparisons needed?
- How do different data distributions affect model accuracy?
- What's the trade-off between model complexity (parameters) and search performance?
- How does training time scale with dataset size and model complexity?
- Why are traditional search methods better/worse than the learned index in this simple example?

Try experimenting with:
- Uniform distributions vs. skewed distributions
- Different model architectures (Linear vs. MLP vs. Decision Tree)
- Various hyperparameter settings
- Small vs. large datasets

If you feel like it you can also change the PyTorch architectures in the code.

### Bonus: Implement Recursive-Model Index (RM Index)

If you have extra time, modify one of the MLP models to use multiple models in a tree structure.
You could also implement a new class, however in this case you need to adapt the dashboard and `run_exercise.py` script to find and use your new class.

Take a look at the paper [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208) for further information on how to structure the ensemble model.
