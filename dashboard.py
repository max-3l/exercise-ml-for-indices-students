import streamlit as st
import random
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
from learned_index_exercise.utils import coords_to_z_order
from learned_index_exercise.searches import search_full_scan, search_binary, search_exponential
from learned_index_exercise.learned_index import PyTorchLinearModel, PyTorchMLPModel, DecisionTreeModel
from learned_index_exercise.data_loader import load_and_process_data

st.set_page_config(page_title="Learned Index Benchmark Dashboard", layout="wide")

st.title("Learned Index Benchmark Dashboard")
st.markdown("Configure and benchmark different search methods on POI data with Z-order curve indexing.")

st.sidebar.header("Configuration")

# Dataset Type Selection
st.sidebar.subheader("Dataset Type")
dataset_type = st.sidebar.radio(
    "Select Dataset",
    options=["Spatial POI Data", "Synthetic 1D Data"],
    help="Choose between real spatial data or custom synthetic distribution"
)

st.sidebar.subheader("Dataset")
dataset_size = st.sidebar.number_input(
    "Dataset Size", 
    min_value=10000, 
    max_value=2000000, 
    value=500000, 
    step=50000,
    help="Target number of data points (will be extended using KDE if needed)"
)

# Synthetic dataset configuration (only shown for synthetic data)
if dataset_type == "Synthetic 1D Data":
    st.sidebar.subheader("Synthetic Data Configuration")
    num_buckets = st.sidebar.slider(
        "Number of Buckets",
        min_value=5,
        max_value=256,
        value=20,
        help="Number of histogram buckets for distribution"
    )
    
    coord_max_value = 2**32 - 1  # Always use 32-bit for synthetic data
    coord_max = 32  # For display purposes
else:
    num_buckets = 20  # Default value for spatial data (not used)
    coord_max = st.sidebar.selectbox(
        "Coordinate Resolution (bits)",
        options=[10, 16, 20, 24, 32],
        index=4,
        format_func=lambda x: f"{x} bits (max: {2**x - 1:,})",
        help="Number of bits for quantized coordinates"
    )
    coord_max_value = 2**coord_max - 1

num_runs = st.sidebar.number_input(
    "Number of Benchmark Runs",
    min_value=1,
    max_value=100,
    value=10,
    help="Number of random queries to average results over"
)


st.sidebar.subheader("Models to Benchmark")
include_baselines = st.sidebar.checkbox("Include Baseline Methods", value=True)
if include_baselines:
    run_full_scan = st.sidebar.checkbox("Full Scan", value=False, help="Warning: Very slow for large datasets")
    run_binary = st.sidebar.checkbox("Binary Search", value=True)
    run_exponential = st.sidebar.checkbox("Exponential Search", value=True)
else:
    run_full_scan = run_binary = run_exponential = False

st.sidebar.subheader("Learned Index Models")
run_pytorch_linear = st.sidebar.checkbox("PyTorch Linear", value=True)
run_pytorch_mlp = st.sidebar.checkbox("PyTorch MLP", value=True)
run_decision_tree = st.sidebar.checkbox("Decision Tree", value=True)


st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameters")

# PyTorch Linear hyperparameters
if run_pytorch_linear:
    with st.sidebar.expander("PyTorch Linear Settings"):
        pt_linear_epochs = st.number_input("Epochs", min_value=1, max_value=200, value=10, key="pt_linear_epochs")
        pt_linear_lr = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001, format="%.4f", key="pt_linear_lr")
        pt_linear_batch = st.number_input("Batch Size", min_value=64, max_value=dataset_size, value=204800, step=1024, key="pt_linear_batch")

if run_pytorch_mlp:
    with st.sidebar.expander("PyTorch MLP Settings"):
        pt_mlp_epochs = st.number_input("Epochs", min_value=1, max_value=200, value=10, key="pt_mlp_epochs")
        pt_mlp_lr = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001, format="%.4f", key="pt_mlp_lr")
        pt_mlp_batch = st.number_input("Batch Size", min_value=64, max_value=dataset_size, value=204800, step=1024, key="pt_mlp_batch")
        pt_mlp_hidden = st.number_input("Hidden Layer Size", min_value=8, max_value=256, value=32, step=8, key="pt_mlp_hidden")
        pt_mlp_layers = st.number_input("Number of Hidden Layers", min_value=1, max_value=5, value=2, key="pt_mlp_layers")

if run_decision_tree:
    with st.sidebar.expander("Decision Tree Settings"):
        dt_max_depth = st.number_input("Max Depth", min_value=1, max_value=100, value=20, key="dt_max_depth")
        dt_min_samples_split = st.number_input("Min Samples Split", min_value=2, max_value=100, value=2, key="dt_min_samples_split")

# Main content area
# Synthetic data drawing interface (shown only for synthetic data)
if dataset_type == "Synthetic 1D Data":
    st.markdown("---")
    st.subheader("Draw Distribution Histogram")
    st.markdown("**Instructions:** Draw with your mouse on the canvas below. The canvas will be divided into buckets, and each bucket's height is the average of drawn points in that region.")
    
    # Initialize drawing data in session state
    if 'bucket_heights' not in st.session_state:
        st.session_state.bucket_heights = [50] * num_buckets
    
    col_canvas, col_preview = st.columns([3, 2])
    
    with col_canvas:
        # Create drawable canvas with smaller width to fit in column
        canvas_height = 400
        canvas_width = 500  # Reduced to fit better in smaller viewports
        padding_percent = 0.05  # 5% padding on left and right
        padding_pixels = int(canvas_width * padding_percent)
        
        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
            stroke_width=3,
            stroke_color="#E74C3C",  # Red stroke
            background_color="#F0F2F6",
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=True,
        )
        
        # Process canvas drawing into bucket heights
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            
            if len(objects) > 0:
                # Extract all points from all drawn paths
                all_points = []
                
                for obj in objects:
                    if obj["type"] == "path":
                        path = obj["path"]
                        for segment in path:
                            if len(segment) >= 3:  # Has x, y coordinates
                                x = segment[1]
                                y = segment[2]
                                
                                # Apply padding: only consider points within the active drawing area
                                if padding_pixels <= x <= (canvas_width - padding_pixels):
                                    # Normalize coordinates relative to the active area (excluding padding)
                                    active_width = canvas_width - (2 * padding_pixels)
                                    x_norm = (x - padding_pixels) / active_width
                                    y_norm = 1 - (y / canvas_height)  # Invert y (canvas has origin at top)
                                    
                                    if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
                                        all_points.append((x_norm, y_norm * 100))
                
                # Calculate bucket heights from drawn points
                if len(all_points) > 0:
                    bucket_heights = [0] * num_buckets
                    bucket_counts = [0] * num_buckets
                    
                    for x, y in all_points:
                        bucket_idx = int(x * num_buckets)
                        if 0 <= bucket_idx < num_buckets:
                            bucket_heights[bucket_idx] += y
                            bucket_counts[bucket_idx] += 1
                    
                    # Calculate average height for each bucket
                    for i in range(num_buckets):
                        if bucket_counts[i] > 0:
                            bucket_heights[i] = bucket_heights[i] / bucket_counts[i]
                    
                    st.session_state.bucket_heights = bucket_heights
        
        # Show bucket overlay
        st.caption(f"Canvas divided into {num_buckets} buckets with {padding_pixels}px padding on each side - draw in the center area to define your distribution")
    
    with col_preview:
        st.markdown("**Distribution Preview:**")
        # Create preview with actual distribution
        bucket_heights = st.session_state.bucket_heights
        
        if sum(bucket_heights) > 0:
            total = sum(bucket_heights)
            percentages = [(h / total) * 100 for h in bucket_heights]
            
            # Show as bar chart
            preview_fig = go.Figure()
            preview_fig.add_trace(go.Bar(
                x=[f'B{i+1}' for i in range(num_buckets)],
                y=percentages,
                marker=dict(color='lightgreen'),
                text=[f'{p:.1f}%' for p in percentages],
                textposition='outside'
            ))
            
            preview_fig.update_layout(
                title="Normalized Distribution",
                xaxis_title="Bucket",
                yaxis_title="Percentage (%)",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(preview_fig, use_container_width=True, key="preview_chart")
            
            # Show expected distribution stats
            st.markdown("**Statistics:**")
            st.caption(f"**Total points:** {dataset_size:,}")
            st.caption(f"**Buckets:** {num_buckets}")
            st.caption(f"**Value range:** 0 to {2**32-1:,}")
            st.caption(f"**Bucket width:** ~{(2**32) // num_buckets:,}")
            
            # Show points per bucket estimate
            with st.expander("Expected points per bucket"):
                for i in range(num_buckets):
                    expected_points = int(dataset_size * (bucket_heights[i] / total))
                    st.caption(f"B{i+1}: ~{expected_points:,} points ({percentages[i]:.1f}%)")
        else:
            st.warning("Draw on the canvas!")
            st.caption("Use your mouse to draw a distribution on the canvas above.")
    
    st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    run_benchmark = st.button("Run Benchmark", type="primary", use_container_width=True)
    
with col2:
    if dataset_type == "Synthetic 1D Data":
        st.info(f"Configuration: {dataset_size:,} points (Synthetic 1D), 32-bit, {num_runs} runs")
    else:
        st.info(f"Configuration: {dataset_size:,} points (Spatial POI), {coord_max}-bit coords, {num_runs} runs")

if 'results' not in st.session_state:
    st.session_state.results = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def create_mlp_model(hidden_size, num_layers):
    """Create MLP model with configurable architecture"""
    import torch.nn as nn
    layers = []
    layers.append(nn.Linear(1, hidden_size))
    layers.append(nn.ReLU())
    
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(hidden_size, 1))
    return nn.Sequential(*layers)

def generate_synthetic_data(bucket_heights, num_points, max_value=2**32 - 1):
    """
    Generate synthetic 1D data based on histogram bucket heights.
    Ensures no duplicates and uses 32-bit values.
    Optimized using NumPy for fast generation.
    Values within each bucket are uniformly distributed to match the gradient.
    
    Args:
        bucket_heights: List of relative heights for each bucket
        num_points: Total number of points to generate
        max_value: Maximum value (default: 2^32 - 1)
    
    Returns:
        Sorted list of unique integer values
    """
    if sum(bucket_heights) == 0:
        # If no distribution drawn, use uniform
        bucket_heights = [1] * len(bucket_heights)
    
    # Normalize heights to probabilities
    total = sum(bucket_heights)
    probabilities = np.array(bucket_heights) / total
    
    # Calculate bucket boundaries
    num_buckets = len(bucket_heights)
    bucket_width = max_value // num_buckets
    
    # Determine points per bucket using vectorized operations
    points_per_bucket = np.round(num_points * probabilities).astype(np.int64)
    
    # Adjust for rounding errors
    diff = num_points - points_per_bucket.sum()
    if diff != 0:
        # Add/subtract difference to buckets with highest probabilities
        adjustment_indices = np.argsort(probabilities)[::-1][:abs(diff)]
        points_per_bucket[adjustment_indices] += np.sign(diff)
    
    # Generate unique values - optimized for when buckets << num_points
    all_values = []
    
    for bucket_idx, count in enumerate(points_per_bucket):
        if count <= 0:
            continue
            
        bucket_start = bucket_idx * bucket_width
        bucket_end = min((bucket_idx + 1) * bucket_width - 1, max_value)
        bucket_range = bucket_end - bucket_start + 1
        
        if count > bucket_range:
            # More points requested than available in bucket - use all values
            all_values.extend(np.arange(bucket_start, bucket_end + 1, dtype=np.int64))
        elif bucket_range <= 10000:
            # Small bucket range - generate all and sample
            bucket_values = np.arange(bucket_start, bucket_end + 1, dtype=np.int64)
            selected = np.random.choice(bucket_values, size=count, replace=False)
            all_values.extend(selected)
        else:
            # Large bucket range - use linspace for gradient distribution + jitter
            # This ensures uniform distribution within the bucket (matches histogram gradient)
            base_positions = np.linspace(bucket_start, bucket_end, count, dtype=np.float64)
            
            # Add jitter to make values random but maintain distribution
            # Jitter range is smaller than spacing to prevent overlap
            spacing = (bucket_end - bucket_start) / count if count > 1 else 1
            jitter_range = min(spacing * 0.4, bucket_range * 0.1)  # 40% of spacing or 10% of range
            
            jittered = base_positions + np.random.uniform(-jitter_range, jitter_range, size=count)
            
            # Clip to bucket boundaries and convert to integers
            jittered = np.clip(jittered, bucket_start, bucket_end).astype(np.int64)
            
            # Handle duplicates from jittering by replacing with random values
            unique_jittered = np.unique(jittered)
            
            if len(unique_jittered) < count:
                # Need to generate additional unique values
                needed = count - len(unique_jittered)
                
                # Create a set of all possible values in bucket
                all_bucket_vals = set(range(bucket_start, bucket_end + 1))
                # Remove already used values
                available = all_bucket_vals - set(unique_jittered)
                
                # Sample from available
                if len(available) >= needed:
                    additional = np.random.choice(list(available), size=needed, replace=False)
                    unique_jittered = np.concatenate([unique_jittered, additional])
                else:
                    # Not enough unique values - use all available
                    unique_jittered = np.concatenate([unique_jittered, np.array(list(available), dtype=np.int64)])
            
            all_values.extend(unique_jittered[:count])
    
    # Convert to array and ensure uniqueness across all buckets
    all_values = np.array(all_values, dtype=np.int64)
    unique_values = np.unique(all_values)
    
    # If we don't have enough unique values due to overlaps between buckets, generate more
    if len(unique_values) < num_points:
        remaining = num_points - len(unique_values)
        
        # Find gaps in the value space and fill them
        used_set = set(unique_values)
        
        # Try to sample from unused values in a smart way
        sample_size = min(remaining * 100, max_value // 100)  # Sample from a reasonable range
        candidates = np.random.randint(0, max_value + 1, size=sample_size, dtype=np.int64)
        candidates = candidates[~np.isin(candidates, list(used_set))]
        
        if len(candidates) >= remaining:
            extra_values = candidates[:remaining]
        else:
            # Fallback: sequential search for unused values
            extra_values = []
            for val in range(max_value + 1):
                if val not in used_set:
                    extra_values.append(val)
                    if len(extra_values) >= remaining:
                        break
            extra_values = np.array(extra_values[:remaining], dtype=np.int64)
        
        unique_values = np.concatenate([unique_values, extra_values])
    
    # Sort and return exactly num_points values
    result = np.sort(unique_values[:num_points])
    return result.tolist()

def count_model_parameters(model):
    """Count the number of parameters in a model"""
    if hasattr(model, 'model'):
        # PyTorch models
        if hasattr(model.model, 'parameters'):
            return sum(p.numel() for p in model.model.parameters())
        # Scikit-learn Decision Tree
        elif hasattr(model.model, 'tree_'):
            # Count tree nodes and features
            # Get memory size of tree structure
            tree = model.model.tree_
            return tree.node_count * (tree.n_features + 1)  # Rough estimate
    return 0

def run_benchmark_pipeline():
    """Run the complete benchmark pipeline"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Load or generate data
    status_text.text("Loading and processing data...")
    progress_bar.progress(10)
    
    if dataset_type == "Synthetic 1D Data":
        # Generate synthetic 1D data
        bucket_heights = st.session_state.get('bucket_heights', [50] * num_buckets)
        status_text.text("Generating synthetic data from distribution...")
        z_values = generate_synthetic_data(bucket_heights, dataset_size, max_value=2**32 - 1)
        data_size = len(z_values)
        
        # Show histogram of generated data
        st.markdown("---")
        st.subheader("Generated Data Distribution")
        
        # Create histogram with same bucket count as target for accurate comparison
        target_heights = st.session_state.get('bucket_heights', [50] * num_buckets)
        
        # Use the same buckets as the target distribution (based on full 32-bit range)
        bucket_width = (2**32) / num_buckets
        bucket_edges = np.array([i * bucket_width for i in range(num_buckets + 1)])
        
        # Count how many generated values fall into each bucket
        hist_counts, _ = np.histogram(z_values, bins=bucket_edges)
        
        # Calculate bucket centers
        bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2
        
        # Create histogram visualization with two representations
        hist_fig = go.Figure()
        
        # 1. Bar chart representation
        hist_fig.add_trace(go.Bar(
            x=list(range(1, num_buckets + 1)),
            y=hist_counts,
            marker=dict(color='steelblue', line=dict(width=1, color='darkblue')),
            name='Generated Data (Bars)',
            hovertemplate='Bucket %{x}<br>Count: %{y}<extra></extra>',
            opacity=0.7
        ))
        
        # 2. Curve representation (smooth line)
        hist_fig.add_trace(go.Scatter(
            x=list(range(1, num_buckets + 1)),
            y=hist_counts,
            mode='lines+markers',
            line=dict(color='darkblue', width=3),
            marker=dict(size=6, color='darkblue'),
            name='Generated Data (Curve)',
            hovertemplate='Bucket %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Overlay the target distribution for comparison
        if sum(target_heights) > 0:
            total_target = sum(target_heights)
            probabilities = np.array(target_heights) / total_target
            expected_counts = probabilities * data_size
            
            hist_fig.add_trace(go.Scatter(
                x=list(range(1, num_buckets + 1)),
                y=expected_counts,
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                name='Target Distribution',
                hovertemplate='Bucket %{x}<br>Expected: %{y:.1f}<extra></extra>'
            ))
        
        hist_fig.update_layout(
            title=f"Generated Data Distribution ({num_buckets} buckets matching target)",
            xaxis_title="Bucket Number",
            yaxis_title="Count",
            height=500,
            showlegend=True,
            hovermode='x unified',
            xaxis=dict(tickmode='linear', dtick=max(1, num_buckets // 20))
        )
        
        st.plotly_chart(hist_fig, use_container_width=True)
        
        # Show statistics
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        with col_stats1:
            st.metric("Total Points", f"{len(z_values):,}")
        with col_stats2:
            st.metric("Min Value", f"{min(z_values):,}")
        with col_stats3:
            st.metric("Max Value", f"{max(z_values):,}")
        with col_stats4:
            st.metric("Unique Values", f"{len(set(z_values)):,}")
        
        st.markdown("---")
        
    else:
        # Load spatial POI data
        quantized_points = load_and_process_data(coord_max_value, target_size=dataset_size)
        
        if not quantized_points:
            st.error("Could not load data points.")
            return None
        
        # Step 2: Convert to Z-order
        status_text.text("Converting to Z-order values...")
        progress_bar.progress(20)
        z_values = sorted([coords_to_z_order(x, y) for x, y in quantized_points])
        data_size = len(z_values)
        
        # Show histogram of spatial POI z-order distribution
        st.markdown("---")
        st.subheader("Spatial POI Data Z-Order Distribution")
        
        # Create histogram with automatic binning
        num_bins = 50
        hist_counts, bin_edges = np.histogram(z_values, bins=num_bins)
        
        # Create histogram visualization
        spatial_hist_fig = go.Figure()
        
        # Bar chart representation
        spatial_hist_fig.add_trace(go.Bar(
            x=list(range(1, num_bins + 1)),
            y=hist_counts,
            marker=dict(color='seagreen', line=dict(width=1, color='darkgreen')),
            name='Z-Order Values',
            hovertemplate='Bin %{x}<br>Count: %{y}<extra></extra>',
            opacity=0.7
        ))
        
        # Curve representation
        spatial_hist_fig.add_trace(go.Scatter(
            x=list(range(1, num_bins + 1)),
            y=hist_counts,
            mode='lines+markers',
            line=dict(color='darkgreen', width=3),
            marker=dict(size=6, color='darkgreen'),
            name='Distribution Curve',
            hovertemplate='Bin %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        spatial_hist_fig.update_layout(
            title=f"Z-Order Value Distribution ({num_bins} bins)",
            xaxis_title="Bin Number",
            yaxis_title="Count",
            height=500,
            showlegend=True,
            hovermode='x unified',
            xaxis=dict(tickmode='linear', dtick=max(1, num_bins // 20))
        )
        
        st.plotly_chart(spatial_hist_fig, use_container_width=True)
        
        # Show statistics
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        with col_stats1:
            st.metric("Total Points", f"{len(z_values):,}")
        with col_stats2:
            st.metric("Min Z-Value", f"{min(z_values):,}")
        with col_stats3:
            st.metric("Max Z-Value", f"{max(z_values):,}")
        with col_stats4:
            st.metric("Unique Values", f"{len(set(z_values)):,}")
        
        st.markdown("---")
    
    progress_bar.progress(20)
    
    # Step 3: Initialize and train models
    models = {}
    training_times = {}
    all_loss_data = {}  # Store loss curves for all PyTorch models
    
    progress_step = 40 / sum([run_pytorch_linear, run_pytorch_mlp, run_decision_tree])
    current_progress = 20
    
    if run_pytorch_linear:
        status_text.text("Training PyTorch Linear model...")
        model = PyTorchLinearModel(epochs=pt_linear_epochs, lr=pt_linear_lr, batch_size=pt_linear_batch)
        
        # Create live loss tracking
        loss_data = {'epoch': [], 'loss': []}
        epoch_progress = st.empty()
        loss_chart = st.empty()
        
        def progress_callback(epoch, total_epochs, loss):
            loss_data['epoch'].append(epoch)
            loss_data['loss'].append(loss)
            epoch_progress.text(f"PyTorch Linear - Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}")
            
            # Update loss curve in real-time
            if len(loss_data['epoch']) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=loss_data['epoch'],
                    y=loss_data['loss'],
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='steelblue', width=2),
                    marker=dict(size=4)
                ))
                fig.update_layout(
                    title="PyTorch Linear - Training Loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=300,
                    margin=dict(l=50, r=20, t=40, b=40)
                )
                loss_chart.plotly_chart(fig, use_container_width=True, key=f"linear_loss_{epoch}")
        
        start_time = time.perf_counter()
        model.train(z_values, progress_callback=progress_callback)
        training_times["PyTorch Linear"] = (time.perf_counter() - start_time) * 1e3
        models["PyTorch Linear"] = model
        all_loss_data["PyTorch Linear"] = loss_data.copy()
        epoch_progress.empty()
        loss_chart.empty()
        current_progress += progress_step
        progress_bar.progress(int(current_progress))
    
    if run_pytorch_mlp:
        status_text.text("Training PyTorch MLP model...")
        mlp_model = create_mlp_model(pt_mlp_hidden, pt_mlp_layers)
        model = PyTorchMLPModel(epochs=pt_mlp_epochs, lr=pt_mlp_lr, batch_size=pt_mlp_batch)
        model.model = mlp_model
        
        # Create live loss tracking
        loss_data = {'epoch': [], 'loss': []}
        epoch_progress = st.empty()
        loss_chart = st.empty()
        
        def progress_callback(epoch, total_epochs, loss):
            loss_data['epoch'].append(epoch)
            loss_data['loss'].append(loss)
            epoch_progress.text(f"PyTorch MLP - Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}")
            
            # Update loss curve in real-time
            if len(loss_data['epoch']) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=loss_data['epoch'],
                    y=loss_data['loss'],
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='darkorange', width=2),
                    marker=dict(size=4)
                ))
                fig.update_layout(
                    title="PyTorch MLP - Training Loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=300,
                    margin=dict(l=50, r=20, t=40, b=40)
                )
                loss_chart.plotly_chart(fig, use_container_width=True, key=f"mlp_loss_{epoch}")
        
        start_time = time.perf_counter()
        model.train(z_values, progress_callback=progress_callback)
        training_times["PyTorch MLP"] = (time.perf_counter() - start_time) * 1e3
        models["PyTorch MLP"] = model
        all_loss_data["PyTorch MLP"] = loss_data.copy()
        epoch_progress.empty()
        loss_chart.empty()
        current_progress += progress_step
        progress_bar.progress(int(current_progress))
    
    if run_decision_tree:
        status_text.text("Training Decision Tree model...")
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeModel()
        model.model = DecisionTreeRegressor(random_state=1, max_depth=dt_max_depth, min_samples_split=dt_min_samples_split)
        start_time = time.perf_counter()
        model.train(z_values)
        training_times["Decision Tree"] = (time.perf_counter() - start_time) * 1e3
        models["Decision Tree"] = model
        current_progress += progress_step
        progress_bar.progress(int(current_progress))
    

    
    # Step 4: Run benchmarks
    all_results = {}
    
    if run_full_scan:
        all_results["Full Scan"] = {'times': [], 'comps': [], 'successes': []}
    if run_binary:
        all_results["Binary Search"] = {'times': [], 'comps': [], 'successes': []}
    if run_exponential:
        all_results["Exponential Search"] = {'times': [], 'comps': [], 'successes': []}
    
    for name in models.keys():
        all_results[name] = {'times': [], 'comps': [], 'successes': []}
    
    status_text.text(f"Running {num_runs} benchmark queries...")
    
    for run in range(num_runs):
        progress_bar.progress(int(60 + (40 * run / num_runs)))
        
        search_query = random.choice(z_values)
        
        # Baselines
        if run_full_scan:
            start_time = time.perf_counter()
            found_idx_fs, comps_fs = search_full_scan(z_values, search_query)
            end_time = time.perf_counter()
            all_results["Full Scan"]['times'].append((end_time - start_time) * 1e6)
            all_results["Full Scan"]['comps'].append(comps_fs)
            all_results["Full Scan"]['successes'].append(True)
        
        if run_binary:
            start_time = time.perf_counter()
            found_idx_bs, comps_bs = search_binary(z_values, search_query)
            end_time = time.perf_counter()
            all_results["Binary Search"]['times'].append((end_time - start_time) * 1e6)
            all_results["Binary Search"]['comps'].append(comps_bs)
            all_results["Binary Search"]['successes'].append(True)
        else:
            # Use binary search as reference even if not shown
            found_idx_bs, _ = search_binary(z_values, search_query)
        
        if run_exponential:
            start_time = time.perf_counter()
            found_idx_es, comps_es = search_exponential(z_values, search_query)
            end_time = time.perf_counter()
            all_results["Exponential Search"]['times'].append((end_time - start_time) * 1e6)
            all_results["Exponential Search"]['comps'].append(comps_es)
            all_results["Exponential Search"]['successes'].append(True)
        
        # Learned models
        for name, model in models.items():
            start_time_search = time.perf_counter()
            found_idx_li, comps_li = model.search(z_values, search_query)
            end_time_search = time.perf_counter()
            
            success = (found_idx_li == found_idx_bs)
            all_results[name]['times'].append((end_time_search - start_time_search) * 1e6)
            all_results[name]['comps'].append(comps_li)
            all_results[name]['successes'].append(success)
    
    progress_bar.progress(100)
    status_text.text("Benchmark complete!")
    
    # Compile results
    results_data = []
    for name, data in all_results.items():
        avg_time = np.mean(data['times']) if data['times'] else 0
        avg_comps = np.mean(data['comps']) if data['comps'] else 0
        success_rate = np.mean(data['successes']) * 100 if data['successes'] else 0
        train_time = training_times.get(name, 0)
        
        # Get parameter count and error bounds for learned models
        model = models.get(name)
        if model:
            num_params = count_model_parameters(model)
            # Estimate memory in KB (assuming 4 bytes per float parameter)
            memory_kb = (num_params * 4) / 1024
            params_str = f"{num_params:,}"
            memory_str = f"{memory_kb:.2f} KB" if memory_kb < 1024 else f"{memory_kb/1024:.2f} MB"
            # Get error bounds
            error_bound = getattr(model, 'max_error', None)
            error_str = f"{error_bound:,}" if error_bound is not None else "N/A"
        else:
            params_str = "N/A"
            memory_str = "N/A"
            error_str = "N/A"
        
        results_data.append({
            'Method': name,
            'Avg Time (µs)': f"{avg_time:.2f}",
            'Avg Comparisons': f"{avg_comps:.2f}",
            'Success Rate': f"{success_rate:.1f}%",
            'Parameters': params_str,
            'Memory': memory_str,
            'Max Error': error_str,
            'Training Time (ms)': f"{train_time:.2f}" if train_time > 0 else "N/A"
        })
    
    return pd.DataFrame(results_data), data_size, all_loss_data

# Run benchmark when button is clicked
if run_benchmark:
    if not any([run_full_scan, run_binary, run_exponential, run_pytorch_linear, run_pytorch_mlp, run_decision_tree]):
        st.warning("Please select at least one method to benchmark!")
    else:
        with st.spinner("Running benchmark..."):
            result_df, data_size, loss_data = run_benchmark_pipeline()
            if result_df is not None:
                st.session_state.results = result_df
                st.session_state.data_size = data_size
                st.session_state.loss_data = loss_data

# Display results
if st.session_state.results is not None:
    st.markdown("---")
    st.subheader("Benchmark Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", f"{st.session_state.data_size:,} points")
    with col2:
        st.metric("Coordinate Resolution", f"{coord_max} bits")
    with col3:
        st.metric("Benchmark Runs", num_runs)
    
    st.dataframe(st.session_state.results, use_container_width=True, hide_index=True)
    
    # Display PyTorch training loss curves
    if 'loss_data' in st.session_state and st.session_state.loss_data:
        st.markdown("---")
        st.subheader("🔥 PyTorch Training Loss Curves")
        
        # Create combined loss plot
        loss_fig = go.Figure()
        
        colors = {'PyTorch Linear': '#1f77b4', 'PyTorch MLP': '#ff7f0e'}
        
        for model_name, data in st.session_state.loss_data.items():
            if data['epoch']:
                loss_fig.add_trace(go.Scatter(
                    x=data['epoch'],
                    y=data['loss'],
                    mode='lines',
                    name=model_name,
                    line=dict(color=colors.get(model_name, '#2ca02c'), width=2),
                    hovertemplate=f'{model_name}<br>Epoch: %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>'
                ))
        
        loss_fig.update_layout(
            title="Training Loss Comparison",
            xaxis_title="Epoch",
            yaxis_title="MSE Loss",
            height=400,
            hovermode='x unified',
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        
        st.plotly_chart(loss_fig, use_container_width=True)
        
        # Show final loss values
        loss_summary_cols = st.columns(len(st.session_state.loss_data))
        for idx, (model_name, data) in enumerate(st.session_state.loss_data.items()):
            with loss_summary_cols[idx]:
                final_loss = data['loss'][-1] if data['loss'] else 0
                st.metric(f"{model_name} Final Loss", f"{final_loss:.4f}")
    
    # Visualization
    st.markdown("---")
    st.subheader("Performance Comparison")
    
    tab1, tab2 = st.tabs(["Average Time", "Average Comparisons"])
    
    with tab1:
        chart_data = st.session_state.results.copy()
        chart_data['Avg Time (µs)'] = chart_data['Avg Time (µs)'].astype(float)
        st.bar_chart(chart_data.set_index('Method')['Avg Time (µs)'])
    
    with tab2:
        chart_data = st.session_state.results.copy()
        chart_data['Avg Comparisons'] = chart_data['Avg Comparisons'].astype(float)
        st.bar_chart(chart_data.set_index('Method')['Avg Comparisons'])

else:
    st.info("Configure your benchmark settings in the sidebar and click 'Run Benchmark' to start.")
    
    st.markdown("""
    ### How to Use
    
    #### Spatial POI Data Mode:
    1. **Select Dataset Type**: Choose "Spatial POI Data" from the dataset type selector
    2. **Configure Dataset**: Set the dataset size and coordinate resolution
    3. **Select Models**: Choose which search methods and learned index models to benchmark
    4. **Tune Hyperparameters**: Expand the settings for each model to adjust hyperparameters
    5. **Run Benchmark**: Click the "Run Benchmark" button to start the evaluation
    6. **Analyze Results**: View the performance comparison table and charts
    
    #### Synthetic 1D Data Mode:
    1. **Select Dataset Type**: Choose "Synthetic 1D Data" from the dataset type selector
    2. **Configure Buckets**: Set the number of histogram buckets
    3. **Draw Distribution**: Use the canvas to create your desired data distribution
    4. **Configure Dataset Size**: Set the total number of data points (always uses 32-bit values)
    5. **Select Models and Run**: Choose models, tune hyperparameters, and run benchmark
    
    ### Key Hyperparameters
    
    - **PyTorch Models**: Epochs (training iterations), learning rate, batch size
    - **MLP**: Hidden layer size and number of layers control model complexity
    - **Decision Tree**: Max depth controls overfitting, min samples split affects granularity
    """)
