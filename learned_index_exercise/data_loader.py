import csv
import os
import subprocess
import zipfile
import numpy as np
from scipy.stats import gaussian_kde
from .utils import coords_to_z_order


def download_and_extract_data():
    """
    Downloads and extracts the POI dataset if it doesn't exist.
    """
    file_path = 'data/poi_data.csv'
    if os.path.exists(file_path):
        return

    print("Data not found. Downloading and extracting...")

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    zip_path = 'data/poi_database.zip'
    url = 'https://www.kaggle.com/api/v1/datasets/download/ehallmar/points-of-interest-poi-database'

    try:
        # Download the file using curl
        print(f"Downloading data from {url}...")
        subprocess.run(['curl', '-L', '-o', zip_path, url], check=True)

        # Unzip the file
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Assuming the csv is named poi.csv in the zip
            zip_ref.extract('poi.csv', 'data')
            os.rename('data/poi.csv', file_path)

        # Clean up the zip file
        os.remove(zip_path)
        print("Download and extraction complete.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"An error occurred during download and extraction: {e}")
        print("Please ensure you have 'curl' and 'unzip' installed and in your PATH.")
        print("Alternatively, download the dataset manually from Kaggle and place 'poi_data.csv' in the 'data' directory.")
        print("Kaggle dataset URL: https://www.kaggle.com/datasets/ehallmar/points-of-interest-poi-database")
        # Clean up partial files if they exist
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists('data/poi.csv'):
            os.remove('data/poi.csv')
    except Exception as e:
        print(f"An error occurred: {e}")


def extend_data_with_kde(points: list[dict], target_size: int) -> list[dict]:
    """
    Extends the dataset to a target size using Kernel Density Estimation.
    """
    if len(points) >= target_size:
        return points

    print(f"Extending dataset from {len(points)} to {target_size} points using KDE...")
    
    # Prepare data for KDE (latitudes and longitudes)
    lats = [p['lat'] for p in points]
    lons = [p['lon'] for p in points]
    
    # Create a 2D array of shape (2, n_points)
    values = np.vstack([lons, lats])
    
    # Fit KDE model
    kde = gaussian_kde(values)
    
    # Generate new samples
    num_new_points = target_size - len(points)
    new_samples = kde.resample(size=num_new_points)
    
    # Add new points to the list
    new_points = [{'lat': lat, 'lon': lon} for lon, lat in zip(new_samples[0], new_samples[1])]
    
    return points + new_points


def load_and_process_data(coord_max: int, target_size: int = 0) -> list[tuple[int, int]]:
    """
    Loads POI data from the CSV, optionally extends it, and then normalizes and quantizes the coordinates.
    Downloads the data if it is not found.
    Returns:
        A list of tuples, where each tuple contains the quantized (x, y) coordinates.
    """
    download_and_extract_data()

    points = []
    file_path = 'data/poi_data.csv'

    # 1. Read raw latitude/longitude from CSV
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                # lat is in col 1 (latitude_radian), lon is in col 2 (longitude_radian)
                lat = float(row[1])
                lon = float(row[2])
                points.append({'lat': lat, 'lon': lon})
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return []
    except (ValueError, IndexError) as e:
        print(f"Error processing file {file_path}: {e}")
        return []

    if not points:
        return []
    
    # Extend the dataset if a target_size is provided
    if target_size > len(points):
        points = extend_data_with_kde(points, target_size)

    # 2. Find the bounding box of the data
    min_lat = min(p['lat'] for p in points)
    max_lat = max(p['lat'] for p in points)
    min_lon = min(p['lon'] for p in points)
    max_lon = max(p['lon'] for p in points)

    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    # Avoid division by zero if all points are the same
    if lat_range == 0 or lon_range == 0:
        return [(0, 0)] * len(points)

    # 3. Normalize and Quantize the points to fill the theoretical z-space
    quantized_points = []
    seen_z_values = set()
    
    # Calculate spacing to spread points evenly across the coordinate space
    # This helps fill the theoretical z-space more uniformly
    target_points = len(points)
    ideal_spacing = int(np.sqrt(coord_max / target_points)) if target_points > 0 else 1
    
    for i, point in enumerate(points):
        # Normalize to [0, 1]
        norm_lat = (point['lat'] - min_lat) / lat_range
        norm_lon = (point['lon'] - min_lon) / lon_range

        # Quantize to [0, coord_max] with added spacing to spread points
        # Use floating point for more precise distribution
        quant_x = int(norm_lon * coord_max)
        quant_y = int(norm_lat * coord_max)
        
        # Add small pseudo-random jitter based on index to spread duplicates
        z_val = coords_to_z_order(quant_x, quant_y)
        jitter_offset = 0
        max_jitter = min(100, ideal_spacing)  # Scale jitter with spacing
        
        while z_val in seen_z_values and jitter_offset < max_jitter:
            jitter_offset += 1
            # Use pseudo-random jitter pattern based on point index and offset
            jitter_x = (jitter_offset * ((i * 7) % 11 - 5)) % (ideal_spacing + 1)
            jitter_y = (jitter_offset * ((i * 13) % 17 - 8)) % (ideal_spacing + 1)
            
            new_x = max(0, min(coord_max, quant_x + jitter_x))
            new_y = max(0, min(coord_max, quant_y + jitter_y))
            z_val = coords_to_z_order(new_x, new_y)
        
        if z_val not in seen_z_values:
            seen_z_values.add(z_val)
            final_x = quant_x if jitter_offset == 0 else new_x
            final_y = quant_y if jitter_offset == 0 else new_y
            quantized_points.append((final_x, final_y))
    
    num_skipped = len(points) - len(quantized_points)
    if num_skipped > 0:
        print(f"Skipped {num_skipped} points that couldn't be uniquely placed with jitter.")
    
    print(f"Loaded and processed {len(quantized_points)} unique data points.")
    return quantized_points
