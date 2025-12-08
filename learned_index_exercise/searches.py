# --- Baseline Search Algorithms (Provided for Benchmarking) ---

def search_full_scan(data: list[int], query: int) -> tuple[int, int]:
    """
    Scans the entire array to find the element.
    Returns the index of the element and the number of comparisons.
    """
    comparisons = 0
    for i, val in enumerate(data):
        comparisons += 1
        if val == query:
            return i, comparisons
    return -1, comparisons

def search_binary(data: list[int], query: int) -> tuple[int, int]:
    """
    Binary search algorithm.
    Returns the index of the element and the number of comparisons.
    """
    low, high = 0, len(data) - 1
    comparisons = 0
    while low <= high:
        comparisons += 1
        mid = (low + high) // 2
        if data[mid] == query:
            return mid, comparisons
        elif data[mid] < query:
            low = mid + 1
        else:
            high = mid - 1
    return -1, comparisons

def search_exponential(data: list[int], query: int) -> tuple[int, int]:
    """
    Baseline 3: Exponential search.
    Finds range where element may exist by exponentially increasing the index,
    then performs binary search within that range.
    Returns the index of the element and the number of comparisons.
    """
    n = len(data)
    if n == 0:
        return -1, 0
    
    comparisons = 0
    
    # Check if the first element is the target
    comparisons += 1
    if data[0] == query:
        return 0, comparisons
    
    # Find range for binary search by doubling index
    i = 1
    while i < n and data[i] <= query:
        comparisons += 1
        if data[i] == query:
            return i, comparisons
        i *= 2
    
    # Perform binary search in the found range
    low = i // 2
    high = min(i, n - 1)
    
    while low <= high:
        comparisons += 1
        mid = (low + high) // 2
        if data[mid] == query:
            return mid, comparisons
        elif data[mid] < query:
            low = mid + 1
        else:
            high = mid - 1
    
    return -1, comparisons
