# --- Z-Order Curve Implementation ---

def coords_to_z_order(x: int, y: int, bits: int = 32) -> int:
    """
    Converts 2D coordinates to a 1D Z-order value by interleaving bits.
    This is a common technique for creating a spatial ordering.
    """
    z = 0
    for i in range(bits):
        # Interleave bits from x and y
        z |= (x & (1 << i)) << i | (y & (1 << i)) << (i + 1)
    return z
