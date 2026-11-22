"""Classify traffic density from the number of vehicles currently visible."""


def classify_density(vehicle_count: int) -> str:
    """Return a simple density label for the current vehicle count."""
    if vehicle_count < 0:
        raise ValueError("vehicle_count cannot be negative")

    if vehicle_count <= 8:
        return "Low"
    if vehicle_count <= 20:
        return "Medium"
    return "High"