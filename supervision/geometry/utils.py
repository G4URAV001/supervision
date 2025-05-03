import numpy as np

from supervision.geometry.core import Point


def get_polygon_center(polygon: np.ndarray) -> Point:
    """
    Calculate the center of a polygon. The center is calculated as the center
    of the solid figure formed by the points of the polygon

    Parameters:
        polygon (np.ndarray): A numpy ndarray representing the vertices of the polygon.
            Can be either a 2D array of shape (n, 2) or a 3D array of shape (n, 1, 2).

    Returns:
        Point: The center of the polygon, represented as a
            Point object with x and y attributes.

    Raises:
        ValueError: If the polygon has no vertices.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        polygon = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        sv.get_polygon_center(polygon=polygon)
        # Point(x=1, y=1)
        ```
    """

    # This is one of the 3 candidate algorithms considered for centroid calculation.
    # For a more detailed discussion, see PR #1084 and commit eb33176

    if len(polygon) == 0:
        raise ValueError("Polygon must have at least one vertex.")
    
    # Handle both 2D and 3D arrays
    original_shape = polygon.shape
    
    # Convert to 3D array with shape (n, 1, 2) for NumPy 2.0 compatibility
    if len(original_shape) == 2:
        # For 2D arrays, reshape to (n, 1, 2)
        polygon_3d = polygon.reshape(original_shape[0], 1, 2)
    else:
        # Already a 3D array
        polygon_3d = polygon
    
    # Extract 2D points for calculations
    polygon_2d = polygon_3d.reshape(polygon_3d.shape[0], -1)[:, :2]
    
    # Calculate signed areas using 2D representation
    shift_polygon = np.roll(polygon_2d, -1, axis=0)
    
    # Manual cross product calculation to avoid NumPy 2.0 deprecation warning
    # For 2D vectors [x1, y1] and [x2, y2], the cross product is x1*y2 - y1*x2
    signed_areas = (polygon_2d[:, 0] * shift_polygon[:, 1] - 
                   polygon_2d[:, 1] * shift_polygon[:, 0]) / 2
    if signed_areas.sum() == 0:
        center = np.mean(polygon_2d, axis=0).round()
        return Point(x=center[0], y=center[1])
    centroids = (polygon_2d + shift_polygon) / 3.0
    center = np.average(centroids, axis=0, weights=signed_areas).round()

    return Point(x=center[0], y=center[1])
