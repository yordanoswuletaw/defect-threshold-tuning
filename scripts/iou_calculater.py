from shapely.geometry import Polygon

def calculate_iou(poly1_coords, poly2_coords):
    """Calculate IoU between two polygons"""
    try:
        poly1 = Polygon(poly1_coords)
        poly2 = Polygon(poly2_coords)
        
        if not (poly1.is_valid and poly2.is_valid):
            return 0.0
        
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area
    except:
        return 0.0