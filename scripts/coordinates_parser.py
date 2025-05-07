import ast

def parse_coordinates(xy_str):
    """Convert string coordinates to list of tuples"""
    try:
        if isinstance(xy_str, str):
            if xy_str.startswith('['):
                coords = ast.literal_eval(xy_str)
            else:
                coords = [float(x) for x in xy_str.split(',')]
            return list(zip(coords[::2], coords[1::2]))
    except:
        return None