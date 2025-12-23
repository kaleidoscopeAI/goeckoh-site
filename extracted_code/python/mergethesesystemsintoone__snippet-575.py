"""Calculates the perpendicular distance of a point from a line segment."""
x0, y0 = point
x1, y1 = start
x2, y2 = end
if x1 == x2 and y1 == y2:
    return math.hypot(x0 - x1, y0 - y1)
else:
    return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.hypot(x2 - x1, y2 - y1)

