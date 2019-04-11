
def _det3(p0, p1, p2) -> float:
    return p0[0] * p1[1] + p1[0] * p2[1] + p2[0] * p0[1] - p0[1] * p1[0] - p1[1] * p2[0] - p2[1] * p0[0]


def dot(p0, p1, p2, p3) -> float:
    """Calculare cosinue of angle between vectors p0->p1 and p2->p3"""
    v0 = (p1[0] - p0[0], p1[1] - p0[1])
    v1 = (p3[0] - p2[0], p3[1] - p2[1])

    return (v0[0] * v1[0] + v0[1] * v1[1]) / (math.hypot(*v0) * math.hypot(*v1))


def intersect(p0, p1, p2, p3) -> bool:
    """Check if p0--p1 intersect with p2--p3"""
    return (_det3(p0, p1, p2) * _det3(p0, p1, p3) <= 0) and (_det3(p2, p3, p0) * _det3(p2, p3, p1) <= 0)


def intersection(p0, p1, p2, p3):
    """Get intersection point of p0--p1 with p2--p3."""
    assert p0[0] != p1[0] # vertical line

    # ax + b = y
    a = (p1[1] - p0[1]) / (p1[0] - p0[0])
    b = p0[1] - a * p0[0]

    assert p2[0] != p3[0]
    c = (p3[1] - p2[1]) / (p3[0] - p2[0])
    d = p2[1] - c*p2[0]

    assert a != c
    x = (d-b)/(a-c)
    y = a*x+b

    assert abs(_det3(p0, p1, (x,y))) < 0.01
    assert abs(_det3(p2, p3, (x,y))) < 0.01
    return (x, y)
