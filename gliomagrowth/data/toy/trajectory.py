import numpy as np


def line(N, start=(0.0, 0.0), end=(1.0, 1.0)):
    return list(zip(*list(map(lambda x: np.linspace(*x, N).tolist(), zip(start, end)))))


def line_(N, start_x=0.0, start_y=0.0, end_x=1.0, end_y=1.0):
    return line(N, (start_x, start_y), (end_x, end_y))


def circle(N, center=(0.5, 0.5), radius=0.25, phase=0.0):

    angles = np.linspace(0, 2 * np.pi, N + 1)[:-1]
    pos = radius * np.array(list(map(np.exp, (angles + phase) * 1j)))
    x = center[0] + pos.real
    y = center[1] + pos.imag
    return list(zip(x, y))


def circle_(N, center_x=0.5, center_y=0.5, radius=0.25, phase=0.0):
    return circle(N, (center_x, center_y), radius, phase)


def polygon(N, coords, closed=True):

    element_lengths = []
    for c1, coord1 in enumerate(coords):
        coord1 = np.array(coord1)
        coord2 = np.array(coords[(c1 + 1) % len(coords)])
        element_lengths.append(np.linalg.norm(coord2 - coord1))
    if not closed:
        element_lengths = element_lengths[:-1]
    element_lengths_cumulative = np.cumsum(element_lengths)

    intervals = np.linspace(0, sum(element_lengths), N + int(closed))

    result_coords = [coords[0]]
    for interval in intervals[1:]:
        el = 0
        while interval > element_lengths_cumulative[el]:
            el += 1
        if el > 0:
            interval = interval - element_lengths_cumulative[el - 1]
        section_ratio = interval / float(element_lengths[el])
        interval_coord = np.array(coords[el]) + section_ratio * (
            np.array(coords[(el + 1) % len(coords)]) - np.array(coords[el])
        )
        result_coords.append(tuple(interval_coord))
    if closed:
        result_coords = result_coords[:-1]

    return result_coords


def poly4(N, x1, y1, x2, y2, x3, y3, x4, y4, closed=True):
    coords = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    return polygon(N, coords, closed)


def square(N, center=(0.5, 0.5), width=0.5):
    coords = []
    coords.append((center[0] - 0.5 * width, center[0] - 0.5 * width))
    coords.append((center[0] + 0.5 * width, center[0] - 0.5 * width))
    coords.append((center[0] + 0.5 * width, center[0] + 0.5 * width))
    coords.append((center[0] - 0.5 * width, center[0] + 0.5 * width))
    return polygon(N, coords, closed=True)


def square_(N, center_x=0.5, center_y=0.5, width=0.5):
    return square(N, (center_x, center_y), width)
