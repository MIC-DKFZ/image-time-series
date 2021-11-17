import numpy as np
from skimage import draw


def circle(center_x=0.5, center_y=0.5, radius=0.25, image_size=64, **kwargs):

    img = np.zeros((image_size, image_size), dtype=np.uint8)
    center = np.array([center_x, center_y]) * image_size
    radius *= image_size
    cc, rr = draw.disk(center, radius, shape=img.shape)

    cc = cc[rr >= 0]
    rr = rr[rr >= 0]
    rr = rr[cc >= 0]
    cc = cc[cc >= 0]

    img[rr, cc] = 1

    return img


def polygon(points_x, points_y, rotation=0.0, image_size=64, **kwargs):

    img = np.zeros((image_size, image_size), dtype=np.uint8)

    points_x = np.array(points_x) * image_size
    points_y = np.array(points_y) * image_size

    if rotation > 0.0:

        rotation = rotation % 1.0

        center_x = np.mean(points_x)
        center_y = np.mean(points_y)
        points_x_c = points_x - center_x
        points_y_c = points_y - center_y

        points_x = points_x_c * np.cos(2 * np.pi * rotation) - points_y_c * np.sin(
            2 * np.pi * rotation
        )
        points_y = points_y_c * np.cos(2 * np.pi * rotation) + points_x_c * np.sin(
            2 * np.pi * rotation
        )

        points_x = points_x + center_x
        points_y = points_y + center_y

    cc, rr = draw.polygon(points_x, points_y, shape=img.shape)

    img[rr, cc] = 1

    return img


def rectangle(
    start_x=0.25,
    start_y=0.25,
    end_x=0.75,
    end_y=0.75,
    rotation=0.0,
    image_size=64,
    **kwargs
):

    start = np.array([start_x, start_y])
    end = np.array([end_x, end_y])
    c1 = np.array([start_x, end_y])
    c2 = np.array([end_x, start_y])
    points = np.stack((start, c1, end, c2))

    return polygon(points[:, 0], points[:, 1], rotation=rotation, image_size=image_size)


def rectangle_c(
    center_x=0.5,
    center_y=0.5,
    size_x=0.5,
    size_y=0.5,
    rotation=0.0,
    image_size=64,
    **kwargs
):
    """Like rectangle, but specified with center and size."""

    start = (center_x - size_x / 2.0, center_y - size_y / 2.0)
    end = (center_x + size_x / 2.0, center_y + size_y / 2.0)

    return rectangle(*start, *end, rotation, image_size, **kwargs)


def square(center_x=0.5, center_y=0.5, size=0.5, rotation=0.0, image_size=64, **kwargs):

    start = (center_x - size / 2.0, center_y - size / 2.0)
    end = (center_x + size / 2.0, center_y + size / 2.0)

    return rectangle(*start, *end, rotation=rotation, image_size=image_size)


def triangle_square(
    center_x=0.5, center_y=0.5, size=0.5, t=0.5, rotation=0.0, image_size=64, **kwargs
):

    x = np.array(
        [
            center_x - size / 2.0,
            center_x - size / 2.0,
            center_x + size / 2.0,
            center_x + t * size / 2.0,
        ]
    )
    y = np.array(
        [
            center_y - size / 2.0,
            center_y + size / 2.0,
            center_y + size / 2.0,
            center_y - t * size / 2.0,
        ]
    )

    return polygon(x, y, rotation, image_size, **kwargs)


def star(
    center_x=0.5, center_y=0.5, size=0.5, t=0.5, rotation=0.0, image_size=64, **kwargs
):

    x = np.array(
        [
            center_x - size / 2.0,
            center_x - t * size / 2.0,
            center_x - size / 2.0,
            center_x,
            center_x + size / 2.0,
            center_x + t * size / 2.0,
            center_x + size / 2.0,
            center_x,
        ]
    )

    y = np.array(
        [
            center_y - size / 2.0,
            center_y,
            center_y + size / 2.0,
            center_y + t * size / 2,
            center_y + size / 2.0,
            center_y,
            center_y - size / 2.0,
            center_y - t * size / 2.0,
        ]
    )

    return polygon(x, y, rotation, image_size, **kwargs)


def get_square_intersection_points(start, end):

    start = np.array(start)
    end = np.array(end)

    if np.any(start < 0) or np.any(start > 1):

        possible_new_starts = []

        # x = 0
        if end[0] != start[0]:
            y = start[1] - start[0] * (end[1] - start[1]) / (end[0] - start[0])
            if 0 <= y <= 1:
                possible_new_starts.append(np.array((0, y)))

        # x = 1
        if end[0] != start[0]:
            y = start[1] + (1 - start[0]) * (end[1] - start[1]) / (end[0] - start[0])
            if 0 <= y <= 1:
                possible_new_starts.append(np.array((1, y)))

        # y = 0
        if end[1] != start[1]:
            x = start[0] - start[1] * (end[0] - start[0]) / (end[1] - start[1])
            if 0 <= x <= 1:
                possible_new_starts.append(np.array((x, 0)))

        # y = 1
        if end[1] != start[1]:
            x = start[0] + (1 - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
            if 0 <= x <= 1:
                possible_new_starts.append(np.array((x, 1)))

        if len(possible_new_starts) == 0:
            return None, None
        else:
            distances = []
            for new_start in possible_new_starts:
                distances.append(np.sum(np.power(new_start - start, 2)))
            start = possible_new_starts[np.argmin(distances)]

    if np.any(end < 0) or np.any(end > 1):

        possible_new_ends = []

        # x = 0
        if end[0] != start[0]:
            y = start[1] - start[0] * (end[1] - start[1]) / (end[0] - start[0])
            if 0 <= y <= 1:
                possible_new_ends.append(np.array((0, y)))

        # x = 1
        if end[0] != start[0]:
            y = start[1] + (1 - start[0]) * (end[1] - start[1]) / (end[0] - start[0])
            if 0 <= y <= 1:
                possible_new_ends.append(np.array((1, y)))

        # y = 0
        if end[1] != start[1]:
            x = start[0] - start[1] * (end[0] - start[0]) / (end[1] - start[1])
            if 0 <= x <= 1:
                possible_new_ends.append(np.array((x, 0)))

        # y = 1
        if end[1] != start[1]:
            x = start[0] + (1 - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
            if 0 <= x <= 1:
                possible_new_ends.append(np.array((x, 1)))

        if len(possible_new_ends) == 0:
            return None, None
        else:
            distances = []
            for new_end in possible_new_ends:
                distances.append(np.sum(np.power(new_end - end, 2)))
            end = possible_new_ends[np.argmin(distances)]

    return start, end


def line(start_x=0.25, start_y=0.25, end_x=0.75, end_y=0.75, image_size=64, **kwargs):

    img = np.zeros((image_size, image_size), dtype=np.uint8)

    start, end = get_square_intersection_points((start_x, start_y), (end_x, end_y))
    if start is None or end is None:
        return img
    else:
        start = np.round(start * image_size).astype(np.uint8)
        end = np.round(end * image_size).astype(np.uint8)

    cc, rr = draw.line(*start, *end)
    img[rr, cc] = 1

    return img


def dial(
    center_object="circle",
    center_x=0.5,
    center_y=0.5,
    center_radius=0.1,
    center_rotation=0.0,
    outer_object="circle",
    outer_radius=0.05,
    outer_distance=0.05,
    outer_rotation=0.0,
    rotation=0.0,
    image_size=64,
    **kwargs
):

    if center_object == "circle":
        center_object = circle(center_x, center_y, center_radius, image_size)
    elif center_object == "square":
        center_object = square(
            center_x, center_y, center_radius, center_rotation, image_size
        )
    else:
        raise ValueError(
            "At the moment center_object must be from ('circle', 'square'), but is {}".format(
                center_object
            )
        )

    radius = center_radius + outer_radius + outer_distance
    outer_x = center_x + radius * np.sin(2 * np.pi * rotation)
    outer_y = center_y - radius * np.cos(2 * np.pi * rotation)

    if outer_object == "circle":
        outer_object = circle(outer_x, outer_y, outer_radius, image_size)
    elif outer_object == "square":
        outer_object = square(
            outer_x, outer_y, outer_radius, outer_rotation, image_size
        )
    else:
        raise ValueError(
            "At the moment center_object must be from ('circle', 'square'), but is {}".format(
                center_object
            )
        )

    center_object[outer_object > 0] = 1

    return center_object