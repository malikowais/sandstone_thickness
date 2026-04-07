import cv2
import imutils
import numpy as np
from imutils import contours

id2name = {0: "scale", 1: "sandstone", 2: "mudstone"}
name2id = {"scale": 0, "sandstone": 1, "mudstone": 2}


def group_width(group):
    """Get the width of a flat line"""
    return group[:, 0].max() - group[:, 0].min()


def group_points_by_y(points, y_range: float = 5):
    """
    Groups a list of 2D points based on their y-coordinates. Points whose y-coordinates are within a range of y_range are
    considered part of the same group.

    Args:
        points (list of list of int or float): A list of 2D points, where each point is represented as a list
            of two coordinates, [x, y].
        y_range (int): An int used to control the range of y coordinates in a group

    Returns:
        list of list of int or float: A list of point groups, where each group is represented as a list of
            points. Points within a group have y-coordinates that are within a range of y_range.

    Raises:
        TypeError: If the input points are not a list of list of int or float.

    Example:
        >>> points = [[0, 0], [1, 2], [2, 2], [3, 3], [4, 5], [5, 5]]
        >>> group_points_by_y(points, y_range=2)
        [[[0, 0], [1, 2], [2, 2]], [[3, 3], [4, 5], [5, 5]]]
    """
    # sort the points by y-coordinate
    points = np.squeeze(points)
    points = points[np.argsort(points[:, 1])]
    groups = []
    group = []
    for i in range(len(points)):
        if len(group) == 0:
            group.append(points[i].tolist())
        else:
            # check if the y-coordinate of the current point is within a range of y_range
            y1 = group[0][1]
            y2 = points[i][1]
            if abs(y2 - y1) <= y_range:
                group.append(points[i].tolist())
            else:
                groups.append(group)
                group = [points[i].tolist()]
    # don't create groups with 1 point only (the last point)
    if len(group) > 1:
        groups.append(group)
    return groups


def lies_inside_contour(cont, x1, x2, y):
    """check if the points at y between x1 and x2 lies inside the contour"""
    for x in range(x1 + 1, x2):
        p = np.array([x, y], dtype="uint8")
        result = cv2.pointPolygonTest(cont, p, False)
        if result == -1:
            return False
    return True


def get_width_and_residuals(group, cont):
    """
    group is sorted by x-coordinate
    this function should be called again on g2 until len(g2) <= 1
    """
    # Get the mean y-coordinate
    y_mean = np.mean(group[:, 1]).astype("uint8")
    x1, y1 = group[0]
    x_max = x1
    y2 = y1
    remaining_group = None
    for i in range(1, len(group)):
        x2, y2 = group[i]
        y_mean = (y1 + y2) // 2
        # mask errors at extremes
        if y_mean > 250:
            y_mean = 250
        if y_mean < 10:
            y_mean = 5
        if lies_inside_contour(cont, x1, x2, y_mean):
            if x2 > x_max:
                x_max = x2
        else:
            remaining_group = group[i:]
            break
    return (x1, x_max, (y1 + y2) // 2), remaining_group


def keep_varying_lines_based_on_similarity(lines, y_range, threshold):
    """
    Remove lines with similar widths and if they are close to each other
    """
    if not lines:
        return []
    lines.sort(key=lambda line: line[0][1])  # Sort by y value
    new_lines = [lines[0]]
    for i in range(1, len(lines)):
        current_line, previous_line = lines[i], new_lines[-1]
        if abs(current_line[0][1] - previous_line[0][1]) <= 2 * y_range:
            width_prev = group_width(previous_line)
            width_curr = group_width(current_line)
            if abs(width_prev - width_curr) < threshold:  # If widths are too similar
                # If the previous line was more similar to its neighbor than the current line, remove the previous line
                if i != len(lines) - 1:
                    next_line = lines[i + 1]
                    width_next = group_width(next_line)
                    if abs(width_next - width_curr) > abs(width_next - width_prev):
                        new_lines.pop()
                        new_lines.append(current_line)
                    else:
                        continue
                else:
                    new_lines.pop()
                    new_lines.append(current_line)
            else:
                new_lines.append(current_line)
        else:
            new_lines.append(current_line)
    return new_lines


def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point."""
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x * xx + y * yx, y0 + x * xy + y * yy
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy


def trace_contour(contour):
    """contour tracing to interpolate points along the contour edges"""
    contour_new = []
    for i in range(len(contour) - 1):
        p1 = contour[i][0]
        p2 = contour[i + 1][0]
        interpolated = list(bresenham(p1[0], p1[1], p2[0], p2[1]))
        contour_new.extend(interpolated)
    contour_new.append(contour[-1][0])
    contour = np.array(contour_new, dtype="int32")
    contour = np.expand_dims(contour, axis=1)
    return contour


def find_flat_lines(
    contour: np.ndarray,
    img: np.ndarray,
    w_min: float = 0,
    w_max: float = float("inf"),
    line_sim_thresh: float = 5.0,
    ratio: float = 1,
    group_y_range: float = 2,
):
    # perform contour tracing
    contour = trace_contour(contour)
    # group points by their y-coordinates
    groups = group_points_by_y(contour, y_range=group_y_range)
    # initialize the list of flat lines
    flat_lines = []
    # iterate over each group of points
    for i in range(len(groups)):
        # convert to numpy array
        group = np.array(groups[i])
        # sort group by x-coordinate
        group = group[np.argsort(group[:, 0])]
        remaining_group = group.copy()
        while remaining_group is not None and len(remaining_group) >= 2:
            (x_min, x_max, y_mean), remaining_group = get_width_and_residuals(
                remaining_group, contour
            )
            # compute the width of the group
            width = x_max - x_min
            # add the group if its width is >= the threshold
            if w_min <= width <= w_max:
                line = np.array([[x_min, y_mean], [x_max, y_mean]])
                flat_lines.append(line)
    # remove lines with similar widths
    flat_lines = keep_varying_lines_based_on_similarity(
        flat_lines, group_y_range, line_sim_thresh
    )
    # the number of flat lines is the number of groups
    num_flat_lines = len(flat_lines)
    # add arrows to show the flat lines
    avg_width = 0
    for i in range(num_flat_lines):
        line = np.array(flat_lines[i])
        x_min = np.min(line[:, 0]).astype("uint8")
        x_max = np.max(line[:, 0]).astype("uint8")
        y = np.mean(line[:, 1]).astype("uint8")
        width = (x_max - x_min) / ratio
        width = round(width, 1)
        avg_width += width
        text = str(width)
        yloc = int(y) + 10
        if yloc >= 240:
            yloc = int(y) - 5
        cv2.arrowedLine(img, (x_min, y), (x_max, y), (0, 255, 0), 2)
        if x_min >= 230:
            x_min -= 5
        cv2.putText(
            img, text, (x_min, yloc), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    if num_flat_lines > 0:
        avg_width /= num_flat_lines
        avg_width = round(avg_width, 1)
    else:
        avg_width = 0
    return avg_width, img


def draw_widths(
    img: np.ndarray,
    mask: np.ndarray,
    ratio: float = 1.0,
    w_min: float = 0,
    w_max: float = float("inf"),
    line_sim_thresh: float = 0.0,
    group_y_range: float = 1.0,
):
    # threshold the mudstone
    thresh = np.where(mask == name2id["mudstone"], 255, 0).astype(np.uint8)
    kernel = np.ones((32, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right
    (cnts, _) = contours.sort_contours(cnts)
    # adjust thresholds based on ratio
    w_min *= ratio
    w_max *= ratio
    line_sim_thresh *= ratio
    group_y_range *= ratio
    anno_imgs = []

    img_to_draw_on = img.copy()
    for c in cnts:
        orig = img.copy()
        # copy the original image to prevent editing it
        h, w = orig.shape[:2]
        # minimum contour area should be >= 2% of image
        area = cv2.contourArea(c)
        perc = area / (h * w) * 100
        if perc < 2:
            continue
        cv2.drawContours(orig, [c], 0, (255, 0, 0), 1)
        cv2.drawContours(img_to_draw_on, [c], 0, (255, 0, 0), 1)
        # find the widths
        avg_width, orig_lines_drawn = find_flat_lines(
            c,
            orig,
            w_min=w_min,
            w_max=w_max,
            line_sim_thresh=line_sim_thresh,
            ratio=ratio,
            group_y_range=group_y_range,
        )
        avg_width, img_to_draw_on = find_flat_lines(
            c,
            img_to_draw_on,
            w_min=w_min,
            w_max=w_max,
            line_sim_thresh=line_sim_thresh,
            ratio=ratio,
            group_y_range=group_y_range,
        )
        anno_imgs.append(orig_lines_drawn)
    return anno_imgs, img_to_draw_on
