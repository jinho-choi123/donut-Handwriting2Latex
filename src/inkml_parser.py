from dataclasses import dataclass
import numpy as np
from xml.etree import ElementTree
import matplotlib.pyplot as plt
from .config import config

IMG_SIZE = config.get("IMG_SIZE", 224)
PADDING = config.get("INK_PADDING", 4)
SEQ_MAX = config.get("INK_SEQ_MAX", 500)
SEQ_MIN = config.get("INK_SEQ_MIN", -500)


# Define Ink class
@dataclass
class Ink:
    """Represents a single ink, as read from an InkML file."""

    # Every stroke in the ink.

    # Each stroke array has shape (3, number of points), where the first

    # dimensions are (x, y, timestamp), in that order.

    strokes: list[np.ndarray]

    # Metadata present in the InkML.

    annotations: dict[str, str]

    min_x: int

    min_y: int

    max_x: int

    max_y: int

    min_t: int

    max_t: int

    max_delta_x: int

    max_delta_y: int


# Define function that reads inkml file, and outputs Ink object


def read_inkml_file(filename: str) -> Ink:
    """Simple reader for MathWriting's InkML files."""

    with open(filename, "r") as f:
        root = ElementTree.fromstring(f.read())

        strokes = []

        annotations = {}

        max_x, max_y, max_t, min_x, min_y, min_t = None, None, None, None, None, None

        max_delta_x, max_delta_y = 0, 0

        for element in root:
            tag_name = element.tag.removeprefix("{http://www.w3.org/2003/InkML}")

            if tag_name == "annotation":
                annotations[element.attrib.get("type")] = element.text

            elif tag_name == "trace":
                points = element.text.split(",")

                stroke_x, stroke_y, stroke_t = [], [], []

                prev_x, prev_y = None, None

                for point in points:
                    x, y, t = [float(p) for p in point.split(" ")]

                    stroke_x.append(x)

                    stroke_y.append(y)

                    stroke_t.append(t)

                    if prev_x is None:
                        prev_x = x

                        prev_y = y

                    if max_x is None:
                        max_x = x

                        min_x = x

                        max_y = y

                        min_y = y

                        max_t = t

                        min_t = t

                    if x > max_x:
                        max_x = x

                    if x < min_x:
                        min_x = x

                    if y > max_y:
                        max_y = y

                    if y < min_y:
                        min_y = y

                    if t > max_t:
                        max_t = t

                    if t < min_t:
                        min_t = t

                    if abs(x - prev_x) > max_delta_x:
                        max_delta_x = abs(x - prev_x)

                    if abs(y - prev_y) > max_delta_y:
                        max_delta_y = abs(y - prev_y)

                    prev_x = x

                    prev_y = y

                strokes.append(np.array((stroke_x, stroke_y, stroke_t)))

    return Ink(
        strokes=strokes,
        annotations=annotations,
        max_x=max_x,
        min_x=min_x,
        max_y=max_y,
        min_y=min_y,
        max_t=max_t,
        min_t=min_t,
        max_delta_x=max_delta_x,
        max_delta_y=max_delta_y,
    )


# display inkml file into image


def display_ink(
    ink: Ink, *, figsize: tuple[int, int] = (15, 10), linewidth: int = 2, color=None
):
    """Simple display for a single ink."""

    plt.figure(figsize=figsize)

    for stroke in ink.strokes:
        plt.plot(stroke[0], stroke[1], linewidth=linewidth, color=color)

        plt.title(
            f"{ink.annotations.get('sampleId', '')} -- "
            f"{ink.annotations.get('splitTagOriginal', '')} -- "
            f"{ink.annotations.get('normalizedLabel', ink.annotations['label'])}"
        )

    plt.gca().invert_yaxis()

    plt.gca().axis("equal")


def get_ink_sequence_token(ink: Ink, timedelta_: int):
    """

    Apply

    1. Time sampling

    2. Scale normalization

    3. Coordinate representation

    """

    # Time sampling

    time_sampled_strokes = []

    # time delta between two adjacent points should be at least timedelta_

    for stroke in ink.strokes:
        stroke_x, stroke_y, stroke_t = stroke[0], stroke[1], stroke[2]

        prev_t = stroke_t[0] - (float(timedelta_) * 2)

        sampled_stroke_x, sampled_stroke_y = [], []

        for x, y, t in zip(stroke_x, stroke_y, stroke_t):
            if t - prev_t >= timedelta_:
                prev_t = t

                # add this point to strokes

                sampled_stroke_x.append(x)

                sampled_stroke_y.append(y)

        time_sampled_strokes.append(np.array((sampled_stroke_x, sampled_stroke_y)))

    # Scale normalization

    scale_normalized_strokes = []

    # print(f'max_x: {max_x}, min_x: {min_x}, max_y: {max_y}, min_y: {min_y}')

    # for every point's x value, (x - min_x) * (IMG_SIZE - 2 * PADDING) / (max_x - min_x) + PADDING

    # for every point's y value, (y - min_y) * IMG_SIZE / (max_y - min_y)

    for stroke in time_sampled_strokes:
        stroke_x, stroke_y = stroke[0], stroke[1]

        normalized_stroke_x, normalized_stroke_y = [], []

        for x, y in zip(stroke_x, stroke_y):
            normalized_stroke_x.append(
                ((x - ink.min_x) * (IMG_SIZE - 2 * PADDING) / (ink.max_x - ink.min_x))
                + PADDING
            )

            normalized_stroke_y.append(
                ((y - ink.min_y) * (IMG_SIZE - 2 * PADDING) / (ink.max_y - ink.min_y))
                + PADDING
            )

        scale_normalized_strokes.append(
            np.array((normalized_stroke_x, normalized_stroke_y))
        )

    # pprint(scale_normalized_strokes)

    # Discretization

    # Converting all float coordinates into int

    discretized_strokes = []

    for stroke in scale_normalized_strokes:
        stroke_x, stroke_y = stroke[0], stroke[1]

        discretized_stroke_x, discretized_stroke_y = [], []

        for x, y in zip(stroke_x, stroke_y):
            discretized_stroke_x.append(round(x))

            discretized_stroke_y.append(round(y))

        discretized_strokes.append(
            np.array((discretized_stroke_x, discretized_stroke_y))
        )

    # pprint(discretized_strokes)

    # Coordinate representation

    relative_position_strokes = []

    for stroke in discretized_strokes:
        stroke_x, stroke_y = stroke[0], stroke[1]

        relative_stroke_x, relative_stroke_y = [], []

        prev_x, prev_y = None, None

        for x, y in zip(stroke_x, stroke_y):
            if prev_x is None and prev_y is None:
                relative_stroke_x.append(x)

                relative_stroke_y.append(y)

                prev_x = x

                prev_y = y

            else:
                relative_stroke_x.append(x - prev_x)

                relative_stroke_y.append(y - prev_y)

        relative_position_strokes.append(
            np.array((relative_stroke_x, relative_stroke_y))
        )

    # pprint(relative_position_strokes)

    # return string of sequences of points

    # new stroke starts with seperator <stroke>

    result = ""

    for stroke in relative_position_strokes:
        stroke_x, stroke_y = stroke[0], stroke[1]

        result += "<stroke> "

        for x, y in zip(stroke_x, stroke_y):
            if x > SEQ_MAX:
                x = SEQ_MAX

            if x < SEQ_MIN:
                x = SEQ_MIN

            if y > SEQ_MAX:
                y = SEQ_MAX

            if y < SEQ_MIN:
                y = SEQ_MIN

            result += f"{x} {y} "

    # print(f'token length: {len(result.split())}')

    return result


def get_ink_image(ink: Ink, figsize: int = 800, linewidth: int = 1):
    """

    returns a ink image of shape (figsize, figsize, 3)

    containing time, delta_x, delta_y information in color channel

    """

    dpi = 100

    width = figsize * 2

    height = figsize // 2

    fig, ax = plt.subplots(figsize=(width // dpi, height // dpi), dpi=dpi)

    ax.axis("off")

    for stroke in ink.strokes:
        stroke_x, stroke_y, stroke_t = stroke[0], stroke[1], stroke[2]

        colors = []

        prev_x, prev_y, _ = None, None, None

        for x, y, t in zip(stroke_x, stroke_y, stroke_t):
            if prev_x is None:
                prev_x = x

                prev_y = y

            # store img_drawing[(x, y)] = (r, g, b)

            # r, g, b range 0 - 1

            r = (t - ink.min_t) / (ink.max_t - ink.min_t)

            g = abs(x - prev_x) / ink.max_delta_x

            b = abs(y - prev_y) / ink.max_delta_y

            colors.append((r, g, b))

            prev_x = x

            prev_y = y

        for i in range(len(stroke_x)):
            ax.plot(
                stroke_x[i : i + 2],
                stroke_y[i : i + 2],
                linewidth=linewidth,
                color=colors[i],
            )

    ax.invert_yaxis()

    ax.axis("equal")

    plt.close()

    fig.canvas.draw()

    plt.tight_layout()

    # plt.show()

    # width, height = fig.canvas.get_width_height()

    img_array = np.array(fig.canvas.buffer_rgba())

    img_array = img_array[:, :, :3]

    height, width, _ = img_array.shape

    left_img_array = img_array[:, : (width // 2), :]

    right_img_array = img_array[:, (width // 2) :, :]

    # print(f'left_img_array shape: {left_img_array.shape}')

    # print(f'right_img_array shape: {right_img_array.shape}')

    img_array = np.concatenate((left_img_array, right_img_array), axis=0)

    # print(f'img_array shape: {img_array.shape}')

    return img_array
