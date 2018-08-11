import json
import sys

import numpy


def hsl_to_rgb(h: float, s: float, l: float) -> (int, int, int):
    """
    Hue [0,1), S [0,1] V [0,1]

    Reference https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSL
    """
    c = (1 - abs(2 * l - 1)) * s
    h = h * 6  # 360/60
    x = c * (1 - abs(h % 2 - 1))
    i = int(h)
    rgb = [0, 0, 0]
    rgb[(7 - i) % 3] = x
    rgb[int((i + 1) / 2) % 3] = c
    m = l - c / 2
    rgb = [x + m for x in rgb]
    rgb = [int(min(x * 256, 255)) for x in rgb]
    return tuple(rgb)

# MarcH @ https://stackoverflow.com/a/14981125/8408486
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def jdump(data):
    return json.dumps(data, cls=NumpyEncoder)

# mgilson @ https://stackoverflow.com/a/27050186
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.int):
            return int(obj)
        elif isinstance(obj, numpy.float):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

if __name__ == '__main__':
    eprint(jdump(numpy.ones([3, 2], dtype=numpy.int32)))
    pass