
def hsl_to_rgb(h: float, s: float, l: float) -> (int, int, int):
    """
    Hue [0,360), S [0,1] V [0,1]

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
