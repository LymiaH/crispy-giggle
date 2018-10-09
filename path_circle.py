from pathlib import Path

import math

loop = []
ox = 100
oy = 100
radius = 100
for angle in range(0, 360):
    angleR = math.radians(angle)
    x = ox + radius * math.cos(angleR)
    y = oy + radius * math.sin(angleR)
    loop.append((int(x), int(y)))

(Path('paths') / 'circle.txt').write_text(str(loop))
print(str(loop))
