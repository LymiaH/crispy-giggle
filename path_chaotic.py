import random
from pathlib import Path

loop = []
random.seed(9786135)
for _ in range(0, 25):
    loop.append((random.randint(0, 500), random.randint(0, 500)))

(Path('paths') / 'chaotic.txt').write_text(str(loop))
print(str(loop))
