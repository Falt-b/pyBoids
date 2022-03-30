import pygame
import numpy as np
from random import randint

BOIDS = int(input("Number of Boids: "))


class OBJ:
    def __init__(self, pos, size) -> None:
        self.pos = np.array(pos, dtype=np.float64)
        self.size = size
        self.hSize = (self.size[0] / 2, self.size[1] / 2)

    def _contains(self, obj) -> bool:
        if (
            obj.pos[0] > self.pos[0]
            and obj.pos[0] < self.pos[0] + self.size[0]
            and obj.pos[1] > self.pos[1]
            and obj.pos[1] < self.pos[1] + self.size[1]
        ):
            return True
        return False


class QUADTREE:
    def __init__(self, pos, size, capacity) -> None:
        self.pos = pos
        self.size = size
        self.hSize = (self.size[0] / 2, self.size[1] / 2)
        self.capacity = capacity
        self.children = {}
        self.divided = False
        self.items = []

    def _contains(self, obj) -> bool:
        if (
            obj.pos[0] > self.pos[0]
            and obj.pos[0] < self.pos[0] + self.size[0]
            and obj.pos[1] > self.pos[1]
            and obj.pos[1] < self.pos[1] + self.size[1]
        ):
            return True
        return False

    def _intersects(self, other):
        if (
            self.pos[0] < other.pos[0] + other.size[0]
            and self.pos[0] + self.size[0] > other.pos[0]
            and self.pos[1] < other.pos[1] + other.size[1]
            and self.pos[1] + self.size[1] > other.pos[1]
        ):
            return True
        return False

    def _sub_divide(self) -> None:
        self.divided = True
        self.children["NW"] = QUADTREE(self.pos, self.hSize, self.capacity)
        self.children["NE"] = QUADTREE(
            (self.pos[0] + self.hSize[0], self.pos[1]), self.hSize, self.capacity
        )
        self.children["SW"] = QUADTREE(
            (self.pos[0], self.pos[1] + self.hSize[1]), self.hSize, self.capacity
        )
        self.children["SE"] = QUADTREE(
            (self.pos[0] + self.hSize[0], self.pos[1] + self.hSize[1]),
            self.hSize,
            self.capacity,
        )

    def _insert(self, obj) -> bool:
        if self._contains(obj):
            if len(self.items) < self.capacity:
                self.items.append(obj)
                return True
            if not self.divided:
                self._sub_divide()
            return (
                self.children["NW"]._insert(obj)
                or self.children["NE"]._insert(obj)
                or self.children["SW"]._insert(obj)
                or self.children["SE"]._insert(obj)
            )
        return False

    def _query(self, other):
        found = []
        if self._intersects(other):
            found = [p for i, p in enumerate(self.items) if other._contains(p)]
            if self.divided:
                return (
                    found
                    + self.children["NW"]._query(other)
                    + self.children["NE"]._query(other)
                    + self.children["SW"]._query(other)
                    + self.children["SE"]._query(other)
                )
        return found


class BOID(OBJ):
    def __init__(self, pos, size, color=(0, 0, 0)) -> None:
        super().__init__(pos, size)
        self.vel = np.array([randint(-5, 5), randint(-5, 5)], dtype=np.float64)
        self.accel = np.array(
            [randint(-5, 5) * 0.1, randint(-5, 5) * 0.1], dtype=np.float64
        )
        self.points = [
            (self.pos[0] - self.hSize[0], self.pos[1] + self.hSize[1]),
            (self.pos[0], self.pos[1] - self.hSize[1]),
            (self.pos[0] + self.hSize[0], self.pos[1] + self.hSize[1]),
            (self.pos[0], self.pos[1] + self.hSize[1] - 5),
        ]
        self.angle = 0
        self.max = randint(7, 9)
        self.color = color

    def update(self, flock, dist, maxX, maxY, turningForce=1):
        f = len(flock)
        if f > 1:
            b = np.array(self.get_flock(flock), object)
            d = np.argsort(b[:, 0])[:1]
            center = np.divide(np.sum(b[:, 1]), f) - self.pos - self.vel
            avoidance = self.pos - b[d[0], 1] - self.vel
            self.accel += (
                np.multiply(
                    np.multiply(
                        np.divide(
                            avoidance,
                            np.sqrt(np.einsum("...i, ...i", avoidance, avoidance)),
                        ),
                        60 - np.sqrt(b[d[0], 0]),
                    ),
                    0.03,
                )
                + np.multiply(
                    np.divide(center, np.sqrt(np.einsum("...i, ...i", center, center))),
                    0.06,
                )
                + np.multiply(np.divide(np.sum(b[:, 2]), f), 0.4)
            )
        if 0 + self.pos[0] < dist:
            self.vel[0] += turningForce
        if maxX - self.pos[0] < dist:
            self.vel[0] -= turningForce
        if 0 + self.pos[1] < dist:
            self.vel[1] += turningForce
        if maxY - self.pos[1] < dist:
            self.vel[1] -= turningForce
        self.vel += self.accel
        self.pos += self.vel
        self.points = [
            (self.pos[0] - self.hSize[0], self.pos[1] + self.hSize[1]),
            (self.pos[0], self.pos[1] - self.hSize[1]),
            (self.pos[0] + self.hSize[0], self.pos[1] + self.hSize[1]),
        ]
        self.angle = pygame.Vector2(0, 0).angle_to(self.vel) + 90
        n = np.sqrt(np.einsum("...i, ...i", self.vel, self.vel))
        if n > 5:
            self.vel = np.multiply(np.divide(self.vel, n), 5)
        self.accel = np.array([0, 0], np.float16)

    def get_flock(self, flock):
        elements = []
        for boid in flock:
            d = (self.pos[0] - boid.pos[0]) ** 2 + (self.pos[1] - boid.pos[1]) ** 2
            if boid != self and d < 14400:
                elements.append((d, boid.pos, boid.vel))
        return elements

    def rotate(self, points, angle):
        rads = np.deg2rad(angle)
        c = np.cos(rads)
        s = np.sin(rads)
        rp = []
        for p in points:
            rp.append(
                (
                    np.trunc(
                        np.multiply(c, (p[0] - self.pos[0]))
                        - np.multiply(s, (p[1] - self.pos[1]))
                        + self.pos[0]
                    ),
                    np.trunc(
                        np.multiply(s, (p[0] - self.pos[0]))
                        + np.multiply(c, (p[1] - self.pos[1]))
                        + self.pos[1]
                    ),
                )
            )
        return rp

    def draw(self, surface):
        pygame.draw.polygon(
            surface, self.color, self.rotate(self.points, self.angle), width=0
        )


def main():
    w, h = 1920, 960
    FPS = 30
    bgColor = (20, 20, 20)
    pygame.font.init()
    pallette = [
        (174, 167, 209),
        (225, 194, 217),
        (251, 249, 226),
        (215, 235, 209),
        (177, 216, 207),
    ]
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Boids Basic")
    clock = pygame.time.Clock()
    last_update = pygame.time.get_ticks()
    Q = QUADTREE((0, 0), (w, h), 6)
    t = []
    for i in range(BOIDS):
        b = BOID(
            (randint(0, w), randint(0, h)),
            (15, 25),
            pallette[randint(0, len(pallette) - 1)],
        )
        t.append(b)
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False
        screen.fill(bgColor)
        current = pygame.time.get_ticks()
        if current - last_update >= 5:
            Q = QUADTREE((0, 0), (w, h), 6)
            for boid in t:
                Q._insert(boid)
        for boid in t:
            S = OBJ((boid.pos[0] - 63, boid.pos[1] - 63), (123, 123))
            boid.update([b for i, b in enumerate(t) if S._contains(b)][:10], 50, w, h)
            boid.draw(screen)
        pygame.display.flip()


if __name__ == "__main__":
    main()
