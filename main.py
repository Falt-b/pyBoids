import pygame
import numpy as np
from math import floor, ceil, sqrt
from random import randint
from sys import exit

WIDTH = 1800
HEIGHT = 1000
FPS = 60
BG_COLOR = (20, 20, 20)
COLOR_PALLETTE = (
    (255, 204, 102),
    (122, 255, 102),
    (102, 255, 230),
    (102, 133, 255),
    (194, 102, 255),
    (255, 102, 201),
    (255, 102, 102),
)
MAX_SPEED = 5
NUM_BOIDS = 250
MAX_FLOCK_SIZE = 10
EDGE_DISTANCE = 100
TURNING_FORCE = 0.25


class Hash_Map:
    def __init__(self, container_size: int, map_width: int) -> None:
        self.container_size = container_size
        self.width = ceil(map_width / container_size)
        self.objects = {}

    def hash_vector(self, position: pygame.Vector2):
        return (
            floor(position.x / self.container_size)
            + floor(position.y / self.container_size) * self.width
        )

    def get_rect_dimensions(self, rect: pygame.Rect):
        topleft = (
            floor(rect.topleft[0] / self.container_size),
            floor(rect.topleft[1] / self.container_size),
        )
        return (
            floor(rect.topright[0] / self.container_size) - topleft[0] + 1,  # x
            floor(rect.bottomleft[1] / self.container_size) - topleft[1] + 1,  # y
            topleft[0] + (topleft[1] * self.width),  # topleft position
        )

    def insert_object(self, object: object, hash: int):
        if hash in self.objects:
            self.objects[hash].append(object)
        else:
            self.objects.setdefault(hash, [object])

    def insert_rect(self, object: object):
        # assumes object has rect in class
        dimensions = self.get_rect_dimensions(object.rect)
        for y in range(dimensions[1]):
            for x in range(dimensions[0]):
                self.objects[dimensions[2] + x + (y * self.width)].append(object)

    def remove_object(self, object: object, hash: int):
        # assumes that hash is known and object is already in list
        self.objects[hash].remove(object)
        if len(self.objects[hash]) == 0:
            del self.objects[hash]

    def remove_rect(self, object: object):
        # assumes object has rect in class
        dimensions = self.get_rect_dimensions(object.rect)
        for y in range(dimensions[1]):
            for x in range(dimensions[0]):
                self.remove_object(
                    object, [dimensions[2] + x + (y * self.width)].remove(object)
                )

    def query(self, hash: int):
        if hash in self.objects:
            return self.objects[hash]
        return []

    def query_rect(self, rect: pygame.Rect):
        found_objects = []
        dimensions = self.get_rect_dimensions(rect)
        for y in range(dimensions[1]):
            for x in range(dimensions[0]):
                found_objects += self.query(dimensions[2] + x + (y * self.width))
        return found_objects


class Boid(pygame.sprite.Sprite):
    def __init__(
        self,
        position: pygame.Vector2,
        color: tuple,
        fill: int,
        map: Hash_Map,
        sight_range: int,
        *groups
    ) -> None:
        super().__init__(*groups)
        self.image = pygame.Surface((18, 10)).convert()
        self.image.fill((0, 0, 0))
        self.image.set_colorkey((0, 0, 0))
        pygame.draw.polygon(self.image, color, ((0, 0), (18, 5), (0, 10), (2, 5)), fill)
        self.original_image = self.image.copy().convert()
        self.rect = self.image.get_rect(center=position)
        self.sight_rect = pygame.Rect(
            sight_range / 2, sight_range / 2, sight_range, sight_range
        )
        self.hash = map.hash_vector(position)
        map.insert_object(self, self.hash)
        self.position = position
        self.vel = pygame.Vector2((randint(-3, 3), randint(-3, 3)))

    def update(self, map: Hash_Map, surface: pygame.Surface) -> None:
        alignment = pygame.Vector2()
        cohesion = pygame.Vector2()
        seperation = pygame.Vector2()
        found_boids = []
        for boid in map.query_rect(self.sight_rect):
            if (
                boid != self
                and self.sight_rect.collidepoint(boid.position.x, boid.position.y)
                and len(found_boids) < MAX_FLOCK_SIZE
            ):
                found_boids.append(
                    (
                        self.position.distance_squared_to(boid.position),
                        boid.position.x,
                        boid.position.y,
                    )
                )
                alignment += boid.vel
                cohesion += boid.position
        if len(found_boids) > 0:
            alignment = alignment / len(found_boids)
            alignment = alignment / (np.linalg.norm(alignment) + 0.1)
            cohesion = cohesion / len(found_boids) - self.position
            cohesion = ((cohesion / np.linalg.norm(cohesion)) - self.vel) * 0.05
            found_boids.sort(key=lambda item: item[0])
            seperation = (
                self.position
                - pygame.Vector2(found_boids[0][1], found_boids[0][2])
                - self.vel
            )
            seperation = (
                seperation
                / np.linalg.norm(seperation)
                * (40 - sqrt(found_boids[0][0]))
                * 0.02
            )  # 0.15
        self.vel += alignment + cohesion + seperation
        if self.position.x > WIDTH - EDGE_DISTANCE:
            self.vel.x -= TURNING_FORCE
        if self.position.x < EDGE_DISTANCE:
            self.vel.x += TURNING_FORCE
        if self.position.y > HEIGHT - EDGE_DISTANCE:
            self.vel.y -= TURNING_FORCE
        if self.position.y < EDGE_DISTANCE:
            self.vel.y += TURNING_FORCE
        norm_vel = np.linalg.norm(self.vel)
        if norm_vel > MAX_SPEED:
            self.vel = self.vel / norm_vel * MAX_SPEED
        self.position += self.vel
        new_hash = map.hash_vector(self.position)
        if new_hash != self.hash:
            map.remove_object(self, self.hash)
            map.insert_object(self, new_hash)
            self.hash = new_hash
        self.image = pygame.transform.rotate(
            self.original_image, pygame.Vector2().angle_to(self.vel) * -1
        )
        self.rect = self.image.get_rect(center=self.position)
        self.sight_rect.center = self.position


def generate_flock(
    max_pos: tuple,
    min_pos: tuple,
    color_pallette: list,
    fill: int,
    hash_map: Hash_Map,
    group: pygame.sprite.Group,
    num_boids: int,
):
    for i in range(num_boids):
        Boid(
            pygame.Vector2(
                (randint(min_pos[0], max_pos[0]), randint(min_pos[1], max_pos[1]))
            ),
            color_pallette[randint(0, len(color_pallette) - 1)],
            fill,
            hash_map,
            100,
            group,
        )


def main():
    pygame.init()
    display = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids")
    clock = pygame.time.Clock()

    boid_map = Hash_Map(60, WIDTH)
    boid_group = pygame.sprite.Group()
    generate_flock(
        (WIDTH, HEIGHT), (0, 0), COLOR_PALLETTE, 0, boid_map, boid_group, NUM_BOIDS
    )

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False

        display.fill(BG_COLOR)

        boid_group.draw(display)
        boid_group.update(boid_map, display)

        pygame.display.update()


if __name__ == "__main__":
    main()
    pygame.quit()
    exit()
