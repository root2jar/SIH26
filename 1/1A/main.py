import pygame
import sys
import heapq
import math

pygame.init()

# ==================================================
# SCREEN
# ==================================================

WIDTH = 1100
HEIGHT = 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "UGV Vision-Based Autonomous Navigation"
)

# ==================================================
# COLORS
# ==================================================

GREEN = (80, 180, 80)
RED = (200, 60, 60)
BLUE = (50, 100, 220)
YELLOW = (240, 200, 50)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (120, 120, 120)

# ==================================================
# WORLD SETTINGS
# ==================================================

WORLD_WIDTH = 700
WORLD_HEIGHT = 600

# ==================================================
# UGV
# ==================================================

ugv = pygame.Rect(100, 300, 30, 30)

speed = 2

# ==================================================
# GOAL
# ==================================================

goal = pygame.Rect(600, 250, 30, 30)

# ==================================================
# REAL ENVIRONMENT
# ==================================================

real_obstacles = [

    pygame.Rect(250, 150, 50, 250),

    pygame.Rect(420, 300, 50, 200),

    pygame.Rect(560, 100, 50, 180)

]

# ==================================================
# CAMERA
# ==================================================

CAMERA_RANGE = 180

CAMERA_WIDTH = 350
CAMERA_HEIGHT = 300

# ==================================================
# GRID
# ==================================================

CELL_SIZE = 30

COLS = WORLD_WIDTH // CELL_SIZE
ROWS = WORLD_HEIGHT // CELL_SIZE


# ==================================================
# PIXEL → GRID
# ==================================================

def pixel_to_grid(x, y):

    return (
        x // CELL_SIZE,
        y // CELL_SIZE
    )


# ==================================================
# GRID → PIXEL
# ==================================================

def grid_to_pixel(cell):

    col, row = cell

    return (
        col * CELL_SIZE + CELL_SIZE // 2,
        row * CELL_SIZE + CELL_SIZE // 2
    )


# ==================================================
# SAFETY MARGIN
# ==================================================

SAFETY_MARGIN = 15


def get_inflated_obstacles():

    inflated = []

    for obstacle in real_obstacles:

        inflated.append(
            obstacle.inflate(
                SAFETY_MARGIN * 2,
                SAFETY_MARGIN * 2
            )
        )

    return inflated


# ==================================================
# CHECK BLOCKED CELL
# ==================================================

def is_blocked(cell):

    x, y = grid_to_pixel(cell)

    for obstacle in get_inflated_obstacles():

        if obstacle.collidepoint(x, y):

            return True

    return False


# ==================================================
# A* HEURISTIC
# ==================================================

def heuristic(a, b):

    return (
        abs(a[0] - b[0])
        + abs(a[1] - b[1])
    )


# ==================================================
# A* PATH PLANNER
# ==================================================

def a_star(start, goal_cell):

    open_set = []

    heapq.heappush(
        open_set,
        (0, start)
    )

    came_from = {}

    g_score = {
        start: 0
    }

    while open_set:

        current = heapq.heappop(
            open_set
        )[1]

        if current == goal_cell:

            path = []

            while current in came_from:

                path.append(current)

                current = came_from[current]

            path.append(start)

            path.reverse()

            return path

        neighbors = [

            (current[0] + 1, current[1]),
            (current[0] - 1, current[1]),
            (current[0], current[1] + 1),
            (current[0], current[1] - 1)

        ]

        for neighbor in neighbors:

            x, y = neighbor

            if x < 0 or x >= COLS:
                continue

            if y < 0 or y >= ROWS:
                continue

            if is_blocked(neighbor):
                continue

            new_cost = (
                g_score[current] + 1
            )

            if (
                neighbor not in g_score
                or new_cost < g_score[neighbor]
            ):

                came_from[neighbor] = current

                g_score[neighbor] = new_cost

                f_score = (
                    new_cost
                    + heuristic(
                        neighbor,
                        goal_cell
                    )
                )

                heapq.heappush(
                    open_set,
                    (f_score, neighbor)
                )

    return []


# ==================================================
# CAMERA RENDERING
# ==================================================

def draw_camera_view():

    # Camera background
    camera = pygame.Surface(
        (CAMERA_WIDTH, CAMERA_HEIGHT)
    )

    camera.fill(
        (100, 160, 100)
    )

    # Find obstacles visible from UGV
    visible_obstacles = []

    for obstacle in real_obstacles:

        distance = math.dist(
            ugv.center,
            obstacle.center
        )

        if distance < CAMERA_RANGE:

            visible_obstacles.append(
                obstacle
            )

    # Draw visible obstacles
    for obstacle in visible_obstacles:

        relative_x = (
            obstacle.centerx
            - ugv.centerx
        )

        relative_y = (
            obstacle.centery
            - ugv.centery
        )

        # Scale world coordinates
        scale = 1.2

        camera_x = (
            CAMERA_WIDTH // 2
            + int(relative_x * scale)
        )

        camera_y = (
            CAMERA_HEIGHT // 2
            + int(relative_y * scale)
        )

        rect = obstacle.copy()

        rect.center = (
            camera_x,
            camera_y
        )

        pygame.draw.rect(
            camera,
            RED,
            rect
        )

    # Camera center / UGV position
    pygame.draw.circle(
        camera,
        BLUE,
        (
            CAMERA_WIDTH // 2,
            CAMERA_HEIGHT // 2
        ),
        12
    )

    return camera


# ==================================================
# INITIAL PATH
# ==================================================

start_cell = pixel_to_grid(
    ugv.centerx,
    ugv.centery
)

goal_cell = pixel_to_grid(
    goal.centerx,
    goal.centery
)

path = a_star(
    start_cell,
    goal_cell
)

path_index = 1

clock = pygame.time.Clock()


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    # ------------------------------------------------
    # EVENTS
    # ------------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            pygame.quit()
            sys.exit()

    # ------------------------------------------------
    # RECALCULATE PATH
    # ------------------------------------------------

    current_cell = pixel_to_grid(
        ugv.centerx,
        ugv.centery
    )

    path = a_star(
        current_cell,
        goal_cell
    )

    path_index = 1

    # ------------------------------------------------
    # MOVE UGV
    # ------------------------------------------------

    if path_index < len(path):

        target_x, target_y = grid_to_pixel(
            path[path_index]
        )

        dx = target_x - ugv.centerx
        dy = target_y - ugv.centery

        if abs(dx) > speed:

            if dx > 0:
                ugv.x += speed
            else:
                ugv.x -= speed

        elif abs(dy) > speed:

            if dy > 0:
                ugv.y += speed
            else:
                ugv.y -= speed

    # ------------------------------------------------
    # DRAW WORLD
    # ------------------------------------------------

    screen.fill(GREEN)

    # World boundary
    pygame.draw.rect(
        screen,
        BLACK,
        (0, 0, WORLD_WIDTH, WORLD_HEIGHT),
        3
    )

    # Obstacles
    for obstacle in real_obstacles:

        pygame.draw.rect(
            screen,
            RED,
            obstacle
        )

    # Goal
    pygame.draw.rect(
        screen,
        YELLOW,
        goal
    )

    # Camera range
    pygame.draw.circle(
        screen,
        WHITE,
        ugv.center,
        CAMERA_RANGE,
        1
    )

    # Path
    if len(path) > 1:

        points = []

        for cell in path:

            points.append(
                grid_to_pixel(cell)
            )

        pygame.draw.lines(
            screen,
            WHITE,
            False,
            points,
            3
        )

    # UGV
    pygame.draw.rect(
        screen,
        BLUE,
        ugv
    )

    # ------------------------------------------------
    # CAMERA PANEL
    # ------------------------------------------------

    camera_view = draw_camera_view()

    screen.blit(
        camera_view,
        (
            WORLD_WIDTH + 25,
            100
        )
    )

    # Camera border
    pygame.draw.rect(
        screen,
        BLACK,
        (
            WORLD_WIDTH + 25,
            100,
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        ),
        3
    )

    # ------------------------------------------------
    # TEXT
    # ------------------------------------------------

    font = pygame.font.Font(
        None,
        28
    )

    world_text = font.render(
        "WORLD",
        True,
        WHITE
    )

    camera_text = font.render(
        "UGV CAMERA",
        True,
        WHITE
    )

    screen.blit(
        world_text,
        (20, 20)
    )

    screen.blit(
        camera_text,
        (WORLD_WIDTH + 25, 60)
    )

    # ------------------------------------------------
    # GOAL
    # ------------------------------------------------

    if ugv.colliderect(goal):

        text = font.render(
            "GOAL REACHED!",
            True,
            WHITE
        )

        screen.blit(
            text,
            (250, 40)
        )

    # ------------------------------------------------
    # UPDATE
    # ------------------------------------------------

    pygame.display.flip()

    clock.tick(60)