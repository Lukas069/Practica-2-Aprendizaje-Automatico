import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from collections import deque
import pygame


class SnakeEnv(gym.Env):
    """
    Entorno de Gymnasium personalizado del juego del Snake
    con obstáculos que se mueven dinámicamente y comida mala.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    DIRECTIONS = [
        (-1,  0), # Arriba
        ( 0,  1), # Derecha
        ( 1,  0), # Abajo
        ( 0, -1)] # Izquierda

    def __init__(self, grid_size=10, obstacle_move_freq=5,
                 max_steps=500, render_mode=None, cell_size=50):
        super().__init__()

        self.grid_size = grid_size # Tamaño de la cuadricula
        self.obstacle_move_freq = obstacle_move_freq # Frecuencia de movimiento de los obstáculos
        self.max_steps = max_steps # Maximo de pasos por episodio
        self.render_mode = render_mode # Modo de renderizado (None, "human", "rgb_array")
        self.cell_size = cell_size # Tamaño de cada celda al mostrar el juego
        self.n_obstacles = 3 # Número de obstáculos en el juego

        # Parámetro para parar el renderizado cuando el usuario cierra la ventana o pulsa ESC
        self.running = True

        # Espacio de acciones (0: izquierda, 1: recto, 2: derecha)
        self.action_space = spaces.Discrete(3) 

        # Espacio de de estados (vector de 14 elementos con valores entre 0 y 1)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)

        # Estado interno del juego
        self.snake = None
        self.direction = None
        self.good_food = None
        self.bad_food = None
        self.obstacles = None
        self.steps = 0

        # Recursos de Pygame
        self.window = None
        self.clock  = None


    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno a su estado inicial. La serpiente comienza
        con un tamaño de 3 celdas en el centro de la cuadrícula mirando
        hacia la derecha. La comida y obstáculos se colocan aleatoriamente.
        """

        super().reset(seed=seed)

        mid = self.grid_size // 2

        # Serpiente inicial
        self.snake = deque([
            (mid, mid),
            (mid, mid - 1),
            (mid, mid - 2)])
        
        self.direction = 1
        self.steps = 0

        # Comida y obstáculos iniciales
        self._place_food()
        self._place_obstacles()

        return self._get_observation(), {}


    def step(self, action):
        """
        Recibe una acción y actualiza el estado del juego. Devuelve la nueva
        observación, la recompensa, si el episodio terminó o fue truncado.
        """

        self.steps += 1

        # Actualiza la dirección si el agente decide girar hacia un lado
        if action == 0: # Girar a la izquierda
            self.direction = (self.direction - 1) % 4
        elif action == 2: # Girar a la derecha
            self.direction = (self.direction + 1) % 4

        # Calcula la nueva posición de la cabeza
        drow, dcol = self.DIRECTIONS[self.direction]
        hrow, hcol = self.snake[0]
        new_head = (hrow + drow, hcol + dcol)

        # Recompensa negativa por paso
        reward = -0.01
        terminated = False
        info = {"cause": None}

        # Comprueba si la serpiente se ha chocado con una pared o su cuerpo
        if self._is_collision(new_head):
            reward = -20.0
            terminated = True
            info["cause"] = "collision"
            return self._get_observation(), reward, terminated, False, info

        # Comprueba si la serpiente se ha chocado con un obstáculo
        if new_head in self.obstacles:
            reward = -20.0
            terminated = True
            info["cause"] = "obstacle"
            return self._get_observation(), reward, terminated, False, info

        # Añade la nueva cabeza a la serpiente
        self.snake.appendleft(new_head)

        # Si come comida buena le damos una recompensa positiva y colocamos nueva comida
        if new_head == self.good_food:
            reward += 10.0
            self._place_food()

        # Si come comida mala se acaba episodio y le damos una recompensa negativa
        elif new_head == self.bad_food:
            reward = -20.0
            terminated = True
            info["cause"] = "bad_food"
            return self._get_observation(), reward, terminated, False, info

        # Si no come comida eliminamos la cola de la serpiente
        else:
            self.snake.pop()
            reward += self._distance_reward(new_head) # Recompensa por acercarse a comida buena

        # Mueve los obstáculos periodicamente
        if self.steps % self.obstacle_move_freq == 0:
            self._move_obstacles()

        # Comprueba si se ha superado el limite de pasos por episodio
        truncated = self.steps >= self.max_steps

        return self._get_observation(), reward, terminated, truncated, info


    def render(self):
        """
        Visualiza el estado actual del juego con Pygame. Muestra la serpiente,
        la comida, los obstáculos y un panel inferior con métricas.
        """

        if self.render_mode != "human":
            return

        # Inicializa Pygame la primera vez que se llama a render()
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Snake — ESC para salir")
            w = self.grid_size * self.cell_size
            self.window = pygame.display.set_mode((w, w + 60))
            self.clock  = pygame.time.Clock()

        # Paleta de colores
        BLACK = (15, 15, 20)
        DK_GRID = (30, 30, 40)
        GREEN = (80, 200, 100)
        RED = (220, 60, 60)
        YELLOW = (240, 200, 50)
        GRAY = (100, 100, 120)
        WHITE = (220, 220, 230)
        PANEL_BG = (25, 25, 35)

        cs = self.cell_size

        # Dibuja el fondo negro
        self.window.fill(BLACK)

        # Dibuja las líneas de la cuadrícula con gris oscuro
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.window, DK_GRID,
                            (i * cs, 0), (i * cs, self.grid_size * cs))
            pygame.draw.line(self.window, DK_GRID,
                            (0, i * cs), (self.grid_size * cs, i * cs))

        # Dibuja los obstáculos (rectángulos grises redondeados)
        for (r, c) in self.obstacles:
            rect = pygame.Rect(c*cs + 2, r*cs + 2, cs - 4, cs - 4)
            pygame.draw.rect(self.window, GRAY, rect, border_radius=4)

        # Dibuja la comida mala (cuadrado rojo)
        if self.bad_food:
            r, c = self.bad_food
            rect = pygame.Rect(c*cs + 4, r*cs + 4, cs - 8, cs - 8)
            pygame.draw.rect(self.window, RED, rect, border_radius=cs // 4)

        # Dibuja la comida buena (círculo amarillo)
        if self.good_food:
            r, c = self.good_food
            cx, cy = c*cs + cs // 2, r*cs + cs // 2
            pygame.draw.circle(self.window, YELLOW, (cx, cy), cs // 2 - 4)

        # Dibuja el cuerpo de la serpiente (de verde a oscuro)
        for i, (r, c) in enumerate(list(self.snake)[1:], 1):
            shade = max(30, 200 - i * 8)
            rect = pygame.Rect(c*cs + 3, r*cs + 3, cs - 6, cs - 6)
            pygame.draw.rect(self.window, (0, shade, 50), rect, border_radius=5)

        # Dibuja la cabeza (verde brillante)
        hrow, hcol = self.snake[0]
        head_rect = pygame.Rect(hcol*cs + 1, hrow*cs + 1, cs - 2, cs - 2)
        pygame.draw.rect(self.window, GREEN, head_rect, border_radius=8)

        # Dibuja un panel inferior con métricas
        panel_y = self.grid_size * cs
        pygame.draw.rect(self.window, PANEL_BG,
                        (0, panel_y, self.grid_size * cs, 60))
        font = pygame.font.SysFont("monospace", 17)
        txt  = font.render(
            f"Longitud: {len(self.snake)}   "
            f"Pasos: {self.steps}   "
            f"Obstáculos: {len(self.obstacles)}   ",
            True, WHITE)
        self.window.blit(txt, (10, panel_y + 20))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        # Apaga la visualización si el usuario cierra la ventana o pulsa ESC
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return


    def close(self):
        """
        Apaga la ventana de Pygame y libera los recursos.
        """

        self.running = False
        if self.window is not None:
            pygame.quit()
            self.window = None


    def _get_observation(self):
        """
        Devuelve un array con el espacio de estados actual:

        [0]: Indica si hay peligro delante (pared, cuerpo u obstáculo)
        [1]: Indica si hay peligro a la izquierda
        [2]: Indica si hay peligro a la derecha
        [3-6]: Indica la dirección actual (arriba/derecha/abajo/izquierda)
        [7]: Indica si hay comida buena arriba
        [8]: Indica si hay comida buena a la derecha
        [9]: Indica si hay comida buena abajo
        [10]: Indica si hay comida buena a la izquierda
        [11]: Indica si hay comida mala justo delante
        [12]: Indica si hay un obstáculo justo delante
        [13]: Indica la longitud de la serpiente normalizada
        """

        head = self.snake[0]
        dir_idx = self.direction

        # Vectores de dirección relativos a la serpiente
        front = self.DIRECTIONS[dir_idx]
        left  = self.DIRECTIONS[(dir_idx - 1) % 4]
        right = self.DIRECTIONS[(dir_idx + 1) % 4]

        def next_cell(pos, dir):
            return (pos[0] + dir[0], pos[1] + dir[1])

        # [0-2] Peligro en las tres direcciones relativas
        danger_front = float(self._is_collision(next_cell(head, front)) or
                             next_cell(head, front) in self.obstacles)
        danger_left = float(self._is_collision(next_cell(head, left)) or
                            next_cell(head, left) in self.obstacles)
        danger_right = float(self._is_collision(next_cell(head, right)) or
                             next_cell(head, right) in self.obstacles)

        # [3-6] Dirección actual en codificación one-hot
        dir_one_hot = [float(dir_idx == i) for i in range(4)]

        # [7-10] Posición relativa de la comida buena
        gf_row, gf_col = self.good_food
        hrow, hcol = head
        food_up = float(gf_row < hrow)
        food_right = float(gf_col > hcol)
        food_down = float(gf_row > hrow)
        food_left = float(gf_col < hcol)

        # [11] Comida mala en la celda justo delante
        bad_front = float(next_cell(head, front) == self.bad_food)

        # [12] Obstáculo en la celda justo delante
        obs_front = float(next_cell(head, front) in self.obstacles)

        # [13] Longitud de la serpiente normalizada al tamaño de la cuadrícula
        length_norm = len(self.snake) / (self.grid_size ** 2)

        return np.array([
            danger_front, danger_left, danger_right,
            *dir_one_hot,
            food_up, food_right, food_down, food_left,
            bad_front, obs_front,
            length_norm], dtype=np.float32)


    def _is_collision(self, pos):
        """
        Devuelve un booleano si la serpiente se ha chocado
        con una pared o su cuerpo.
        """

        row, col = pos
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return True # Choque con pared
        return pos in self.snake # Choque con el cuerpo


    def _distance_reward(self, head):
        """
        Devuelve una recompensa positiva o negativa si la serpiente
        se acerca o se aleja de la comida buena.
        """

        # Calcula la distancia a la comida en la posición actual
        gf_row, gf_col = self.good_food
        hrow,  hcol  = head
        dist_actual = abs(gf_row - hrow) + abs(gf_col - hcol)

        # Calcula la distancia a la comida en la posición anterior
        prev_row, prev_col = self.snake[1] if len(self.snake) > 1 else head
        dist_previa = abs(gf_row - prev_row) + abs(gf_col - prev_col)

        # Recompensa positiva si se acerca a la comida, negativa si se aleja, neutra si no cambia
        if dist_actual < dist_previa:
            return  0.1
        if dist_actual > dist_previa:
            return -0.1
        return 0.0


    def _free_cells(self):
        """
        Devuelve una lista con todas las celdas que no están ocupadas
        por ningún elemento.
        """

        # Serpiente
        occupied = set(self.snake)

        # Comida buena
        if self.good_food:
            occupied.add(self.good_food)

        # Comida mala
        if self.bad_food:
            occupied.add(self.bad_food)

        # Obstáculos
        if self.obstacles:
            occupied.update(self.obstacles)
        
        all_cells = {(row, col)
            for row in range(self.grid_size)
            for col in range(self.grid_size)}
        return list(all_cells - occupied)


    def _place_food(self):
        """
        Coloca comida buena y comida mala en celdas libres aleatoriamente.
        """

        free = self._free_cells()

        # Si hay al menos dos celdas libres, colocamos comida buena y mala
        if len(free) >= 2:
            pos = random.sample(free, 2)
            self.good_food = pos[0]
            self.bad_food  = pos[1]

        # Si únicamente queda una celda libre, colocamos solo comida buena
        elif len(free) == 1:
            self.good_food = free[0]
            self.bad_food  = None


    def _place_obstacles(self):
        """
        Coloca los obstáculos en celdas libres aleatoriamente al inicio
        del episodio.
        """

        free = self._free_cells()
        n = min(self.n_obstacles, len(free))
        self.obstacles = set(random.sample(free, n))

    def _move_obstacles(self):
        """
        Mueve cada obstáculo a una celda adyacente libre aleatoriamente.
        """

        new_obstacles = set()
        for obs in self.obstacles:
            candidates = []
            for drow, dcol in self.DIRECTIONS:
                nrow, ncol = obs[0] + drow, obs[1] + dcol
                candidate = (nrow, ncol)
                if (0 <= nrow < self.grid_size and 0 <= ncol < self.grid_size and
                    candidate not in self.snake and candidate != self.good_food and
                    candidate != self.bad_food and candidate not in new_obstacles):
                    candidates.append(candidate)
            new_obstacles.add(random.choice(candidates) if candidates else obs)
        self.obstacles = new_obstacles
