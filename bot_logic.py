import hlt
import numpy as np
import functools
import pickle_wrap as pw
import time

from hlt.entity import Entity
from hlt.entity import Position
from hlt.entity import Ship


def first(tup):
    return tup[0]


def second(tup):
    return tup[1]


def compute_planets_mask_and_xx_yy(game_map: hlt.game_map, no_go_radius):
    yy, xx = np.mgrid[:game_map.height, :game_map.width]
    planets_masks = [(xx - planet.x) ** 2 + (yy - planet.y) ** 2 < (planet.radius + no_go_radius) ** 2
                     for planet in game_map.all_planets()]
    planets_mask = functools.reduce(np.logical_or, planets_masks)
    return planets_mask, xx, yy


def compute_distances(game_map: hlt.game_map, target_x, target_y, target_radius, no_go_radius=.1):
    distances = np.ones((game_map.height, game_map.width)) * np.infty
    no_go_mask, xx, yy = compute_planets_mask_and_xx_yy(game_map, no_go_radius)
    target = (xx - target_x) ** 2 + (yy - target_y) ** 2 < target_radius ** 2
    distances[target] = 0

    prev_updated = target

    s2 = np.sqrt(2)

    def jumps(inc_diagonal):

        yield 0, 1, 1
        yield 0, -1, 1
        yield 1, 0, 1
        yield -1, 0, 1

        if inc_diagonal:
            yield 1, 1, s2
            yield 1, -1, s2
            yield -1, 1, s2
            yield -1, -1, s2

    jumps_all = list(jumps(True))
    jumps_less = list(jumps(False))

    count = 0
    while prev_updated[prev_updated].size > 0:
        count += 1
        updated = np.zeros((game_map.height, game_map.width)).astype(bool)
        for (y0, x0) in zip(*np.where(prev_updated)):
            dist0 = distances[y0, x0]
            for dx, dy, dr in jumps_all if count < 150 else jumps_less:
                x1, y1, dist1 = x0 + dx, y0 + dy, dist0 + dr
                if 0 <= x1 < game_map.width and 0 <= y1 < game_map.height:
                    if not no_go_mask[y1, x1] and distances[y1, x1] > dist1:
                        distances[y1, x1] = dist1
                        updated[y1, x1] = True
        prev_updated = updated

    distances[no_go_mask] += game_map.width
    return distances


def calc_r2(dx0, dy0, dx1, dy1, theta):
    dx = dx1 * theta + (1 - theta) * dx0
    dy = dy1 * theta + (1 - theta) * dy0
    return dx ** 2 + dy ** 2


def min_r2(a0, a1, b0, b1):
    dx0 = float(a0.x - b0.x)
    dx1 = float(a1.x - b1.x)
    dy0 = float(a0.y - b0.y)
    dy1 = float(a1.y - b1.y)

    denominator = dx0 ** 2 - 2 * dx0 * dx1 + dx1 ** 2 + dy0 ** 2 - 2 * dy0 * dy1 + dy1 ** 2

    if denominator != 0:

        numerator = dx0 ** 2 - dx0 * dx1 + dy0 ** 2 - dy0 * dy1
        theta0 = numerator / denominator

        if 0 <= theta0 <= 1:
            return calc_r2(dx0, dy0, dx1, dy1, theta0)

    return min([calc_r2(dx0, dy0, dx1, dy1, 0), calc_r2(dx0, dy0, dx1, dy1, 1)])


class GameController(object):

    def __init__(self, load_from_disk=None, debug=False, args=None):
        self._t0 = time.time()
        self._game = pw.load('game_init.p') if load_from_disk else hlt.Game('DaanV123' + ('_debug' if debug else ''))
        self._debug = debug
        self._turn_idx = 0
        if self._debug:
            pw.dump(self._game, "game_init.p")
            pw.dump(args, "args.p")

        if load_from_disk:
            args = pw.load("args.p")
            print(args)

        self._num_players = len(self._game.map.all_players())
        heads_up = self._num_players == 2
        self._r_attack_docked = args.r_attack_docked
        self._r2_chase_enemy = args.r_chase_enemy ** 2
        self._enemy_weight = args.enemy_weight_2p if heads_up else args.enemy_weight_4p
        self._unowned_planet_weight = args.unowned_planet_weight
        self._r_enemy_no_dock = args.r_enemy_no_dock
        self._owned_planet_weight = args.owned_planet_weight
        self._enemy_planet_weight = args.enemy_planet_weight
        self._expected_enemy_theta = args.expected_enemy_theta

        self._distances_to_planets = {planet.id: compute_distances(self._game.map, planet.x, planet.y, planet.radius + 3)
                                      for planet in self._game.map.all_planets()}

        center = Position(self._game.map.width / 2, self._game.map.height / 2)
        self._planet_distance_from_center = {planet.id: planet.calculate_distance_sq_between(center) for planet in self._game.map.all_planets()}

        self._should_i_dock = True
        self._should_i_undock = False
        self._hide = False
        self.compute_stuff()

    def update_map(self, game_id_to_load_from_disk=None):
        self._t0 = time.time()
        if game_id_to_load_from_disk:
            self._game = pw.load('game_{}.p'.format(game_id_to_load_from_disk))
            self._turn_idx = game_id_to_load_from_disk
        else:
            self._game.update_map()

        if self._debug:
            pw.dump(self._game, 'game_{}.p'.format(self._turn_idx))
            pw.dump(self._turn_idx + 1, 'num_turns.p')
        self._turn_idx += 1

        self.compute_stuff()

    def compute_stuff(self):
        self._num_players = len([0 for player in self._game.map.all_players() if len(player.all_ships()) > 0])

        num_ships = sorted([(player.id, len(player.all_ships())) for player in self._game.map.all_players() if len(player.all_ships()) > 0], key=second)
        # min_ships_player_id, min_num_ships = num_ships[0]
        _, snd_num_ships = num_ships[1]
        # max_ships_player_id, max_num_ships = num_ships[-1]
        my_num_ships = len(self._game.map.get_me().all_ships())

        if self._turn_idx > 20 and self._num_players > 2 and my_num_ships * 1.5 < snd_num_ships:
            self._should_i_dock = False
            self._should_i_undock = True
            self._hide = True

    def has_planet(self, x, y):
        if x <= 0 or x >= self._game.map.width or y <= 0 or y >= self._game.map.height:
            return True
        pos = Position(x, y)
        for planet in self._game.map.all_planets():
            if pos.calculate_distance_between(planet) <= planet.radius + .50001:
                return True

        return False

    def distance_to_planet(self, planet_id, x, y):
        return 0 <= int(x) < self._game.map.width and 0 <= int(y) < self._game.map.height and self._distances_to_planets[planet_id][int(y), int(x)]

    def compute_utility(self, x, y, ship, enemy_ships):
        planet_utilities, docked_ship_utilities, enemy_ship_utility = [0], [0], 0
        pos = Position(x, y)

        if self._hide:
            hiding_positions = [Position(3, 3),
                                Position(self._game.map.width-3, 3),
                                Position(3, self._game.map.height-3),
                                Position(self._game.map.width-3, self._game.map.height-3)]
            utility = 0

            for hiding_position in hiding_positions:
                utility += 1/pos.calculate_distance_sq_between(hiding_position)

            return utility

        for planet in self._game.map.all_planets():

            r = self.distance_to_planet(planet.id, x, y)
            if r == np.infty:
                continue

            r += 1

            planet_weight = 1 if self._num_players < 3 else self._planet_distance_from_center[planet.id]

            if not planet.is_owned() and self._should_i_dock:
                free_docking_spots = planet.num_docking_spots - planet.num_docked_ships()
                planet_utilities.append(planet_weight * free_docking_spots * self._unowned_planet_weight / r ** 2)
            elif planet.owner == self._game.map.get_me() and not planet.is_full() and self._should_i_dock:
                planet_utilities.append(planet_weight * self._owned_planet_weight / r ** 2)
            elif planet.is_owned() and planet.owner != self._game.map.get_me():
                planet_utilities.append(planet_weight * self._enemy_planet_weight / r ** 2)

                if r < self._r_attack_docked:
                    angle_pos = planet.calculate_angle_between(pos)

                    docked_ship_utility = 0
                    for docked_ship in planet.all_docked_ships():
                        angle_ship = planet.calculate_angle_between(docked_ship)
                        angle_diff = (angle_ship - angle_pos) % 360
                        angle_diff = min(angle_diff, 360 - angle_diff)
                        docked_ship_utility += 180 - angle_diff

                    docked_ship_utilities.append(docked_ship_utility)

        for enemy_ship in enemy_ships:
            enemy_ship_previous = self._game.map.get_player_previous(enemy_ship.owner.id).get_ship(enemy_ship.id)
            enemy_ship_previous = enemy_ship_previous if enemy_ship_previous else enemy_ship

            x, y = enemy_ship.x, enemy_ship.y
            dx = enemy_ship.x - enemy_ship_previous.x
            dy = enemy_ship.y - enemy_ship_previous.y
            theta = self._expected_enemy_theta
            expected_next_pos = Position(x + theta * dx, y + theta * dy)

            r2 = pos.calculate_distance_sq_between(expected_next_pos)
            if r2 < self._r2_chase_enemy:
                enemy_ship_utility += self._enemy_weight / r2

        return np.sum(planet_utilities) + max(docked_ship_utilities) + enemy_ship_utility

    def compute_potential_new_positions_and_utilities(self, ship, enemy_ships):
        positions_all = [(ship.x, ship.y, 0, 0)]
        positions_all += [(ship.x + speed * np.cos(angle / 360. * 2 * np.pi),
                           ship.y + speed * np.sin(angle / 360. * 2 * np.pi),
                           speed,
                           angle)
                          for angle in range(0, 360, 15)
                          for speed in [1, 2, 3, 6, 7]]

        positions_no_crash_into_planet = []

        for new_x, new_y, speed, angle in positions_all:
            if not self.has_planet(new_x, new_y):
                positions_no_crash_into_planet.append((new_x, new_y, speed, angle))

        return [(-self.compute_utility(new_x, new_y, ship, enemy_ships), new_x, new_y, speed, angle, ship)
                for new_x, new_y, speed, angle in positions_no_crash_into_planet]

    def time_left(self):
        return 2.0 - (time.time() - self._t0)

    def no_enemy_close(self, ship, enemy_ships):

        r_enemy_no_dock = self._r_enemy_no_dock if self._turn_idx > 15 else 80

        for enemy_ship in enemy_ships:
            if enemy_ship.docking_status == Ship.DockingStatus.UNDOCKED and ship.calculate_distance_between(enemy_ship) < r_enemy_no_dock:
                return False

        return True

    def no_crash_into_planet(self, source: Entity, target: Entity):

        for planet in self._game.map.all_planets():

            collision_possible = abs(source.x - planet.x) < planet.radius + 8 and abs(source.y - planet.y) < planet.radius + 8
            if collision_possible and min_r2(source, target, planet, planet) < (.5 + 0.001 + planet.radius) ** 2:
                return False

        return True

    def compute_command_queue(self):

        my_ships = [ship for ship in self._game.map.get_me().all_ships()]
        my_docked_ships = [ship for ship in my_ships if ship.docking_status in (Ship.DockingStatus.DOCKED,
                                                                                Ship.DockingStatus.DOCKING,
                                                                                Ship.DockingStatus.UNDOCKING)]
        my_undocked_ships = [ship for ship in my_ships if ship.docking_status == Ship.DockingStatus.UNDOCKED]
        enemy_ships = [ship for player in self._game.map.all_players() if not player == self._game.map.get_me() for ship in player.all_ships()]

        moves = []

        def no_collision(source: Entity, target: Entity):

            for s, t in moves:

                collision_possible = abs(source.x - s.x) < 15 and abs(source.y - s.y) < 15
                if collision_possible and min_r2(source, target, s, t) < 1.001:
                    return False

            return True

        for ship in my_docked_ships:
            moves.append((ship, ship))

        potential_new_positions_and_utilities = []

        free_docking_spots = {}
        for planet in self._game.map.all_planets():
            if not planet.is_owned() or planet.owner == self._game.map.get_me():
                free_docking_spots[planet.id] = planet.num_docking_spots - planet.num_docked_ships()
            else:
                free_docking_spots[planet.id] = 0

        for ship in my_undocked_ships:

            time_left = self.time_left()
            if time_left < 0.3:
                break

            potential_new_positions_and_utilities += self.compute_potential_new_positions_and_utilities(ship, enemy_ships)

        command_queue = []
        ships_moved = set()

        if self._should_i_undock:
            for ship in my_docked_ships:
                if ship.docking_status == Ship.DockingStatus.DOCKED:
                    command_queue.append(ship.undock())

        for utility, new_x, new_y, speed, angle, ship in sorted(potential_new_positions_and_utilities, key=first):

            time_left = self.time_left()
            if time_left < 0.2:
                return command_queue

            if ship not in ships_moved:
                for planet in self._game.map.all_planets():
                    if self._should_i_dock and free_docking_spots[planet.id] > 0 and ship.can_dock(planet) and no_collision(ship, ship) and self.no_enemy_close(ship, enemy_ships):
                        free_docking_spots[planet.id] -= 1
                        command_queue.append(ship.dock(planet))
                        ships_moved.add(ship)
                        moves.append((ship, ship))
                        # print('Ship {}: I''m docking'.format(ship.id))
                        break
                else:
                    new_position = Position(new_x, new_y)

                    if no_collision(ship, new_position) and self.no_crash_into_planet(ship, new_position):
                        command_queue.append(ship.thrust(speed, angle))
                        ships_moved.add(ship)
                        moves.append((ship, new_position))
                        # print('Ship {}: I''m moving to {} {} ({})'.format(ship.id, new_x, new_y, utility))

        return command_queue

    def send_command_queue(self, command_queue):
        self._game.send_command_queue(command_queue)
