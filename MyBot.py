import bot_logic as bl
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MyBot')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--r_attack_docked', type=float, default=6)
    parser.add_argument('--r_chase_enemy', type=float, default=40)
    parser.add_argument('--enemy_weight_2p', type=float, default=12.5)
    parser.add_argument('--enemy_weight_4p', type=float, default=7.5)
    parser.add_argument('--r_enemy_no_dock', type=float, default=21)
    parser.add_argument('--unowned_planet_weight', type=float, default=2)
    parser.add_argument('--owned_planet_weight', type=float, default=6)
    parser.add_argument('--enemy_planet_weight', type=float, default=2)
    parser.add_argument('--expected_enemy_theta', type=float, default=0.875)

    args = parser.parse_args()

    game_controller = bl.GameController(debug=args.debug, args=args)

    while True:
        game_controller.update_map()
        command_queue = game_controller.compute_command_queue()
        game_controller.send_command_queue(command_queue)
