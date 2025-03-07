import argparse
import sys
#sys.path.appeng('../iGibsonChallenge2021/')

from gibson2.challenge.daw import Challenge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-class", type=str, default="SAC", choices=["Random", "ForwardOnly", "SAC"])
    parser.add_argument("--ckpt-path", default="", type=str)
    parser.add_argument('--cuda', action="store_false",
                        help='run on CUDA (default: True)')
    parser.add_argument("--device", default='cuda')

    # SAC
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--train_seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1500000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    
    
    parser.add_argument('--train_continue', type=bool, default=False)
    
    # Reward
    parser.add_argument('--goal_reward', type=int, default=20)
    parser.add_argument('--pedestrian_collision_reward', type=int, default=-20)

    parser.add_argument('--consider_all_waypoint', type=bool, default=False)

    parser.add_argument('--pca_reward', type=bool, default=False)
    parser.add_argument('--pca_reward_value', type=float, default=1)
    parser.add_argument('--pca_threshold', type=float, default=1)

    parser.add_argument('--waypoint_reach_threshold', type=float, default=0.1)


    # Pedestrian
    parser.add_argument('--pedestrian_present', type=bool, default=True)
    parser.add_argument('--num_pedestrians', type=int, default=7)
    parser.add_argument('--orca_radius', type=float, default=0.5)
    parser.add_argument('--pedestrian_collision_threshold', type=float, default=0.3)
    parser.add_argument('--pedestrian_goal_threshold', type=float, default=0.3)
    parser.add_argument('--replan_if_collision', type=bool, default=False)
    
    # Frontier based navigation
    parser.add_argument('--frontier_based_navigation', type=bool, default=False)
    parser.add_argument('--frontier_selection_method', type=str, default="closest")
    parser.add_argument('--replan_steps', type=int, default=10)

    # Observation
    parser.add_argument('--obs_goal', type=bool, default=False)
    parser.add_argument('--obs_waypoints', type=bool, default=False)
    parser.add_argument('--obs_map', type=bool, default=False)
    parser.add_argument('--obs_previous_action', type=bool, default=False)
    parser.add_argument('--obs_pedestrian_map', type=bool, default=False)
    parser.add_argument('--obs_map_lstm', type=bool, default=False)
    parser.add_argument('--obs_lidar', type=bool, default=False)
    parser.add_argument('--obs_replan', type=bool, default=False)
    parser.add_argument('--obs_pedestrian_pos', type=bool, default=False)

    parser.add_argument('--map_encoder', type=str, default="cnn")
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--num_wps_input', type=int, default=6)
    parser.add_argument('--obs_goal_input_size', type=int, default=2)

    # Env
    parser.add_argument('--env', type=str, default='simple')
    parser.add_argument('--env_type', type=str, default='with_map')
    parser.add_argument('--inflation_radius', type=float, default=2.5)
    parser.add_argument('--fov', type=float, default=42.5)
    parser.add_argument('--depth', type=float, default=50)
    parser.add_argument('--no_ep_after_print', type=int, default=30)   
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--map', type=str, default='cropped_map')
    parser.add_argument('--waypoint_interval', type=int, default=10)
    parser.add_argument('--robot_visible_to_pedestrians', type=bool, default=False) # ROBOT VISIBLE

    # Checkpoint
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/sac_checkpoint_DWA")
    parser.add_argument('--checkpoint_name', type=str, default="DWA1_newObsCost_collisionThresholdChanged_replanFixed_headingCostLastPoint_wpi10_sg1_og0.1_hg_0.4_noPed3_noTrajDiscard_6wp_robotNotVisible")

    parser.add_argument('--write_results', type=bool, default=True)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--eval_episodes_per_scene', type=int, default=100)
    

    
    args = parser.parse_args()

    challenge = Challenge()
    challenge.submit(None, args)

if __name__ == "__main__":
    main()
