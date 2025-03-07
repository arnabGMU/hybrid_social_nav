import argparse
import sys
#sys.path.appeng('../iGibsonChallenge2021/')

from gibson2.challenge.test_SAC import Challenge


def get_agent(agent_class, ckpt_path=""):
    if agent_class == "SAC":
        return SAC()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-class", type=str, default="SAC", choices=["Random", "ForwardOnly", "SAC"])
    parser.add_argument("--ckpt-path", default="", type=str)
    parser.add_argument('--policy', default="Deterministic",
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
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=10000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_false",
                        help='run on CUDA (default: True)')
    parser.add_argument("--device", default='cuda')
    parser.add_argument('--obs_goal', type=bool, default=True)
    parser.add_argument('--obs_waypoints', type=bool, default=True)
    parser.add_argument('--obs_map', type=bool, default=True)
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--num_wps_input', type=int, default=6)

    parser.add_argument('--load_checkpoint', type=bool, default=True)
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/sac_checkpoint_deterministic_polar_potential0.1_orientation0.01_threshold0.1_final_fixed")
    parser.add_argument('--checkpoint_name', type=str, default="deterministic_polar_potential0.1_orientation0.01_threshold0.1_final_fixed_1")

    parser.add_argument('--load_checkpoint_memory', type=bool, default=False)
    parser.add_argument('--checkpoint_path_memory', type=str, default="checkpoints/sac_buffer_deterministic_polar_potential0.1_orientation0.01_threshold0.1_final_fixed_memory")
    parser.add_argument('--checkpoint_name_memory', type=str, default="deterministic_polar_potential0.1_orientation0.01_threshold0.1_final_fixed_memory_1")

    parser.add_argument('--train_continue', type=bool, default=True)
    parser.add_argument('--no_episode_trained', type=int, default=4150)

    parser.add_argument('--env', type=str, default='iGibson')
    args = parser.parse_args()

    challenge = Challenge()
    challenge.submit(None, args)

if __name__ == "__main__":
    main()
