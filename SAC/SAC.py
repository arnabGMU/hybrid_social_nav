import os
import torch
import torch.nn.functional as F
import copy

from torch.optim import Adam
from SAC.utils.utils import soft_update, hard_update, soft_update_obs_train
from SAC.model import GaussianPolicy, QNetwork, DeterministicPolicy
from encoder.obs_encoder import ObsEncoder

class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.args = args
        self.args.obs_train = False #CHANGE
        args.obs_train = False #CHANGE

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.obs_encoder = ObsEncoder(args).to(args.device) #CHANGE
        if args.obs_train:    
            pass        
            #self.target_obs_encoder = copy.deepcopy(self.obs_encoder) #CHANGE

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        if args.obs_train:
            #self.critic_optim = Adam(list(self.obs_encoder.parameters()) + list(self.critic.parameters()), lr=args.lr)
            pass
            
        else:
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            if args.obs_train:
                #self.policy_optim = Adam(list(self.obs_encoder.parameters()) + list(self.policy.parameters()), lr=args.lr)
                self.optimizer = Adam(list(self.critic.parameters()) + list(self.policy.parameters()), lr=args.lr)
            else:
                self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            if args.obs_train:
                self.policy_optim = Adam(list(self.obs_encoder.parameters()) + list(self.policy.parameters()), lr=args.lr)
            else:
                self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        if self.args.obs_train:
            task_obs, waypoints_obs, local_map, prev_action, pedestrian_map, ped_pos_obs, action_batch, reward_batch, \
                next_task_obs, next_waypoints_obs, next_local_map, next_prev_action, next_pedestrian_map, next_ped_pos_obs,\
                     mask_batch = memory.sample_obs_train(batch_size=batch_size)
            
            # task_obs, waypoints_obs, local_map, prev_action, pedestrian_map, ped_pos_obs = \
            #     map(torch.tensor, (task_obs, waypoints_obs, local_map, prev_action, pedestrian_map, ped_pos_obs))
            task_obs = task_obs.to(self.device).float()
            waypoints_obs = waypoints_obs.to(self.device).float()
            local_map = local_map.to(self.device).float()
            prev_action = prev_action.to(self.device).float()
            pedestrian_map = pedestrian_map.to(self.device).float()
            ped_pos_obs = ped_pos_obs.to(self.device).float()
            
            # next_task_obs, next_waypoints_obs, next_local_map, next_prev_action, next_pedestrian_map, next_ped_pos_obs = \
            #     map(torch.tensor, (next_task_obs, next_waypoints_obs, next_local_map, next_prev_action, next_pedestrian_map, next_ped_pos_obs))
            next_task_obs = next_task_obs.to(self.device).float()
            next_waypoints_obs = next_waypoints_obs.to(self.device).float()
            next_local_map = next_local_map.to(self.device).float()
            next_prev_action = next_prev_action.to(self.device).float()
            next_pedestrian_map = next_pedestrian_map.to(self.device).float()
            next_ped_pos_obs = next_ped_pos_obs.to(self.device).float()

            state_batch = self.obs_encoder(task_obs, waypoints_obs, local_map, prev_action, pedestrian_map, ped_pos_obs)
            
            with torch.no_grad():
                next_state_batch = self.target_obs_encoder(next_task_obs, next_waypoints_obs, next_local_map, next_prev_action, next_pedestrian_map, next_ped_pos_obs) 
            reward_batch = reward_batch.unsqueeze(1)
            mask_batch = mask_batch.unsqueeze(1)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
            
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
            

        #CRITIC
        #torch.autograd.set_detect_anomaly(True)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        #print("qf_loss", qf_loss)
        if self.args.obs_train:
            self.optimizer.zero_grad()
        #self.critic_optim.zero_grad()
        qf_loss.backward(retain_graph=True)
        #self.critic_optim.step()
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # POLICY
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        #self.policy_optim.zero_grad()
        policy_loss.backward()
        #print("policy loss", policy_loss)
        #self.policy_optim.step()

        if self.args.obs_train:
            self.optimizer.step()
        

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            if self.args.obs_train:
                soft_update_obs_train(self.critic_target, self.critic, self.tau, self.target_obs_encoder, self.obs_encoder)
            else:
                soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}".format(env_name)
        #print('Saving models to {}'.format(ckpt_path))
        if self.args.obs_train:
            torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    #'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    #'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'obs_encoder': self.obs_encoder.state_dict()}, ckpt_path)
        else:
            torch.save({'policy_state_dict': self.policy.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'critic_target_state_dict': self.critic_target.state_dict(),
                        'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                        'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            if self.args.obs_train:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
                if self.args.obs_train:
                    self.obs_encoder.load_state_dict(checkpoint['obs_encoder'])
                    self.obs_encoder.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
                if self.args.obs_train:
                    self.obs_encoder.load_state_dict(checkpoint['obs_encoder'])
                    self.obs_encoder.train()
