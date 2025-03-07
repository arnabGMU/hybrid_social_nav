import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os

torch.manual_seed(123)#torch.autograd.set_detect_anomaly(True)
class ObsEncoder(nn.Module):
    def __init__(self, args):
        super(ObsEncoder, self).__init__()
        self.state_dim = 0
        self.args = args
        #self.args.obs_train = False

        if args.obs_goal:
            self.goal_encoder = GoalEncoder(args)
            self.state_dim += args.feature_dim

        if args.obs_waypoints:
            self.waypoints_encoder = WayPointsEncoder(args)
            self.state_dim += args.feature_dim
        
        # if args.obs_lidar:
        #     self.map_encoder = LidarEncoder(args)
        #     self.state_dim += args.feature_dim

        if args.obs_map:
            if args.map_encoder == "cnn":
                self.map_encoder = MapEncoderCNN(args)
            else:
                self.map_encoder = MapEncoder(args)
            self.state_dim += args.feature_dim

        if args.obs_previous_action:
            self.previous_action_encoder = ActionEncoder(args)
            self.state_dim += args.feature_dim

        if args.obs_pedestrian_map:
            if args.map_encoder == "cnn":
                self.pedestrian_map_encoder = MapEncoderCNN(args)
            else:
                self.pedestrian_map_encoder = MapEncoder(args)
            self.state_dim += args.feature_dim
        
        # if args.obs_replan:
        #     self.replan_encoder = ReplanEncoder(args)
        #     self.state_dim += args.feature_dim

        if args.obs_pedestrian_pos:
            self.pedestrian_pos_encoder = PedPosEncoder(args)
            self.state_dim += args.feature_dim

        self.fc = nn.Linear(self.state_dim, args.feature_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x_goal=None, x_waypoints=None, x_lidar=None, x_action=None, x_ped_map=None, x_ped_pos=None):
        state = []
        if x_goal is not None:
            state.append(self.goal_encoder(x_goal))
        if x_waypoints is not None:
            state.append(self.waypoints_encoder(x_waypoints))
        if x_lidar is not None:
            state.append(self.map_encoder(x_lidar))
        if x_action is not None:
            state.append(self.previous_action_encoder(x_action))
        if x_ped_map is not None:
            state.append(self.pedestrian_map_encoder(x_ped_map))
        # if x_replan is not None:
        #     state.append(self.replan_encoder(x_replan))
        if x_ped_pos is not None:
            state.append(self.pedestrian_pos_encoder(x_ped_pos))
        x =  torch.cat(state, -1)
        
        x = self.fc(x)
        x = self.relu(x)
        return x
    
    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_obsEncoder".format(env_name)
        #print('Saving models to {}'.format(ckpt_path))
        d = {}
        d['obs_goal_encoder'] = self.goal_encoder.state_dict()
        d['obs_waypoints_encoder'] = self.waypoints_encoder.state_dict()
        d['map_encoder'] = self.map_encoder.state_dict()
        d['previous_action_encoder'] = self.previous_action_encoder.state_dict()
        if self.args.obs_pedestrian_map:
            d['obs_pedestrian_map_encoder'] = self.pedestrian_map_encoder.state_dict()
        if self.args.obs_pedestrian_pos:
            d['obs_pedestrian_pos_encoder'] = self.pedestrian_pos_encoder.state_dict()

        torch.save(d, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        ckpt_path += '_obsEncoder'
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.goal_encoder.load_state_dict(checkpoint['obs_goal_encoder'])
            self.waypoints_encoder.load_state_dict(checkpoint['obs_waypoints_encoder'])
            self.map_encoder.load_state_dict(checkpoint['map_encoder'])
            self.previous_action_encoder.load_state_dict(checkpoint['previous_action_encoder'])
            if self.args.obs_pedestrian_map:
                self.pedestrian_map_encoder.load_state_dict(checkpoint['obs_pedestrian_map_encoder'])
            if self.args.obs_pedestrian_pos:
                self.pedestrian_pos_encoder.load_state_dict(checkpoint['obs_pedestrian_pos_encoder'])

            if evaluate:
                self.goal_encoder.eval()
                self.waypoints_encoder.eval()
                self.map_encoder.eval()
                self.previous_action_encoder.eval()
                if self.args.obs_pedestrian_map:
                    self.pedestrian_map_encoder.eval()
                if self.args.obs_pedestrian_pos:
                    self.pedestrian_pos_encoder.eval()
            else:
                self.goal_encoder.train()
                self.waypoints_encoder.train()
                self.map_encoder.train()
                self.previous_action_encoder.train()
                if self.args.obs_pedestrian_map:
                    self.pedestrian_map_encoder.train()
                if self.args.obs_pedestrian_pos:
                    self.pedestrian_pos_encoder.train()
    
    def eval_encoder(self):
        self.goal_encoder.eval()
        self.waypoints_encoder.eval()
        self.map_encoder.eval()
        self.previous_action_encoder.eval()
        if self.args.obs_pedestrian_map:
            self.pedestrian_map_encoder.eval()
        if self.args.obs_pedestrian_pos:
            self.pedestrian_pos_encoder.eval()
    
    def train_encoder(self):
        self.goal_encoder.train()
        self.waypoints_encoder.train()
        self.map_encoder.train()
        self.previous_action_encoder.train()
        if self.args.obs_pedestrian_map:
            self.pedestrian_map_encoder.train()
        if self.args.obs_pedestrian_pos:
            self.pedestrian_pos_encoder.train()



class GoalEncoder(nn.Module):
    def __init__(self, args):
        super(GoalEncoder, self).__init__()
        if args.env == "iGibson":
            input_size = 4
        else:
            input_size = 2
        self.fc = nn.Linear(input_size, args.feature_dim, bias=True)
        self.relu = nn.ReLU()
        #self.bn1 = nn.BatchNorm1d(args.feature_dim)

    def forward(self, x):
        x = self.fc(x)
        #x = self.bn1(x)
        x = self.relu(x)
        return x

class WayPointsEncoder(nn.Module):
    def __init__(self, args):
        super(WayPointsEncoder, self).__init__()        
        self.fc = nn.Linear(2*args.num_wps_input, args.feature_dim, bias=True)
        self.relu = nn.ReLU()
        #self.bn1 = nn.BatchNorm1d(args.feature_dim)

    def forward(self, x):
        x = self.fc(x)
        #x = self.bn1(x)
        x = self.relu(x)
        return x
    
class PedPosEncoder(nn.Module):
    def __init__(self, args):
        super(PedPosEncoder, self).__init__()        
        self.fc = nn.Linear(3*args.fixed_num_pedestrians, args.feature_dim, bias=True)
        self.relu = nn.ReLU()
        #self.bn1 = nn.BatchNorm1d(args.feature_dim)

    def forward(self, x):
        x = self.fc(x)
        #x = self.bn1(x)
        x = self.relu(x)
        return x

class ActionEncoder(nn.Module):
    def __init__(self, args):
        super(ActionEncoder, self).__init__()
        input_size = 2
        self.fc = nn.Linear(input_size, args.feature_dim, bias=True)
        self.relu = nn.ReLU()
        #self.bn1 = nn.BatchNorm1d(args.feature_dim)

    def forward(self, x):
        x = self.fc(x)
        #x = self.bn1(x)
        x = self.relu(x)
        return x

# class ReplanEncoder(nn.Module):
#     def __init__(self, args):
#         super(ReplanEncoder, self).__init__()
#         input_size = 1
#         self.fc = nn.Linear(input_size, args.feature_dim, bias=True)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.fc(x)
#         x = self.relu(x)
#         return x
    
class LidarEncoder(nn.Module):
    def __init__(self, args):
        super(LidarEncoder, self).__init__()
        self.args = args
        self.lidar_measurements = 128
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(128 * self.lidar_measurements, args.feature_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(self.args.device).unsqueeze(0)
        x = x.view(-1, 1, self.lidar_measurements)

        if self.args.obs_train:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = x.view(-1, 128 * self.lidar_measurements)

            x = self.fc(x)
            x = self.relu(x)
        else:
            with torch.no_grad():
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = x.view(-1, 128 * self.lidar_measurements)

                x = self.fc(x)
                x = self.relu(x)
        return x

class MapEncoderCNN(nn.Module):
    def __init__(self, args):
        super(MapEncoderCNN, self).__init__()
        self.args = args
        self.cv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2)
        #self.bn1 = nn.BatchNorm2d(64)
        self.cv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1)
        #self.bn2 = nn.BatchNorm2d(64)
        self.cv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        #self.bn3 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32*21*21, args.feature_dim, bias=True)

        # self.cv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1)
        # self.cv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1)
        # self.cv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.fc = nn.Linear(32*10*10, args.feature_dim, bias=True)
        
        
        self.relu = nn.ReLU()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def forward(self, x):
        if self.args.obs_train:
            #print(x.shape)
            x = torch.stack([self.preprocess(i) for i in x]).to(self.args.device)
            #print(x.shape)
        else:    
            x = self.preprocess(x).to(self.args.device).unsqueeze(0)
        if self.args.obs_train:
            # batch norm
            # x = self.relu(self.bn1(self.cv1(x)))
            # x = self.relu(self.bn2(self.cv2(x)))
            # x = self.relu(self.bn3(self.cv3(x)))
            # x = x.view(-1, 32*21*21)

            x = self.relu(self.cv1(x))
            x = self.relu(self.cv2(x))
            x = self.relu(self.cv3(x))
            x = x.view(-1, 32*21*21)
            
            # Maxpool
            # x = self.pool(self.relu(self.cv1(x)))
            # x = self.pool(self.relu(self.cv2(x)))
            # x = self.pool(self.relu(self.cv3(x)))
            # x = x.view(-1, 32*10*10)

            x = self.fc(x)
            x = self.relu(x)
        else:
            with torch.no_grad():
                x = self.relu(self.cv1(x))
                x = self.relu(self.cv2(x))
                x = self.relu(self.cv3(x))
                x = x.view(-1, 32*21*21)

                # x = self.pool(self.relu(self.cv1(x)))
                # x = self.pool(self.relu(self.cv2(x)))
                # x = self.pool(self.relu(self.cv3(x)))
                # x = x.view(-1, 32*10*10)

                x = self.fc(x)
                x = self.relu(x)
        return x

class MapEncoder(nn.Module):
    def __init__(self, args):
        super(MapEncoder, self).__init__()
        self.args = args
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)
        self.resnet.eval()
        self.relu = nn.ReLU()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),                     # Convert to PIL image
            transforms.Resize((224, 224)),                # Resize to 224x224 pixels
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.ToTensor(),                        # Convert to tensor
            transforms.Normalize((0.5,), (0.5,))          # Normalize the image
        ])

        if self.args.obs_map_lstm == True:
            self.lstm = nn.LSTM(input_size=256,
                            hidden_size=256,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)
    
    def initialize(self):
        self.h = torch.zeros(2, 1, 256).to(self.args.device)
        self.c = torch.zeros(2, 1, 256).to(self.args.device)

    def forward(self, x):
        x = self.preprocess(x).to(self.args.device)
        if self.args.obs_train:
            x = self.resnet(x)
        else:
            with torch.no_grad():
                x = self.resnet(x)
        x = self.relu(x)
        if self.args.obs_map_lstm == True:
            x = x.view(1,1,x.shape[1])
            x, (self.h, self.c) = self.lstm(x, (self.h,self.c))
            x = x[:, -1, :]
        return x

