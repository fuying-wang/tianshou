import torch
import numpy as np
from torch import nn
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Sequence

from tianshou.data import to_torch
from .utils import conv2d_layers_size_out, conv2d_size_out, weight_init


def miniblock(
    inp: int,
    oup: int,
    norm_layer: Optional[Callable[[int], nn.modules.Module]],
) -> List[nn.modules.Module]:
    """Construct a miniblock with given input/output-size and norm layer."""
    ret: List[nn.modules.Module] = [nn.Linear(inp, oup)]
    if norm_layer is not None:
        ret += [norm_layer(oup)]
    ret += [nn.ReLU(inplace=True)]
    return ret


class Net(nn.Module):
    """Simple MLP backbone.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape.
    :param bool dueling: whether to use dueling network to calculate Q values
        (for Dueling DQN), defaults to False.
    :param norm_layer: use which normalization before ReLU, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``, defaults to None.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: tuple,
        use_cam_obs: bool = False,
        use_phy_obs: bool = False,
        phy_state_shape: Optional[Union[tuple, int]] = 0,
        channel: int = 4,
        height: int = 64,
        width: int = 64,
        action_shape: Optional[Union[tuple, int]] = 0,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        hidden_layer_size: int = 128,
        dueling: Optional[Tuple[int, int]] = None,
        norm_layer: Optional[Callable[[int], nn.modules.Module]] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dueling = dueling
        self.softmax = softmax
        input_size = np.prod(state_shape)
        if concat:
            input_size += np.prod(action_shape)

        self.use_cam_obs = use_cam_obs
        self.use_phy_obs = use_phy_obs
        if not self.use_cam_obs:
            model = miniblock(input_size, hidden_layer_size, norm_layer)

            for _ in range(layer_num):
                model += miniblock(hidden_layer_size,
                                   hidden_layer_size, norm_layer)
            if self.use_phy_obs:
                phy_state_dim = np.prod(phy_state_shape)
                model = [PhysicalNet(
                    model, input_size, phy_state_dim, hidden_layer_size)]
        else:
            # model = [CNNBaseline(channel, height, width,
            #                      input_size, hidden_layer_size // 2,
            #                      self.use_phy_obs, device)]
            model = [CNN(channel, height, width, input_size,
                         hidden_layer_size, self.use_phy_obs, device)]
        if dueling is None:
            if action_shape and not concat:
                model += [nn.Linear(hidden_layer_size, np.prod(action_shape))]
        else:  # dueling DQN
            q_layer_num, v_layer_num = dueling
            Q, V = [], []

            for i in range(q_layer_num):
                Q += miniblock(
                    hidden_layer_size, hidden_layer_size, norm_layer)
            for i in range(v_layer_num):
                V += miniblock(
                    hidden_layer_size, hidden_layer_size, norm_layer)

            if action_shape and not concat:
                Q += [nn.Linear(hidden_layer_size, np.prod(action_shape))]
                V += [nn.Linear(hidden_layer_size, 1)]

            self.Q = nn.Sequential(*Q)
            self.V = nn.Sequential(*V)
        self.model = nn.Sequential(*model)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten -> logits."""
        if type(s) is tuple:
                s_0 = s[0]
                s_1 = s[1]
        elif type(s) is np.ndarray:
            if s.dtype == object:
                s_0 = s[:, 0]
                s_1 = s[:, 1]
        else:
            raise ValueError("No type %s!" % type(s))

        if not self.use_cam_obs:
            if self.use_phy_obs:
                robot_state = to_torch(
                    np.stack(s_0), device=self.device, dtype=torch.float32)
                physical_state = to_torch(
                    np.stack(s_1), device=self.device, dtype=torch.float32)
                logits = self.model([robot_state, physical_state])
            else:
                s = to_torch(s, device=self.device, dtype=torch.float32)
                s = s.reshape(s.size(0), -1)
                logits = self.model(s)
        else:
            img_top = to_torch(np.stack(s_0).transpose(
                (0, 3, 1, 2)), device=self.device, dtype=torch.float32)
            robot_state = to_torch(
                np.stack(s_1), device=self.device, dtype=torch.float32)
            logits = self.model([img_top, robot_state])
        if self.dueling is not None:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            logits = q - q.mean(dim=1, keepdim=True) + v
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state


class Recurrent(nn.Module):
    """Simple Recurrent network based on LSTM.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc1 = nn.Linear(np.prod(state_shape), hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, np.prod(action_shape))

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mapping: s -> flatten -> logits.

        In the evaluation mode, s should be with shape ``[bsz, dim]``; in the
        training mode, s should be with shape ``[bsz, len, dim]``. See the code
        and comment for more detail.
        """
        s = to_torch(s, device=self.device, dtype=torch.float32)
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            s = s.unsqueeze(-2)
        s = self.fc1(s)
        self.nn.flatten_parameters()
        if state is None:
            s, (h, c) = self.nn(s)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            s, (h, c) = self.nn(s, (state["h"].transpose(0, 1).contiguous(),
                                    state["c"].transpose(0, 1).contiguous()))
        s = self.fc2(s[:, -1])
        # please ensure the first dim is batch size: [bsz, len, ...]
        return s, {"h": h.transpose(0, 1).detach(),
                   "c": c.transpose(0, 1).detach()}


class FrontImgNet(nn.Module):
    def __init__(self, c, h, w, hidden_size=256):
        super(FrontImgNet, self).__init__()

        convh = conv2d_layers_size_out(h)
        convw = conv2d_layers_size_out(w)
        linear_input_size = convh * convw * 64

        self.output_shape = (hidden_size,)

        self.features = nn.Sequential(
            nn.Conv2d(c,  32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(linear_input_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, x):
        # c * h * w -> hidden_size
        return self.features(x)


class TopDownImgNet(nn.Module):
    def __init__(self, c, h, w, hidden_size=256):
        super(TopDownImgNet, self).__init__()

        convh = conv2d_layers_size_out(h)
        convw = conv2d_layers_size_out(w)
        linear_input_size = convh * convw * 64

        self.features = nn.Sequential(
            nn.Conv2d(c,  32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(linear_input_size, hidden_size),
            nn.ReLU()
        )

        self.output_shape = (hidden_size,)

    def forward(self, x):
        # 4 * 128 * 128 -> 64 * 12 * 12 -> 256
        return self.features(x)


class CNNBaseline(nn.Module):
    def __init__(self, c, h, w, state_dim, hidden_size, use_phy_obs, device):
        '''Feature extractor baseline
        Note: assume the shape of first-person view depth image is the same as the shape of top-down view depth image
        '''
        super(CNNBaseline, self).__init__()

        self.front_img_net = FrontImgNet(c, h, w, hidden_size)
        # if use_phy_obs:
        #     self.top_down_img_net = TopDownImgNet(c*2, h, w)
        # else:
        #     self.top_down_img_net = TopDownImgNet(c, h, w)
        self.top_down_img_net = TopDownImgNet(c, h, w)

        self.out_c, self.out_h, self.out_w = self.top_down_img_net.output_shape

        self.orientation_net = nn.Sequential(
            nn.Linear(state_dim, self.out_c * self.out_h * self.out_w),
            nn.ReLU()
        )

        out_h = conv2d_size_out(self.out_h, kernel_size=3, stride=1)
        out_w = conv2d_size_out(self.out_w, kernel_size=3, stride=1)
        linear_output_size = 64 * out_h * out_w

        self.fusion_net = nn.Sequential(
            nn.Conv2d(self.out_c, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(linear_output_size, hidden_size),
            nn.ReLU()
        )

        self.hidden_size = hidden_size
        self.output_shape = (2 * hidden_size,)
        self.device = device

        for module in self.modules():
            weight_init(module)

    def forward(self, s, use_lstm=False):
        img_front, img_top, robot_state = s
        feature_front = self.front_img_net(img_front)
        feature_top_down = self.top_down_img_net(img_top)
        feature_ori = self.orientation_net(robot_state)

        feature_ori = feature_ori.view(feature_ori.size(
            0), self.out_c, self.out_h, self.out_w)
        feature_top_down += feature_ori

        # B * 64 * 8 * 8 -> B * 64
        feature_fusion = self.fusion_net(feature_top_down)

        feature = torch.cat((feature_front, feature_fusion), dim=1)
        feature = feature.view(feature.size(0), -1)

        if not use_lstm:
            return feature
        else:
            bz = feature.size(0)
            input_size = feature.size(1)
            lstm_nn = nn.LSTMCell(input_size=input_size,
                                  hidden_size=2 * self.hidden_size).to(self.device)
            h_0 = torch.randn(bz, 2 * self.hidden_size).to(self.device)
            c_0 = torch.randn(bz, 2 * self.hidden_size).to(self.device)
            h_1, c_1 = lstm_nn(feature, (h_0, c_0))
            return h_1.reshape(bz, -1)


class PhysicalNet(nn.Module):
    def __init__(self, net, robot_state_dim, phy_state_dim, hidden_size):
        super(PhysicalNet, self).__init__()
        self.preprocess_net = nn.Sequential(*net)
        phy_model = miniblock(phy_state_dim, hidden_size, norm_layer=None)
        for _ in range(2):
            phy_model += miniblock(hidden_size, hidden_size, norm_layer=None)
        self.phy_model = nn.Sequential(*phy_model)

        model = miniblock(hidden_size, hidden_size, norm_layer=None)
        self.model = nn.Sequential(*model)

    def forward(self, s):
        robot_state, physical_state = s
        x = self.preprocess_net(robot_state)
        if physical_state.shape[1] == 0:
            return x
        else:
            y = self.phy_model(physical_state).mean(dim=1)
            return self.model(x + y)


class CNN(nn.Module):
    def __init__(self, c, h, w, state_dim, hidden_size, use_phy_obs, device):
        super(CNN, self).__init__()
        self.top_down_img_net = TopDownImgNet(c, h, w, hidden_size)
        self.orientation_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU()
        )
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        self.output_shape = (hidden_size,)
        for module in self.modules():
            weight_init(module)

    def forward(self, s):
        img_top, robot_state = s
        feature_top_down = self.top_down_img_net(img_top)
        feature_ori = self.orientation_net(robot_state)

        feature = torch.cat((feature_top_down, feature_ori), dim=1)
        out = self.fusion_net(feature)

        return out
