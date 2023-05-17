import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.modules.activation as activation_func


class CNNEncoder(nn.Module):

    def __init__(self, config, activation='ReLU'):
        super(CNNEncoder, self).__init__()
        self.config = config
        kernel_list = config.get('net', 'kernel_list')
        self.kernel_size_list = [int(i) for i in kernel_list.replace("[", "").replace("]", "").split(",")]
        self.in_channel = config.getint('data', 'vec_size')
        self.filters = config.getint('net', 'filters')
        self.activation = getattr(activation_func, activation)()

        self.convs_0 = nn.Conv1d(in_channels=self.in_channel, out_channels=self.filters, kernel_size=self.kernel_size_list[0])
        self.convs_1 = nn.Conv1d(in_channels=self.in_channel, out_channels=self.filters, kernel_size=self.kernel_size_list[1])
        self.convs_2 = nn.Conv1d(in_channels=self.in_channel, out_channels=self.filters, kernel_size=self.kernel_size_list[2])
        self.convs_3 = nn.Conv1d(in_channels=self.in_channel, out_channels=self.filters, kernel_size=self.kernel_size_list[3])

        # self.convs.append(nn.Conv1d(in_channels=self.in_channel, out_channels=self.filters, kernel_size=kernel_size))
        self.feature_len = len(self.kernel_size_list) * self.filters

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: [batch_size, sentence_length, feature_dim]
        :return:
        """
        inputs = inputs.transpose(1, 2)  # [batch_size, feature_dim, sentence_length]
        conv_out_0 = torch.max(F.relu(self.convs_0(inputs)), dim=-1)[0]
        conv_out_1 = torch.max(F.relu(self.convs_1(inputs)), dim=-1)[0]
        conv_out_2 = torch.max(F.relu(self.convs_2(inputs)), dim=-1)[0]
        conv_out_3 = torch.max(F.relu(self.convs_3(inputs)), dim=-1)[0]

        conv_out = torch.cat([conv_out_0, conv_out_1, conv_out_2, conv_out_3], dim=-1)

        return conv_out


class CNNEncoder2D(nn.Module):

    def __init__(self, config, activation='ReLU'):
        super(CNNEncoder2D, self).__init__()
        self.config = config
        kernel_list = config.get('net', 'kernel_list')
        self.kernel_size_list = [int(i) for i in kernel_list.replace("[", "").replace("]", "").split(",")]
        self.in_channel = config.getint('data', 'vec_size')
        self.filters = config.getint('net', 'filters')
        self.activation = getattr(activation_func, activation)()

        self.convs_0 = nn.Conv2d(in_channels=1, out_channels=self.filters,
                                 kernel_size=(self.kernel_size_list[0], self.in_channel))
        self.convs_1 = nn.Conv2d(in_channels=1, out_channels=self.filters,
                                 kernel_size=(self.kernel_size_list[1], self.in_channel))
        self.convs_2 = nn.Conv2d(in_channels=1, out_channels=self.filters,
                                 kernel_size=(self.kernel_size_list[2], self.in_channel))
        self.convs_3 = nn.Conv2d(in_channels=1, out_channels=self.filters,
                                 kernel_size=(self.kernel_size_list[3], self.in_channel))

        # self.convs.append(nn.Conv1d(in_channels=self.in_channel, out_channels=self.filters, kernel_size=kernel_size))
        self.feature_len = len(self.kernel_size_list) * self.filters

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: [batch_size, sentence_length, feature_dim]
        :return:
        """
        inputs = inputs.unsqueeze(dim=1)  # [batch_size, 1, sentence_length, feature_dim]
        conv_out_0 = torch.max(F.relu(self.convs_0(inputs)).squeeze(), dim=-1)[0]     # [batch_size, 75, 1, 1]
        conv_out_1 = torch.max(F.relu(self.convs_1(inputs)).squeeze(), dim=-1)[0]
        conv_out_2 = torch.max(F.relu(self.convs_2(inputs)).squeeze(), dim=-1)[0]
        conv_out_3 = torch.max(F.relu(self.convs_3(inputs)).squeeze(), dim=-1)[0]

        conv_out = torch.cat([conv_out_0, conv_out_1, conv_out_2, conv_out_3], dim=-1)

        return conv_out


