import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd.variable import Variable
from typing import List
from torchvision import models

class ResnetEncoder(nn.Module):
    def __init__(self, arch_name, in_channels=3, pretrained=False):
        super(ResnetEncoder, self).__init__()
        self.arch_name = arch_name
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.build_model()

    def build_model(self):
        print(f'Building {self.arch_name} model (pretrained={self.pretrained})!!')
                    
        if self.arch_name == 'resnet50':
            base_model = models.resnet50(pretrained=self.pretrained)
            self.features = nn.Sequential(*list(base_model.children())[:-1])

        elif self.arch_name == 'resnet18':
            base_model = models.resnet18(pretrained=self.pretrained)
            self.features = nn.Sequential(*list(base_model.children())[:-1]) 
                
        else:
            raise('This architecture is not supported!!')
        
        if self.in_channels != 3:
            self.features[0] =  nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if self.pretrained:
            i = 4 if self.in_channel != 3 else 0
            # Freeze all weights before CB4
            for param in self.features[i:6].parameters():
                param.requires_grad = False
                                        

    def forward(self, x):
        return self.features(x)

    
class Encoder(nn.Module):
    def __init__(self, in_channels=3, dropout=0.2):
        """
        Encoder for Image2NodeNet.
        :param dropout: dropout
        """
        super(Encoder, self).__init__()
        self.p = dropout
        self.conv1 = nn.Conv2d(in_channels, 8, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, 3, padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.conv4 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.max_pool2d(self.drop(F.relu(self.conv1(x))), (2, 2))
        x = F.max_pool2d(self.drop(F.relu(self.conv2(x))), (2, 2))
        x = F.max_pool2d(self.drop(F.relu(self.conv3(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        return x

class Image2NodeNet(nn.Module):
    def __init__(self,
                 hd_sz,
                 input_size,
                 inp_op_sz,
                 encoder,
                 num_layers=1,
                 time_steps=5,
                 dropout=0.5):
        """
        Defines RNN structure that takes features encoded by CNN and produces program
        instructions at every time step.
        :inp_op_sz: total number of unique operations
        :param dropout: dropout
        :param hd_sz: rnn hidden size
        :param input_size: input_size (CNN feature size) to rnn
        :param encoder: Feature extractor network object
        :param num_layers: Number of layers to rnn
        :param time_steps: max length of program
        """
        super(Image2NodeNet, self).__init__()
        self.hd_sz = hd_sz
        self.in_sz = input_size
        self.input_op_sz = inp_op_sz
        self.num_layers = num_layers
        self.encoder = encoder
        self.time_steps = time_steps

        self.rnn = nn.GRU(
            input_size=self.in_sz + self.input_op_sz,
            hidden_size=self.hd_sz,
            num_layers=self.num_layers,
            batch_first=False)


        self.logsoftmax = nn.LogSoftmax(1)
        self.softmax = nn.Softmax(1)

        self.dense_fc_1 = nn.Linear(
            in_features=self.hd_sz, out_features=self.hd_sz)
        self.dense_output = nn.Linear(
            in_features=self.hd_sz, out_features=(self.input_op_sz))
        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x: List):
        data, input_op, program_len = x

        assert input_op.size()[1] == program_len + 1, "Incorrect stack size!!"
        batch_size = data.size()[0]
        h = Variable(torch.zeros(self.num_layers, batch_size, self.hd_sz)).cuda()
        x_f = self.encoder(data)
        x_f = x_f.view(1, batch_size, self.in_sz)
        outputs = []
        for timestep in range(0, program_len + 1):
            # X_f is always input to the RNN at every time step
            # along with previous predicted label
            input_op_rnn = input_op[:, timestep, :]
            input_op_rnn = input_op_rnn.view(1, batch_size,
                                                self.input_op_sz)
            input = torch.cat((self.drop(x_f), input_op_rnn), 2)
            out, h = self.rnn(input, h)
            hd = self.relu(self.dense_fc_1(self.drop(out[0])))
            output = self.logsoftmax(self.dense_output(self.drop(hd)))
            outputs.append(output)
        return outputs
    
    def test(self, x: List):

        data, input_op, program_len = x

        batch_size = data.size()[0]
        h = Variable(torch.zeros(self.num_layers, batch_size, self.hd_sz))
        x_f = self.encoder(data)
        x_f = x_f.view(1, batch_size, self.in_sz)
        last_output = input_op[:,0,:]
        outputs = []
        for timestep in range(0, program_len + 1):
            input_op_rnn = last_output.view(1, batch_size,
                                                self.input_op_sz)
            input = torch.cat((self.drop(x_f), input_op_rnn), 2)
            out, h = self.rnn(input, h)
            hd = self.relu(self.dense_fc_1(self.drop(out[0])))
            output = self.logsoftmax(self.dense_output(self.drop(hd)))
            next_input_op = torch.max(output, 1)[1].view(batch_size, 1)
            arr = Variable(
                    torch.zeros(batch_size, self.input_op_sz).scatter_(
                        1, next_input_op.data.cpu(), 1.0)).cuda()

            last_output = arr
            outputs.append(output)
        return outputs

    def beam_search(self, data: List, w: int, max_time: int):
        """
        Implements beam search for different models.
        :param data: Input data
        :param w: beam width
        :param max_time: Maximum length till the program has to be generated
        :return all_beams: all beams to find out the indices of all the
        """
        data, input_op = data

        # Beam, dictionary, with elements as list. Each element of list
        # containing index of the selected output and the corresponding
        # probability.
        batch_size = data.size()[0]
        h = Variable(torch.zeros(1, batch_size, self.hd_sz))
        # Last beams' data
        B = {0: {"input": input_op, "h": h}, 1: None}
        next_B = {}
        x_f = self.encoder(data)
        x_f = x_f.view(1, batch_size, self.in_sz)
        # List to store the probs of last time step
        prev_output_prob = [
            Variable(torch.ones(batch_size, self.input_op_sz))
        ]
        all_beams = []
        all_inputs = []
        for timestep in range(0, max_time):
            outputs = []
            for b in range(w):
                if not B[b]:
                    break
                input_op = B[b]["input"]

                h = B[b]["h"]
                input_op_rnn = input_op[:,0,:].view(1, batch_size,
                                                 self.input_op_sz)
                input = torch.cat((x_f, input_op_rnn), 2)
                out, h = self.rnn(input, h)
                hd = self.relu(self.dense_fc_1(self.drop(out[0])))
                dense_output = self.dense_output(self.drop(hd))
                output = self.logsoftmax(dense_output)
                # Element wise multiply by previous probabs
                output = torch.nn.Softmax(1)(output)

                output = output * prev_output_prob[b]
                outputs.append(output)
                next_B[b] = {}
                next_B[b]["h"] = h
            if len(outputs) == 1:
                outputs = outputs[0]
            else:
                outputs = torch.cat(outputs, 1)

            next_beams_index = torch.topk(outputs, w, 1, sorted=True)[1]
            next_beams_prob = torch.topk(outputs, w, 1, sorted=True)[0]
            # print (next_beams_prob)
            current_beams = {
                "parent":
                next_beams_index.data.cpu().numpy() // (self.input_op_sz),
                "index": next_beams_index % (self.input_op_sz)
            }
            # print (next_beams_index % (self.num_draws))
            next_beams_index %= (self.input_op_sz)
            all_beams.append(current_beams)

            # Update previous output probabilities
            temp = Variable(torch.zeros(batch_size, 1))
            prev_output_prob = []
            for i in range(w):
                for index in range(batch_size):
                    temp[index, 0] = next_beams_prob[index, i]
                prev_output_prob.append(temp.repeat(1, self.input_op_sz))
            # hidden state for next step
            B = {}
            for i in range(w):
                B[i] = {}
                temp = Variable(torch.zeros(h.size()))
                for j in range(batch_size):
                    temp[0, j, :] = next_B[current_beams["parent"][j, i]]["h"][
                        0, j, :]
                B[i]["h"] = temp

            # one_hot for input to the next step
            for i in range(w):
                arr = Variable(
                    torch.zeros(batch_size, self.input_op_sz).scatter_(
                        1, next_beams_index[:, i:i + 1].data.cpu(),
                        1.0))
                B[i]["input"] = arr.unsqueeze(1)
            all_inputs.append(B)

        return all_beams, next_beams_prob, all_inputs

