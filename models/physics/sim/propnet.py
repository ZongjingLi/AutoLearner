
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        '''
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.linear_1(self.relu(self.linear_0(x)))
        return x.view(B, N, self.output_size)

class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.D = output_size

    def forward(self, inputs):
        """
        inputs: particle states: [B, N, F]
        Batch, Num-Particle, Feature-Dim

        outputs: encoded states: [B, N, D]
        """
        D = self.D
        B, N, Z = inputs.shape
        x = inputs.reshape(B * N, Z)
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return x.reshape([B,N,D])

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.D = output_size

    def forward(self, inputs):
        """
        inputs: relation states: [B, N, F]
        Batch, Num-relations, Feature-Dim

        outputs: encoded states: [B, N, D]
        """
        D = self.D
        B, N, Z = inputs.shape
        x = inputs.reshape(B * N, Z)
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return x.reshape([B,N,D])

class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.residual = residual

        self.linear_0 = nn.Linear(input_size, output_size)

    def forward(self, inputs, res = None):
        B, N, D = inputs.size
        if self.residual:
            x = self.linear_0(inputs.reshape([B * N , D]))
            x = F.relu(x + res.reshape([B * N, self.output_size]))
        else:
            x = F.relu(self.linear_0(x.reshape(B * N, D)))
        return x.reshape([B, N, self.output_size])

class PropModule(nn.Module):
    def __init__(self, config, input_dim, output_dim, batch = True, residual = False):
        super().__init__()
        device = config.device

        self.device = device
        self.config = config

        self.batch = batch

        state_dim = config.state_dim
        attr_dim = config.attr_dim
        relation_dim = config.relation_dim
        action_dim = config.action_dim

        nf_particle = config.nf_particle
        nf_relation = config.nf_relation
        nf_effect = config.nf_effect

        self.nf_effect = nf_effect
        self.residual = residual

        # Particle Encoder
        self.particle_encoder = ParticleEncoder(
            input_dim, nf_particle, nf_effect
        )
        # Relation Encoder
        self.relation_encoder = RelationEncoder(
            2 * input_dim + relation_dim, nf_relation, nf_relation
        )

        # input: (1) particle encode (2) particle effect
        self.particle_propagator = Propagator(
            2 * nf_effect, nf_effect, self.residual
        )

        # input: (1) relation encode (2) sender_effect (3) receiver effect
        self.relation_propagator = Propagator(
            nf_relation + 2 * nf_effect, nf_effect
        )

        # input: (1) particle effect
        self.particle_predictor = ParticlePredictor(
            nf_effect, nf_effect, output_dim
        )

    def forward(self, state, Rr, Rs, Ra, pstep):

        # calculate the particle encoding
        particle_effect = Variable(torch.zeros((state.shape[0], state.shape[1], self.nf_effect)))
        particle_effect = particle_effect.to(self.device)

        # receiver_state and sender state

        if self.batch:
            Rrp = torch.transpose(Rr, 1, 2)
            Rsp = torch.transpose(Rs, 1, 2)
            state_r = Rrp.bmm(state)
            state_s = Rsp.bmm(state)
        else:
            Rrp = Rr.t()
            Rsp = Rs.t()
            assert state.shape[0] == 1
            state_r = Rrp.mm(state[0])[None, :, :]
            state_s = Rsp.mm(state[0])[None, :, :]
        
        # particle encode
        particle_encode = self.particle_encoder(state)

        # calcualte the relation encoding
        relation_encode = self.relation_encoder(torch.cat([state_r, state_s, Ra], 2))

        for i in range(pstep):
            if self.batch:
                effect_r = Rrp.bmm(particle_effect)
                effect_s = Rsp.bmm(particle_effect)
            else:
                assert particle_effect.shape[0] == 1
                effect_r = Rrp.mm(particle_effect[0])[None, :, :]
                effect_s = Rsp.mm(particle_effect[0])[None, :, :]
            
            # calculate relation effect
            relation_effect = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2)
            )

            # calculate particle effect by aggregating relation effect
            if self.batch:
                effect_agg = Rr.bmm(relation_effect)
            else:
                assert relation_effect.shape[0] == 1
                effect_agg = Rr.mm(relation_effect[0])[None, :, :]
            
            # calculate particle effect
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_agg], 2), 
                res = particle_effect
                )
        
        pred = self.particle_predictor(particle_effect)

        return pred

class PropNet(nn.Module):
    def __init__(self, config, residual = False):
        super().__init__()
        self.config = config
        nf_effect = config.nf_effect
        attr_dim = config.attr_dim
        state_dim = config.state_dim
        action_dim = config.action_dim
        position_dim = config.position_dim

        # input: (1) attr (2) state (3) [optional] action
        if config.pn_mode == "partial":
            batch = False
            input_dim = attr_dim + state_dim
            
            self.encoder = PropModule()
            self.decoder = PropModule()

            input_dim = (nf_effect + action_dim) * config.history_window
            self.roller = ParticlePredictor(input_dim, nf_effect, nf_effect)

        elif config.pn_mode == "full":
            batch = True
            input_dim = attr_dim + state_dim + action_dim
            self.model = PropModule(config, input_dim, position_dim, batch, residual)
        
    def encode(self, data, pstep):
        # used only for partially observable case
        config = self.config
        state, Rr, Rs, Ra = data 
        return self.encoder(state, Rr, Rs, Ra, pstep)

    def decode(self, data, pstep):
        # used only for partilly observable case
        config = self.config
        state, Rr, Rs, Ra = data
        return self.decoder(state, Rr, Rs, Ra, pstep)

        
    def rollout(self, state, action):
        # used only for partially observable case
        return self.roller(torch.cat([state, action], 2))

    def to_latent(self, state):
        if self.args.agg_method == 'sum':
            return torch.sum(state, 1, keepdim=True)
        elif self.args.agg_method == 'mean':
            return torch.mean(state, 1, keepdim=True)
        else:
            raise AssertionError("Unsupported aggregation method")

    def forward(self,data, pstep, action = None):
        # used only for fully observable case
        config = self.config
        attr, state, Rr, Rs, Ra = data

        if action is not None:
            state = torch.cat([attr, state, action], 2)
        else:
            state = torch.cat([attr, state], 2)
        return self.model(state, Rr, Rs, Ra, config.pstep)