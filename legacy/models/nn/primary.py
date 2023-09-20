import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace

class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.CELU(),
        )

    def forward(self, input):
        return self.net(input)

class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))

        if outermost_linear:
            self.net.append(nn.Linear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))

        self.net = nn.Sequential(*self.net)

    def __getitem__(self,item):
        return self.net[item]

    def forward(self, input):
        return self.net(input)


class VAE(torch.nn.Module):

    # For simplicity we assume encoder and decoder share same hyperparameters
    def __init__(self, in_features, num_hidden_layers=2, hidden_ch=128,
                                                 latent_features=5, beta=10):
        super().__init__()

        self.beta = beta

        self.encoder = FCBlock(in_features=in_features,
                               hidden_ch = hidden_ch,
                               num_hidden_layers=num_hidden_layers,
                               out_features=2*latent_features,
                               outermost_linear=True)

        self.decoder = FCBlock(in_features = latent_features,
                               hidden_ch = hidden_ch,
                               num_hidden_layers=num_hidden_layers,
                               out_features=in_features,
                               outermost_linear=True)

    # Returns reconstruction, reconstruction loss, and KL loss
    def forward(self, x):
        """
        return: recon, recon_loss, self.beta * kld_loss
        """

        # Split encoding into mu/logvar, reparameterize, and decode

        mu, log_var = self.encoder(x).chunk(2,dim=1)

        std    = torch.exp(0.5*log_var)
        eps    = torch.randn_like(std)
        sample = mu + (eps * std)

        recon  = self.decoder(sample)

        # Compute reconstruction and kld losses

        recon_loss = torch.linalg.norm(recon-x,dim=1)
        kld_loss   = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

        return recon, recon_loss, self.beta*kld_loss


class Attention(nn.Module):
    """Attention layer"""
        
    def __init__(self, dim, use_weight=False, hidden_size=512):
        super(Attention, self).__init__()
        self.use_weight = use_weight
        self.hidden_size = hidden_size
        if use_weight:
            print('| using weighted attention layer')
            self.attn_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_out = nn.Linear(2*dim, dim)

    def forward(self, output, context):
        """
        - args
        output : Tensor
            decoder output, dim (batch_size, output_size, hidden_size)
        context : Tensor
            context vector from encoder, dim (batch_size, input_size, hidden_size)
        - returns
        output : Tensor
            attention layer output, dim (batch_size, output_size, hidden_size)
        attn : Tensor
            attention map, dim (batch_size, output_size, input_size)
        """
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        if self.use_weight:
            output = self.attn_weight(output.contiguous().view(-1, hidden_size)).view(batch_size, -1, hidden_size)

        attn = torch.bmm(output, context.transpose(1, 2))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size) # (batch_size, output_size, input_size)

        mix = torch.bmm(attn, context) # (batch_size, output_size, hidden_size)
        comb = torch.cat((mix, output), dim=2) # (batch_size, output_size, 2*hidden_size)
        output = torch.tanh(self.linear_out(comb.view(-1, 2*hidden_size)).view(batch_size, -1, hidden_size)) # (batch_size, output_size, hidden_size)

        return output, attn

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self,node_feature_size, args = SimpleNamespace(G0=32  ,RDNkSize=3,n_colors=3,
                               RDNconfig=(4,3,16),scale=[2],no_upsampling=True)):
        super(RDN, self).__init__()
        self.args = args
        r = args.scale[0]
        G0 = node_feature_size
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = args.RDNconfig
        """
        {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]
        """

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            elif r == 4:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)