import torch
import torch.nn as nn

import torch
import torch.nn as nn

class FeatureMapEncoder(nn.Module):
    def __init__(self,input_nc=3,z_dim=64,bottom=False):
        super().__init__()
        self.bottom = bottom

        if self.bottom:
            self.enc_down_0 = nn.Sequential([
                nn.Conv2d(input_nc + 4,z_dim,3,stride=1,padding=1),
                nn.ReLU(True)])
        self.enc_down_1 = nn.Sequential(nn.Conv2d(z_dim if bottom else input_nc+4, z_dim, 3, stride=2 if bottom else 1,  padding=1),
                                        nn.ReLU(True))
        self.enc_down_2 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_down_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_up_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_2 = nn.Sequential(nn.Conv2d(z_dim*2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_1 = nn.Sequential(nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True))

    def forward(self,x):
        """
        input:
            x: input image, [B,3,H,W]
        output:
            feature_map: [B,C,H,W]
        """
        W,H = x.shape[3], x.shape[2]
        X = torch.linspace(-1,1,W)
        Y = torch.linspace(-1,1,H)
        y1_m,x1_m = torch.meshgrid([Y,X])
        x2_m,y2_m = 2 - x1_m,2 - y1_m # Normalized distance in the four direction
        pixel_emb = torch.stack([x1_m,x2_m,y1_m,y2_m]).to(x.device).unsqueeze(0) # [1,4,H,W]
        pixel_emb = pixel_emb.repeat([x.size(0),1,1,1])
        inputs = torch.cat([x,pixel_emb],dim=1)

        if self.bottom:
            x_down_0 = self.enc_down_0(inputs)
            x_down_1 = self.enc_down_1(x_down_0)
        else:
            x_down_1 = self.enc_down_1(inputs)
        x_down_2 = self.enc_down_2(x_down_1)
        x_down_3 = self.enc_down_3(x_down_2)
        x_up_3 = self.enc_up_3(x_down_3)
        x_up_2 = self.enc_up_2(torch.cat([x_up_3, x_down_2], dim=1))
        feature_map = self.enc_up_1(torch.cat([x_up_2, x_down_1], dim=1))  # BxCxHxW
        return feature_map

class SlotAttention(nn.Module):
    def __init__(self,num_slots,in_dim=64,slot_dim=64,iters=3,eps=1e-8,hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots-1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots-1, slot_dim))
        nn.init.xavier_uniform_(self.slots_logsigma)
        self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_logsigma_bg)
        
        self.kslots_mu = nn.Parameter(torch.randn(1,num_slots,slot_dim))
        self.kslots_logsigma = nn.Parameter(torch.randn(1,num_slots,slot_dim))

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = slot_dim

    def forward(self,feat,num_slots = None):
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: BxKxC, attn: BxKxN
        """
        B, _, _ = feat.shape
        K = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.repeat(B,1,1)#.expand(B, K-1, -1)
        sigma = self.slots_logsigma.exp().repeat(B,1,1)#.expand(B, K-1, -1)
        slot_fg = mu + sigma * torch.randn_like(mu)
        
        mu_bg = self.slots_mu_bg.expand(B, 1, -1)
        sigma_bg = self.slots_logsigma_bg.exp().expand(B, 1, -1)
        slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)
        
        #mu_bg = self.slots_mu.expand(B, 1, -1)
        #sigma_bg = self.slots_logsigma.exp().expand(B, 1, -1)
        #slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)

        feat = self.norm_feat(feat)
        k = self.to_k(feat)
        v = self.to_v(feat)

        attn = None
        for _ in range(self.iters):
            slot_prev_bg = slot_bg
            slot_prev_fg = slot_fg
            q_fg = self.to_q(slot_fg)
            q_bg = self.to_q_bg(slot_bg)

            dots_fg = torch.einsum('bid,bjd->bij', q_fg, k) * self.scale
            dots_bg = torch.einsum('bid,bjd->bij', q_bg, k) * self.scale
            dots = torch.cat([dots_bg, dots_fg], dim=1)  # BxKxN
            attn = dots.softmax(dim=1) + self.eps  # BxKxN
            attn_bg, attn_fg = attn[:, 0:1, :], attn[:, 1:, :]  # Bx1xN, Bx(K-1)xN
            attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
            attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN

            updates_bg = torch.einsum('bjd,bij->bid', v, attn_weights_bg)
            updates_fg = torch.einsum('bjd,bij->bid', v, attn_weights_fg)

            slot_bg = self.gru_bg(
                updates_bg.reshape(-1, self.slot_dim),
                slot_prev_bg.reshape(-1, self.slot_dim)
            )
            slot_bg = slot_bg.reshape(B, -1, self.slot_dim)
            slot_bg = slot_bg + self.to_res_bg(slot_bg)

            slot_fg = self.gru(
                updates_fg.reshape(-1, self.slot_dim),
                slot_prev_fg.reshape(-1, self.slot_dim)
            )
            slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
            slot_fg = slot_fg + self.to_res(slot_fg)

        slots = torch.cat([slot_bg, slot_fg], dim=1)
        return slots, attn


class FeatureDecoder(nn.Module):
    def __init__(self, inchannel,input_channel,object_dim = 100):
        super(FeatureDecoder, self).__init__()
        self.im_size = 128
        self.conv1 = nn.Conv2d(inchannel + 2, 64, 3, bias=False)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(128, 128, 3, bias=False)
        # self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(128, 64, 3, bias=False)
        # self.bn4 = nn.BatchNorm2d(32)
        self.celu = nn.CELU()
        self.celu = nn.Sigmoid()
        self.inchannel = inchannel
        self.conv5_img = nn.Conv2d(64, input_channel, 1)
        self.conv5_mask = nn.Conv2d(64, 1, 1)

        x = torch.linspace(-1, 1, self.im_size + 8)
        y = torch.linspace(-1, 1, self.im_size + 8)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))
        self.bias = 0

        self.object_score_marker  =  nn.Linear(128 * 128 * 64,1)
        #self.object_score_marker   = FCBlock(256,2,64 * 64 * 16,1)
        #self.object_feature_marker = FCBlock(256,3,64 * 64 * 16,object_dim)
        self.object_feature_marker = nn.Linear(128 * 128 * 64,object_dim)
        self.conv_features         = nn.Conv2d(32,16,3,2,1)


    def forward(self, z):
        # z (bs, 32)
        bs,_ = z.shape

        z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.im_size + 8, self.im_size + 8)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        x = torch.cat((self.x_grid.expand(bs, -1, -1, -1),
                       self.y_grid.expand(bs, -1, -1, -1), z), dim=1)
        # x (bs, 32, image_h, image_w)

        x = self.conv1(x);x = self.celu(x) * 1.0
        x = torch.clamp(x, min = -32000, max = 32000)
        # x = self.bn1(x)
        x = self.conv2(x);x = self.celu(x) * 1.0#self.celu(x)
        x = torch.clamp(x, min = -32000, max = 32000)
        # x = self.bn2(x)
        x = self.conv3(x);x = self.celu(x) * 1.0
        x = torch.clamp(x, min = -32000, max = 32000)
        # x = self.bn3(x)
        x = self.conv4(x);x = self.celu(x) * 1.0
        x = torch.clamp(x, min = -32000, max = 32000)
        #x = self.bn4(x)

        img = self.conv5_img(x)
        img = .5 + 0.5 * torch.tanh(img + self.bias)
        logitmask = self.conv5_mask(x)
        
        #x = self.conv_features(x)
        #python3 train.py --name="KFT" --training_mode="joint" --pretrain_joint="checkpoints/Boomsday_toy_slot_attention.ckpt" --"save_path"="checkpoints/Boomsday_toy_slot_attention.ckpt"
        conv_features = torch.clamp(x.flatten(start_dim=1),min = -10, max = 10)
        #object_scores = torch.sigmoid(0.000015 * self.object_score_marker(conv_features)) 
        score = torch.clamp( 0.001 * self.object_score_marker(conv_features), min = -10, max = 10)
        object_scores = torch.sigmoid(score) 
        object_features = self.object_feature_marker(conv_features)

        return img, logitmask, object_features,object_scores

    def freeze_perception(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.object_score_marker.parameters():
            param.requires_grad = True
        for param in self.object_feature_marker.parameters():
            param.requires_grad = True
        for param in self.conv_features.parameters():
            param.requires_grad = True

    def unfreeze_perception(self):
        for param in self.parameters():
            param.requires_grad = True
        for param in self.object_score_marker.parameters():
            param.requires_grad = True
        for param in self.object_feature_marker.parameters():
            param.requires_grad = True
        for param in self.conv_features.parameters():
            param.requires_grad = True
        
class SlotAttentionParser(nn.Module):
    def __init__(self,num_slots,object_dim,num_iters):
        """
        output value:
        "full_recons":          [B,W,H,C],
        "masks":                [B,N,W,H,1],
        "recons":               [B,N,W,H,C],
        "object_features":      [B,N,O],
        "object_scores":        [B,N,1]
        "loss":loss,   
        """
        super().__init__()
        self.num_slots = num_slots # the number of objects and backgrounds in the scene
        self.object_dim = object_dim # the dimension of object level after the projection

        self.encoder_net = FeatureMapEncoder(input_nc = 3,z_dim = 256)
        self.slot_attention = SlotAttention(num_slots,256,256,num_iters)
        self.decoder_net = FeatureDecoder(256,3,object_dim)
        self.use_obj_score = False

        

    def allow_obj_score(self):self.use_obj_score = True

    def ban_obj_score(self):self.use_obj_score = False

    def freeze_perception(self):
        for param in self.encoder_net.parameters():
            param.requires_grad = False
        self.decoder_net.freeze_perception()
      
    def unfreeze_perception(self):
        for param in self.encoder_net.parameters():
            param.requires_grad = True
        self.decoder_net.unfreeze_perception()

    def forward(self,image):
        """
        "full_recons":recons.permute([0,2,3,1]),
        "masks":masks.permute([0,1,3,4,2]),
        "recons":img.permute([0,1,3,4,2]),
        "loss":loss,
        "object_features":object_features,
        "object_scores":object_scores
        """
        # preprocessing of image: channel first operation
        image = image.permute([0,3,1,2])
        b,c,w,h = image.shape
        # encoder model: extract visual feature map from the image
        feature_map = self.encoder_net(image)
        feature_inp = feature_map.flatten(start_dim = 2).permute([0,2,1]) # [B,N,C]
        
        # slot attention: generate slot proposals based on the feature net
        slots,attn  = self.slot_attention(feature_inp) 
        # slots: [b,K,C] attn: [b,N,C]
  
        # decoder model: make reconstructions and masks based on the 
        img, logitmasks, object_features, object_scores= self.decoder_net(slots.view([-1,256]))
        
        object_features = object_features.view([b,self.num_slots,-1])
        object_scores   = object_scores.view([b,self.num_slots,1])
        img             = img.view([b,self.num_slots,c,w,h])
        logitmasks      = 2 * logitmasks.view([b,self.num_slots,1,w,h])
        masks = torch.softmax(logitmasks,1) # masks of shape [b,ns,1,w,h]
        
        # reconstruct the image and calculate the reconstruction loss
        if self.use_obj_score:
            recons = torch.sum(object_scores.unsqueeze(-1).unsqueeze(-1) * img * masks,1)
        else:
            recons = torch.sum( img * masks,1)
        loss = torch.mean((recons-image) ** 2) # the mse loss of the reconstruction
        
        outputs = {"full_recons":recons.permute([0,2,3,1]),
                "masks":masks.permute([0,1,3,4,2]),
                "recons":img.permute([0,1,3,4,2]),
                "loss":loss,
                "object_features":object_features,
                "object_scores":object_scores}
        

        return outputs
        

class SlotAttentionRecursiveParser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scene_parsers = nn.ModuleList([
            SlotAttentionParser(config)
        ])
        self.effective_level = 1
    
    def forward(self,x):
        abstract_scene = []
        for parser in self.scene_parsers:
            outputs = parser(x)
            abstract_scene.append({})
        return abstract_scene


class FeatureDecoder64(nn.Module):
    def __init__(self, inchannel,input_channel,object_dim = 100):
        super(FeatureDecoder64, self).__init__()
        self.im_size = 64
        self.conv1 = nn.Conv2d(inchannel + 2, 32, 3, bias=False)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, bias=False)
        # self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, bias=False)
        # self.bn4 = nn.BatchNorm2d(32)
        self.celu = nn.CELU()
        self.inchannel = inchannel
        self.conv5_img = nn.Conv2d(32, input_channel, 1)
        self.conv5_mask = nn.Conv2d(32, 1, 1)

        x = torch.linspace(-1, 1, self.im_size + 8)
        y = torch.linspace(-1, 1, self.im_size + 8)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))
        self.bias = 0

        self.object_score_marker   = nn.Linear(64 * 64 * 32,1)
        #self.object_score_marker   = FCBlock(256,2,64 * 64 * 16,1)
        #self.object_feature_marker = FCBlock(256,3,64 * 64 * 16,object_dim)
        self.object_feature_marker = nn.Linear(inchannel,object_dim)
        self.conv_features         = nn.Conv2d(32,16,3,2,1)


    def forward(self, z):
        # z (bs, 32)
        bs,_ = z.shape
        object_features = self.object_feature_marker(z)
        z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.im_size + 8, self.im_size + 8)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        x = torch.cat((self.x_grid.expand(bs, -1, -1, -1),
                       self.y_grid.expand(bs, -1, -1, -1), z), dim=1)
        # x (bs, 32, image_h, image_w)
        x = self.conv1(x);x = self.celu(x)
        # x = self.bn1(x)
        x = self.conv2(x);x = self.celu(x)
        # x = self.bn2(x)
        x = self.conv3(x);x = self.celu(x)
        # x = self.bn3(x)
        x = self.conv4(x);x = self.celu(x)
        # x = self.bn4(x)

        img = self.conv5_img(x)
        img = .5 + 0.5 * torch.tanh(img + self.bias)
        logitmask = self.conv5_mask(x)

        x = self.conv_features(x)
        conv_features = x.flatten(start_dim=1)
        
        object_scores = torch.sigmoid( self.object_score_marker(conv_features)) 

        return img, logitmask, object_features,object_scores

    def freeze_perception(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.object_score_marker.parameters():
            param.requires_grad = True
        for param in self.object_feature_marker.parameters():
            param.requires_grad = True
        for param in self.conv_features.parameters():
            param.requires_grad = True

    def unfreeze_perception(self):
        for param in self.parameters():
            param.requires_grad = True
        for param in self.object_score_marker.parameters():
            param.requires_grad = True
        for param in self.object_feature_marker.parameters():
            param.requires_grad = True
        for param in self.conv_features.parameters():
            param.requires_grad = True


class SlotAttentionParser64(nn.Module):
    def __init__(self,num_slots,object_dim,num_iters):
        """
        output value:
        "full_recons":          [B,W,H,C],
        "masks":                [B,N,W,H,1],
        "recons":               [B,N,W,H,C],
        "object_features":      [B,N,O],
        "object_scores":        [B,N,1]
        "loss":loss,   
        """
        super().__init__()
        self.num_slots = num_slots # the number of objects and backgrounds in the scene
        self.object_dim = object_dim # the dimension of object level after the projection

        self.encoder_net = FeatureMapEncoder(input_nc = 3,z_dim = 72)
        self.slot_attention = SlotAttention(num_slots,72,72,num_iters)
        self.decoder_net = FeatureDecoder64(72,3,object_dim)
        self.use_obj_score = False

    def allow_obj_score(self):self.use_obj_score = True

    def ban_obj_score(self):self.use_obj_score = False

    def freeze_perception(self):
        for param in self.encoder_net.parameters():
            param.requires_grad = False
        self.decoder_net.freeze_perception()
      
    def unfreeze_perception(self):
        for param in self.encoder_net.parameters():
            param.requires_grad = True
        self.decoder_net.unfreeze_perception()

    def forward(self,image):
        """
        "full_recons":recons.permute([0,2,3,1]),
        "masks":masks.permute([0,1,3,4,2]),
        "recons":img.permute([0,1,3,4,2]),
        "loss":loss,
        "object_features":object_features,
        "object_scores":object_scores
        """
        # preprocessing of image: channel first operation
        image = image.permute([0,3,1,2])

        b,c,w,h = image.shape
        # encoder model: extract visual feature map from the image
        feature_map = self.encoder_net(image)
        feature_inp = feature_map.flatten(start_dim = 2).permute([0,2,1]) # [B,N,C]
        
        # slot attention: generate slot proposals based on the feature net
        slots,attn  = self.slot_attention(feature_inp) 
        # slots: [b,K,C] attn: [b,N,C]

        # decoder model: make reconstructions and masks based on the 
        img, logitmasks, object_features, object_scores= self.decoder_net(slots.view([-1,72]))
        
        object_features = object_features.view([b,self.num_slots,-1])
        object_scores   = object_scores.view([b,self.num_slots,1])
        img             = img.view([b,self.num_slots,c,w,h])
        logitmasks      = 2 * logitmasks.view([b,self.num_slots,1,w,h])
        masks = torch.softmax(logitmasks,1) # masks of shape [b,ns,1,w,h]
        
        # reconstruct the image and calculate the reconstruction loss
        if self.use_obj_score:
            recons = torch.sum(object_scores.unsqueeze(-1).unsqueeze(-1) * img * masks,1)
        else:
            recons = torch.sum( img * masks,1)
        loss = torch.mean((recons-image) ** 2) # the mse loss of the reconstruction
        return {"full_recons":recons.permute([0,2,3,1]),
                "masks":masks.permute([0,1,3,4,2]),
                "recons":img.permute([0,1,3,4,2]),
                "loss":loss,
                "object_features":object_features,
                "object_scores":object_scores}