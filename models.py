import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import numpy as np
import random
from networks.resnet import resnet50, resnet101
from config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = cfg.device

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        self.resnet =resnet50(pretrained=True)  # pretrained ImageNet ResNet-101
        # Remove linear and pool layers (since we're not doing classification)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        try:

            [x4, x3, x2, x1] = self.resnet(images)  # [x4, x3, x2, x1]; x4 [B, 2048, image_size/32, image_size/32], x3 [B, 1024, image_size/16, image_size/16] x2, x1 
        except:
            import pdb; pdb.set_trace()

        encodedImage = self.adaptive_pool(x4)  # (batch_size, 2048, encoded_image_size, encoded_image_size)

        encodedImage = encodedImage.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        
        return [encodedImage, x4, x3, x2, x1]


# NOT USING ATTENTION!
class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PatchEncoder(nn.Module):
    def __init__(self, encoded_image_size=14, n_channels= 3, bilinear=True):
        super(PatchEncoder, self).__init__()
        #hard coded as 64 patch size -> 64 - 8 = 
        self.c0 = Down(n_channels, 64)

        self.c1 = Down(64, 128)
        self.c2 = Down(128, 256)
        self.c3 = Down(256, 256)
        if cfg.patch_size < 64:
            self.l1 = nn.Linear(128*8*8, 256)
        else:
            self.l1 = nn.Linear(256*8*8, 256)

        self.b1 = nn.BatchNorm2d(128)
        self.b2 = nn.BatchNorm2d(256)
        self.b3 = nn.BatchNorm2d(256)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.b1(self.c1(h)))
        if cfg.patch_size >=64:
            h = F.relu(self.b2(self.c2(h)))
        if cfg.patch_size == 128:
            h = F.relu(self.b3(self.c3(h)))
        # import pdb; pdb.set_trace()
        encoded = self.l1(h.view(x.shape[0], -1))
        
        return encoded
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)
        # import pdb; pdb.set_trace()
        # out = self.adaptive_pool(x)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        # out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)

        # x = F.relu(self.t_conv1(x))
        # x = F.sigmoid(self.t_conv2(x))

        # return encoded

class patchDecoder(nn.Module):
    def __init__(self, input_dim=512, n_channels= 3, output_channels = 4, bilinear=True):
        super(patchDecoder, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 4096))

        if cfg.patch_size ==32:
            self.decoder = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                # nn.Upsample(scale_factor=2, mode="bilinear"),
                # nn.Conv2d(64, 32, kernel_size=3, padding=1),
                # nn.BatchNorm2d(32),
                # nn.ReLU(),

                # nn.Upsample(scale_factor=2, mode="bilinear"),
                # nn.Conv2d(32, 32, kernel_size=3, padding=1),
                # nn.BatchNorm2d(32),
                # nn.ReLU(),

                nn.Conv2d(64, output_channels, kernel_size=1))

        elif cfg.patch_size == 64:

            self.decoder = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                # nn.Upsample(scale_factor=2, mode="bilinear"),
                # nn.Conv2d(32, 32, kernel_size=3, padding=1),
                # nn.BatchNorm2d(32),
                # nn.ReLU(),

                nn.Conv2d(32, output_channels, kernel_size=1))   

        elif cfg.patch_size == 128:

            self.decoder = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(32, output_channels, kernel_size=1))            

    def forward(self, x):
        assert x.size(1) == self.input_dim
        x = self.fc(x)

        x = x.view(x.size(0), 256, 4, 4)
        
        x = self.decoder(x)
        
        return x

# class patchDecoder(nn.Module):
#     def __init__(self):
#         super(patchDecoder, self).__init__()
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 5, stride=2, padding=1),  # b, 8, 15, 15
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
#             nn.Tanh()
#         )
#     def forward(self, x):
#         x = self.decoder(x)
#         return x

class patchEmbedding(nn.Module):
    def __init__(self, embedding_dim, patch_size, channel=1):
        super(patchEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.channel = channel
        self.flatten_dim = patch_size ** 2 * channel
        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        self.patch_encoder = PatchEncoder()

    def forward(self, input_batch, keypointSequence):
        '''
        input shape : (B, C, H, W)
        keypointSequence : [B, x, y]
        '''
        
        B,C,H, W = input_batch.shape
        _, max_len, _ = keypointSequence.shape

        padding_size = self.patch_size//2 + self.patch_size // 4
        input_batch = F.pad(input=input_batch, pad=(padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)

        # input_batch = input_batch.unsqueeze(2).repeat(1,1,max_len,1,1)

        patches_embedding = torch.zeros((B, max_len, self.channel, self.patch_size, self.patch_size)).to(device)
        canvas = torch.zeros((B, max_len, 1, H, W)).to(device)

        for b in range(B):
            for t in range(max_len):

                center = keypointSequence[b,t,:]
                # shift_x = random.randint(0,self.patch_size // 4)
                # shift_y = random.randint(0,self.patch_size // 4)
                shift_x = 0
                shift_y = 0
                center[0] = center[0] + shift_x
                center[1] = center[1] + shift_y
                x_coor_st = int(center[0].item() - self.patch_size//2 + padding_size)
                x_coor_ed = int(center[0].item() + self.patch_size//2 + padding_size)
                y_coor_st = int(center[1].item()- self.patch_size//2 + padding_size)
                y_coor_ed = int(center[1].item() + self.patch_size//2 + padding_size)
                try:
                    patches_embedding[b,t,:, :,:] = input_batch[b, :, y_coor_st :y_coor_ed, x_coor_st :x_coor_ed]
                except:
                    import pdb; pdb.set_trace()
                '''
                Visualize pathces
                '''
                # from torchvision import transforms
                # from PIL import Image
                # from PIL import ImageFile
                # to_image = transforms.ToPILImage()
                # to_image(patches_embedding[b,t,:,:,:].cpu()).show()
                # import pdb; pdb.set_trace()
                #############################################


                # patches_embedding shape should be [B,max_len, channel, patch_size, patch_size]
        # import pdb; pdb.set_trace()
        new_patch_embeddings = patches_embedding.reshape(B * max_len, 3, self.patch_size,self.patch_size)

        patches_embedding = self.patch_encoder(new_patch_embeddings)
        patch_embedding = patches_embedding.view(B,max_len,-1, self.embedding_dim)
        # import pdb; pdb.set_trace()
        #################################
        # patch_embedding = self.linear_encoding(patches_embedding.view(B,max_len,-1, self.flatten_dim).to(device))

        return patch_embedding



class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()
        self.encoder = Encoder()

        self.patch_size = cfg.patch_size

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.patchDecoder = patchDecoder()
        self.patchSegmentationDecoder = patchDecoder(output_channels=1)

        self.patchEmbedding = patchEmbedding(256, patch_size=self.patch_size, channel= 3)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        # self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # self.init_h = nn.Linear(encoder_dim + embed_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        # self.init_c = nn.Linear(encoder_dim + embed_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        

        self.decode_step = nn.LSTMCell(256, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(256, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(256, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate

        self.sigmoid = nn.Sigmoid()
        # self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, 2)  # linear layer to find scores over vocabulary
        self.fcStop = nn.Linear(decoder_dim, 1)  # linear layer to find scores over vocabulary
        self.fcStop_act = nn.Sigmoid()
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fcStop.bias.data.fill_(0)
        self.fcStop.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    # def init_hidden_state(self, encoder_out, patch_embedding):
    #     """
    #     Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

    #     :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
    #     :return: hidden state, cell state
    #     """
    #     mean_encoder_out = encoder_out.mean(dim=1)

    #     mean_encoder_embeddings = torch.cat([mean_encoder_out,patch_embedding], dim=1)
        
    #     h = self.init_h(mean_encoder_embeddings)  # (batch_size, decoder_dim)
    #     c = self.init_c(mean_encoder_embeddings)
    #     return h, c

    def init_hidden_state(self, patch_embedding):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # mean_encoder_out = encoder_out.mean(dim=1)

        # mean_encoder_embeddings = torch.cat([mean_encoder_out,patch_embedding], dim=1)
        
        h = self.init_h(patch_embedding)  # (batch_size, decoder_dim)
        c = self.init_c(patch_embedding)
        return h, c

    def forward(self, encoder_out, imgs, sequence, sequence_offset,caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        _, max_len, _ = sequence.shape
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        sort_ind = sort_ind.to(device)
        encoder_out = encoder_out[sort_ind]
        imgs = imgs[sort_ind]
        # import pdb; pdb.set_trace()
        #
        # d = torch.linspace(-1, 1, 8)
        # meshx, meshy = torch.meshgrid((d, d))
        #
        # out = self.patchEncoder(imgs)
        # import pdb; pdb.set_trace()
        '''
        '''
        encoded_captions = sequence_offset
        sequence_offset = sequence_offset[sort_ind]
        sequence = sequence[sort_ind].to(device)
        # import pdb; pdb.set_trace()
        embeddings = self.patchEmbedding(imgs, sequence)
        
        #[]
        embeddings = embeddings.squeeze()


        # Embedding
        # embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state

        # h, c = self.init_hidden_state(encoder_out, embeddings[:batch_size,0,:])  # (batch_size, decoder_dim)
        h, c = self.init_hidden_state(embeddings[:batch_size,0,:])  # (batch_size, decoder_dim)

        # h, c = self.init_hidden_state(encoder_out, None)  # (batch_size, decoder_dim)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1

        decode_lengths = (caption_lengths.to(device))
        
        # import pdb; pdb.set_trace()
        ###
        # Create tensors to hold word predicion scores and alphas
        if cfg.multi_gpu:
            # Create tensors to hold word predicion scores and alphas
            predictions = torch.zeros(batch_size, max_len, vocab_size).to(device)
            offset_prediction = torch.zeros(batch_size, max_len, 4, self.patch_size, self.patch_size).to(device)
            segmentation_prediction = torch.zeros(batch_size, max_len, 1, self.patch_size, self.patch_size).to(device)
            predictionsStop = torch.zeros(batch_size, max_len, 1).to(device)
            alphas = torch.zeros(batch_size, max_len, num_pixels).to(device)
        else:
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
            offset_prediction = torch.zeros(batch_size, max(decode_lengths), 4, self.patch_size, self.patch_size).to(device)
            segmentation_prediction = torch.zeros(batch_size, max(decode_lengths), 1, self.patch_size, self.patch_size).to(device)
            predictionsStop = torch.zeros(batch_size, max(decode_lengths), 1).to(device)
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        ###

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding

        if cfg.multi_gpu:
            max_steps = max_len
        else:
            max_steps = max(decode_lengths)

        for t in range(max_steps):
            # print(t)
            if cfg.multi_gpu:
                batch_size_t = len(decode_lengths)
            else:
                batch_size_t = sum([l > t for l in decode_lengths])
            # import pdb; pdb.set_trace()
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)

            attention_weighted_encoding = gate * attention_weighted_encoding

            # h, c = self.decode_step(
            #     torch.cat([attention_weighted_encoding, embeddings[:batch_size_t,t,:]], dim=1),
            #     # torch.cat([torch.floor(preds.data)], dim=1),
            #     (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            h, c = self.decode_step(
                embeddings[:batch_size_t,t,:],
                # torch.cat([torch.floor(preds.data)], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)

            # embeddings[:batch_size] -> CNN conv, + down sample
            # decoder : up sample
            #
            # disc,seg, short_offset_map, next_offset_map = decoder(h, [intermidiate]) 
            # post process -> 
            # import pdb; pdb.set_trace()
            patch_offset = self.patchDecoder(self.dropout(h))  

            patch_Seg = self.sigmoid(self.patchSegmentationDecoder(self.dropout(h)))        

            # preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predsStop = self.fcStop(self.dropout(h))  # (batch_size_t, vocab_size)
            predsStop = self.fcStop_act(predsStop)
            
            # import pdb; pdb.set_trace()
            
            offset_prediction[:batch_size_t, t, :,:,:] = patch_offset
            segmentation_prediction[:batch_size_t, t, :,:,:] = patch_Seg
            # predictions[:batch_size_t, t, :] = preds
            predictionsStop[:batch_size_t, t, :] = predsStop
            alphas[:batch_size_t, t, :] = alpha

        return predictions, predictionsStop, sequence_offset, decode_lengths, offset_prediction, segmentation_prediction, alphas, sort_ind
