import torch.nn as nn
from torchvision.models import resnext50_32x4d


class SiameseEncodingModel(nn.Module):
    def __init__(self,
                 encoder_func=resnext50_32x4d,
                 enc_dim=256,
                 encoder_pretrained=True,
                 normalize=True,
                 freeze_encoder=False,
                 train_mode=True):
        super(SiameseEncodingModel, self).__init__()
        # self.encoder_func = encoder_func

        self.encoder = encoder_func(pretrained=encoder_pretrained,
                                    progress=True)

        if freeze_encoder:
            print("Freezing encoder")

            for param in self.encoder.parameters():
                param.requires_grad = False

        # modify the encoder to get the desired encoding length
        self.encoder.fc = nn.Linear(2048, out_features=enc_dim)

        for param in self.encoder.fc.parameters():
            param.requires_grad = True

        self.should_normalize = normalize
        self.enc_dim = enc_dim
        self.train_mode = train_mode

    def _normalize(self, x):
        return x.div(x.norm(p=2, dim=1,
                            keepdim=True)) if self.should_normalize else x

    def forward_train(self, x1, x2):
        enc1 = self._normalize(self.encoder(x1))
        enc2 = self._normalize(self.encoder(x2))

        return enc1, enc2

    def forward_test(self, x1):
        return self._normalize(self.encoder(x1))

    def forward(self, *args):
        if self.train_mode:
            return self.forward_train(*args)
        else:
            return self.forward_test(*args)
