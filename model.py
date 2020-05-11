import torch.nn as nn
from torchvision.models import resnext50_32x4d


def get_model(model_type,
              freeze_encoder=True,
              train_mode=True,
              principal_encoder=1):

    if model_type == "siamese":
        model = SiameseEncodingModel(freeze_encoder=freeze_encoder,
                                     train_mode=train_mode)
    elif model_type == "dual":
        model = DualBranchModel(freeze_encoder=freeze_encoder,
                                train_mode=train_mode,
                                principal_encoder=principal_encoder)
    else:
        raise RuntimeError("{}: not a valid model type".format(model_type))

    return model


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


class DualBranchModel(nn.Module):
    def __init__(self,
                 encoder_func=resnext50_32x4d,
                 enc_dim=256,
                 encoder_pretrained=True,
                 normalize=True,
                 freeze_encoder=False,
                 train_mode=True,
                 principal_encoder=1):
        super(DualBranchModel, self).__init__()

        # self.encoder_func = encoder_func

        def make_encoder():
            return encoder_func(pretrained=encoder_pretrained, progress=True)

        self.encoder1 = make_encoder()
        self.encoder2 = make_encoder()

        if freeze_encoder:
            print("Freezing encoders")

            for param in self.encoder1.parameters():
                param.requires_grad = False

            for param in self.encoder2.parameters():
                param.requires_grad = False

        # modify the encoder to get the desired encoding length
        self.encoder1.fc = nn.Linear(2048, out_features=enc_dim)
        self.encoder2.fc = nn.Linear(2048, out_features=enc_dim)

        for param in self.encoder1.fc.parameters():
            param.requires_grad = True

        for param in self.encoder2.fc.parameters():
            param.requires_grad = True

        self.should_normalize = normalize
        self.enc_dim = enc_dim
        self.train_mode = train_mode
        assert principal_encoder == 1 or principal_encoder == 2, "got {}".format(
            principal_encoder)
        self.principal_encoder = self.encoder1 if principal_encoder == 1 else self.encoder2

    def _normalize(self, x):
        return x.div(x.norm(p=2, dim=1,
                            keepdim=True)) if self.should_normalize else x

    def forward_train(self, x1, x2):
        enc1 = self._normalize(self.encoder1(x1))
        enc2 = self._normalize(self.encoder2(x2))

        return enc1, enc2

    def forward_test(self, x1):
        return self._normalize(self.principal_encoder(x1))

    def forward(self, *args):
        if self.train_mode:
            return self.forward_train(*args)
        else:
            return self.forward_test(*args)

    # def set_principal_encoder(self, n):
    #     if n == 1:
    #         self.principal_encoder = self.encoder1
    #     elif n == 2:
    #         self.principal_encoder = self.encoder2
    #     else:
    #         raise Exception(
    #             "Got invalid arg, pls enter 1 or 2, got {}".format(n))


def load_siamese_ckpt_into_dual(model, state_dict):
    print(state_dict)
    raise NotImplementedError()
