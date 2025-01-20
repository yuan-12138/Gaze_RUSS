import torch
import torch.nn as nn

class Conv_3d(nn.Module):
    """ 3d convolution block"""
    def __init__(self, in_channels, out_channels):
        super(Conv_3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Conv_2d(nn.Module):
    """2d convolution block"""
    def __init__(self, in_channels, out_channels):
        super(Conv_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Estimate_Module(nn.Module):
    def __init__(self, in_channels=1, init_features=12, time_steps = 64):
        super(Estimate_Module, self).__init__()

        features = init_features
        self.gaze_encoder1_3d = Conv_3d(in_channels, features)
        self.gaze_pool1_3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.gaze_encoder2_3d = Conv_3d(features, 2 * features)
        self.gaze_pool2_3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.gaze_encoder3_3d = Conv_3d(2 * features, 4 * features)
        self.gaze_pool3_3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.gaze_encoder4_3d = Conv_3d(4 * features, 8 * features)

        self.gaze_conv_1by1_1 = nn.Conv3d(time_steps, 1, kernel_size=1)
        self.gaze_conv_1by1_2 = nn.Conv3d(int(time_steps/2), 1, kernel_size=1)
        self.gaze_conv_1by1_3 = nn.Conv3d(int(time_steps/4), 1, kernel_size=1)
        self.gaze_conv_1by1_4 = nn.Conv3d(int(time_steps/8), 1, kernel_size=1)

        self.unet_encoder1_3d = Conv_3d(in_channels, features)
        self.unet_pool1_3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.unet_encoder2_3d = Conv_3d(features, 2 * features)
        self.unet_pool2_3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.unet_encoder3_3d = Conv_3d(2 * features, 4 * features)
        self.unet_pool3_3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.unet_encoder4_3d = Conv_3d(4 * features, 8 * features)

        self.unet_conv_1by1_1 = nn.Conv3d(time_steps, 1, kernel_size=1)
        self.unet_conv_1by1_2 = nn.Conv3d(int(time_steps/2), 1, kernel_size=1)
        self.unet_conv_1by1_3 = nn.Conv3d(int(time_steps/4), 1, kernel_size=1)
        self.unet_conv_1by1_4 = nn.Conv3d(int(time_steps/8), 1, kernel_size=1)

        self.up3_enc = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder3_enc = Conv_2d((features * 8) * 2, features * 8)

        self.up2_enc = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder2_enc = Conv_2d((features * 4) * 2, features * 4)

        self.up1_enc = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder1_enc = Conv_2d(features * 4, features * 2)

        self.conv = nn.Conv2d(features * 2, 1, kernel_size=1)

    def forward(self, gaze_heatmap, unet_heatmap):

        gaze_features_1 = self.gaze_encoder1_3d(gaze_heatmap)
        gaze_features_2 = self.gaze_encoder2_3d(self.gaze_pool1_3d(gaze_features_1))
        gaze_features_3 = self.gaze_encoder3_3d(self.gaze_pool2_3d(gaze_features_2))
        gaze_features_4 = self.gaze_encoder4_3d(self.gaze_pool3_3d(gaze_features_3))

        gaze_features_1_reduced = self.gaze_conv_1by1_1(gaze_features_1.permute(0, 2, 1, 3, 4)).squeeze(1)
        gaze_features_2_reduced = self.gaze_conv_1by1_2(gaze_features_2.permute(0, 2, 1, 3, 4)).squeeze(1)
        gaze_features_3_reduced = self.gaze_conv_1by1_3(gaze_features_3.permute(0, 2, 1, 3, 4)).squeeze(1)
        gaze_features_4_reduced = self.gaze_conv_1by1_4(gaze_features_4.permute(0, 2, 1, 3, 4)).squeeze(1)

        unet_features_1 = self.unet_encoder1_3d(unet_heatmap)
        unet_features_2 = self.unet_encoder2_3d(self.unet_pool1_3d(unet_features_1))
        unet_features_3 = self.unet_encoder3_3d(self.unet_pool2_3d(unet_features_2))
        unet_features_4 = self.unet_encoder4_3d(self.unet_pool3_3d(unet_features_3))

        unet_features_1_reduced = self.unet_conv_1by1_1(unet_features_1.permute(0, 2, 1, 3, 4)).squeeze(1)
        unet_features_2_reduced = self.unet_conv_1by1_2(unet_features_2.permute(0, 2, 1, 3, 4)).squeeze(1)
        unet_features_3_reduced = self.unet_conv_1by1_3(unet_features_3.permute(0, 2, 1, 3, 4)).squeeze(1)
        unet_features_4_reduced = self.unet_conv_1by1_4(unet_features_4.permute(0, 2, 1, 3, 4)).squeeze(1)

        center_reduced = torch.cat([gaze_features_4_reduced , unet_features_4_reduced], dim=1)

        skip3_gaze_unet = torch.cat([gaze_features_3_reduced, unet_features_3_reduced], dim=1)
        skip2_gaze_unet = torch.cat([gaze_features_2_reduced, unet_features_2_reduced], dim=1)
        skip1_gaze_unet = torch.cat([gaze_features_1_reduced, unet_features_1_reduced], dim=1)

        dec3_enc = self.decoder3_enc(torch.cat([self.up3_enc(center_reduced), skip3_gaze_unet], dim=1))
        dec2_enc = self.decoder2_enc(torch.cat([self.up2_enc(dec3_enc), skip2_gaze_unet], dim=1))
        dec1_enc = self.decoder1_enc(torch.cat([self.up1_enc(dec2_enc), skip1_gaze_unet], dim=1))

        return torch.sigmoid(self.conv(dec1_enc))