import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Create dictionary to record intermediate encoder outputs
        self.inter_encoder_outputs = {
            "level1": None,
            "level2": None,
            "level3": None,
            "level4": None
        }

        # Set conv and max pool hyperparameters
        self.conv3_size = 3
        self.pool_stride = 2
        self.pool_size = 2
        self.up_conv = 2

        # Set up max pool layer
        self.max_pool = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_stride)

        # Set up level 1 encoder conv layers
        self.level1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=self.conv3_size
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self.conv3_size
            ),
            nn.ReLU()
        )

        # Set up level 2 encoder conv layers
        self.level2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=self.conv3_size
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=self.conv3_size
            ),
            nn.ReLU()
        )

        # Set up level 3 encoder conv layers
        self.level3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=self.conv3_size
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=self.conv3_size
            ),
            nn.ReLU()
        )

        # Set up level 4 encoder conv layers
        self.level4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=self.conv3_size
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=self.conv3_size
            ),
            nn.ReLU()
        )

        # Set up level 5 encoder conv layers
        self.level5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=self.conv3_size
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=self.conv3_size
            ),
            nn.ReLU()
        )

        # Set up level 1 decoder conv layers
        self.upsample1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.upconv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=self.conv3_size
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=self.conv3_size
            ),
            nn.ReLU()
        )
        # Set up level 2 decoder conv layers
        self.upsample2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.upconv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=self.conv3_size
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=self.conv3_size
            ),
            nn.ReLU()
        )
        # Set up level 3 decoder conv layers
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upconv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=self.conv3_size
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=self.conv3_size
            ),
            nn.ReLU()
        )

        # Set up level 4 decoder conv layers
        self.upsample4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.upconv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=self.conv3_size
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self.conv3_size
            ),
            nn.ReLU()
        )

        self.output = nn.Conv2d(
            in_channels=64,
            out_channels=21,
            kernel_size=1
        )

    
    def crop_image(self, input_data, target):
        input_size = input_data.size()[2]
        target_size = target.size()[2]
        delta = input_size - target_size
        delta = delta // 2
        return input_data[:,:,delta:input_size-delta,delta:input_size-delta]

    def forward(self, input_data):
        """
        Forward pass of the network
        :param input_data: Current images input of size (batches, channels, height, width)
        :return: output: segmentation masks
        """
        # TODO implement forward pass with encoder/decoder
        output_encoder = self.encoder(input_data)
        output_decoder = self.decoder(output_encoder)
        return output_decoder

    def encoder(self, input_data):
        """
        Implements the downsampling portion of UNet
        :param input_data: input images of size (batches, channels, height, width)
        :return: l5_relu2: output of encoder
        """
        # level 1
        level1_out = self.level1(input_data)
        self.inter_encoder_outputs["level1"] = level1_out
        pool1 = self.max_pool(level1_out)

        # level 2
        level2_out = self.level2(pool1)
        self.inter_encoder_outputs["level2"] = level2_out
        pool2 = self.max_pool(level2_out)

        # level 3
        level3_out = self.level3(pool2)
        self.inter_encoder_outputs["level3"] = level3_out
        pool3 = self.max_pool(level3_out)

        # level 4
        level4_out = self.level4(pool3)
        self.inter_encoder_outputs["level4"] = level4_out
        pool4 = self.max_pool(level4_out)

        # level 5
        level5_out = self.level5(pool4)

        return level5_out

    def decoder(self, input_data):
        """
        Implements the upsampling portion of UNet
        :param input_data: Final output of encoder.
        :return: output: Final output of network
        """
        # TODO Yosuke
        # TODO: Note this input has not had up an up-conv operation applied
        #       to it yet. This should be the first step of the decoder
        # TODO: You can get the intermediate outputs from the encoder through self.inter_encoder_outputs
        output = None
        """
        Implements the upsampling portion of UNet
        :param input_data: input images of size (batches, channels, height, width)
        :return: l5_relu2: output of encoder
        """
        # level 6
        output_1 = self.upsample1(input_data)
        output_1_cropped = self.crop_image(self.inter_encoder_outputs['level4'], output_1)
        output_1 = self.upconv1(torch.cat((output_1, output_1_cropped), 1))
       
        # level 7
        output_2 = self.upsample2(output_1)
        output_2_cropped = self.crop_image(self.inter_encoder_outputs['level3'], output_2)
        output_2 = self.upconv2(torch.cat((output_2, output_2_cropped), 1))
        
        # level 8
        output_3 = self.upsample3(output_2)
        output_3_cropped = self.crop_image(self.inter_encoder_outputs['level2'], output_3)
        output_3 = self.upconv3(torch.cat((output_3, output_3_cropped), 1))
        
        # level 9
        output_4 = self.upsample4(output_3)
        output_4_cropped = self.crop_image(self.inter_encoder_outputs['level1'], output_4)
        output_4 = self.upconv4(torch.cat((output_4, output_4_cropped), 1))

        output = self.output(output_4)
        return output

if __name__ == "__main__":
    image = torch.rand(1,3,572,572)
    model = UNet()
    print(model(image))
