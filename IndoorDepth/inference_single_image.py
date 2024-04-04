# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2

import torch
from torchvision import transforms

import IndoorDepth.networks as networks
from IndoorDepth.layers import disp_to_depth

class DepthEstimator:

    #      ____                _              _
    #     / ___|___  _ __  ___| |_ _   _  ___| |_ ___  _ __
    #    | |   / _ \| '_ \/ __| __| | | |/ __| __/ _ \| '__|
    #    | |__| (_) | | | \__ \ |_| |_| | (__| || (_) | |
    #     \____\___/|_| |_|___/\__|\__,_|\___|\__\___/|_|
    def __init__(self, model_path, device) -> None:
        # init model
        self.device = device
        self.model_path = model_path
        self.encoder, self.decoder, self.thisH, self.thisW = self.prepare_model_for_test()
 
    #    ___       _ _   
    #   |_ _|_ __ (_) |_ 
    #    | || '_ \| | __|
    #    | || | | | | |_ 
    #   |___|_| |_|_|\__|
    def prepare_model_for_test(self):
        print("-> Loading model from ", self.model_path)
        encoder_path = os.path.join(self.model_path, "encoder.pth")
        decoder_path = os.path.join(self.model_path, "depth.pth")
        encoder_dict = torch.load(encoder_path, map_location=self.device)
        decoder_dict = torch.load(decoder_path, map_location=self.device)

        encoder = networks.ResnetEncoder(18, False)
        decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, 
            scales=range(1),num_output_channels=3, use_skips=True, PixelCoorModu=True
        )

        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
        decoder.load_state_dict(decoder_dict)

        encoder = encoder.to(self.device).eval()
        decoder = decoder.to(self.device).eval()

        return encoder, decoder, encoder_dict['height'], encoder_dict['width']

    #    _____                 _   _                 
    #   |  ___|   _ _ __   ___| |_(_) ___  _ __  ___ 
    #   | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
    #   |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
    #   |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
    def inference(self, input_image, gaze):
        with torch.no_grad():
            # convert openCV image to PIL image
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = pil.fromarray(input_image)

            # save original image size
            original_width, original_height = input_image.size
            crop_width, crop_height = input_image.size

            # resize because ...?
            input_image = input_image.resize((self.thisW, self.thisH), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # Norm_pix_coords
            # TODO: those values might need adjusting
            fx = 5.1885790117450188e+02 / (original_width - 2*16)
            fy = 5.1946961112127485e+02 / (original_height - 2*16)

            cx = (3.2558244941119034e+02 -16) / (original_width - 2*16)
            cy = (2.5373616633400465e+02 -16) / (original_height - 2*16)

            feed_height = self.thisH 
            feed_width = self.thisW

            Us, Vs = np.meshgrid(np.linspace(0, feed_width - 1, feed_width, dtype=np.float32),
                                np.linspace(0, feed_height - 1, feed_height, dtype=np.float32),
                                indexing='xy')
            Us /= feed_width
            Vs /= feed_height
            Ones = np.ones([feed_height, feed_width], dtype=np.float32)
            norm_pix_coords = np.stack(((Us - cx) / fx, (Vs - cy) / fy, Ones), axis=0)
            norm_pix_coords = torch.from_numpy(norm_pix_coords).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(self.device)
            features_tmp = self.encoder(input_image)
            outputs = self.decoder(features_tmp, norm_pix_coords)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (crop_height, crop_width), mode="bilinear", align_corners=False)

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

            im = pil.fromarray(colormapped_im) # TODO: convoert to cv2 image and return

            # get depth at gaze point
            depth_gaze = disp_resized_np[round(gaze[1]), round(gaze[0])]

            # convert to cv2 image
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

        return im, depth_gaze
