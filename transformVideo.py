import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
import sys
import os.path
from PIL import Image
from torchvision import transforms
import mimetypes
import subprocess
from tqdm import tqdm
import os

class ResidualBlock(nn.Module):
  def __init__(self):
    super(ResidualBlock, self).__init__()
    self.conv_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.conv_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.norm_1 = nn.BatchNorm2d(256)
    self.norm_2 = nn.BatchNorm2d(256)

  def forward(self, x):
    output = self.norm_2(self.conv_2(F.relu(self.norm_1(self.conv_1(x)))))
    return output + x #ES

class Generator(nn.Module):
    def __init__(self):
      super(Generator, self).__init__()
      self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
      self.norm_1 = nn.BatchNorm2d(64)
      
      # down-convolution #
      self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
      self.conv_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
      self.norm_2 = nn.BatchNorm2d(128)
      
      self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
      self.conv_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
      self.norm_3 = nn.BatchNorm2d(256)
      
      # residual blocks #
      residualBlocks = []
      for l in range(8):
        residualBlocks.append(ResidualBlock())
      self.res = nn.Sequential(*residualBlocks)
      
      # up-convolution #
      self.conv_6 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
      self.conv_7 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
      self.norm_4 = nn.BatchNorm2d(128)

      self.conv_8 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
      self.conv_9 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
      self.norm_5 = nn.BatchNorm2d(64)
      
      self.conv_10 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
      x = F.relu(self.norm_1(self.conv_1(x)))
      
      x = F.relu(self.norm_2(self.conv_3(self.conv_2(x))))
      x = F.relu(self.norm_3(self.conv_5(self.conv_4(x))))
      
      x = self.res(x)
      x = F.relu(self.norm_4(self.conv_7(self.conv_6(x))))
      x = F.relu(self.norm_5(self.conv_9(self.conv_8(x))))

      x = self.conv_10(x)

      x = sigmoid(x)

      return x


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]


if __name__=='__main__':
    if(len(sys.argv) < 2):
        print("Usage: make transform IMAGE=PATH_TO_IMAGE_FILENAME")
        exit(0)
    if not (os.path.isfile('checkpoint_epoch_125.pth')):
        print('Can not find pre-trained weights file checkpoint_epoch_125.pth. Please provide within current directory.')
        exit(0)
    if mimetypes.guess_type(sys.argv[1])[0].startswith("image") and (os.path.isfile(sys.argv[1])):
        # print("{} is not a file".format(sys.argv[1]))
        # exit(0)
    
        checkpoint = torch.load('./checkpoint_epoch_125.pth', map_location='cpu')
        G = Generator().to('cpu')
        G.load_state_dict(checkpoint['g_state_dict'])
        transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
             ])

        with Image.open(sys.argv[1]) as img:
            # The input is needed as a batch, I got the solution from here:
            # https://discuss.pytorch.org/t/pytorch-1-0-how-to-predict-single-images-mnist-example/32394
            pseudo_batched_img = transformer(img)
            pseudo_batched_img = pseudo_batched_img[None]
            result = G(pseudo_batched_img)
            result = transforms.ToPILImage()(result[0]).convert('RGB')
            result.save('transformed.'+img.format)

    elif mimetypes.guess_type(sys.argv[1])[0].startswith("video"):
        # Create temp folder for storing frames as images
        temp_dir = tempfile.TemporaryDirectory()
        # Extract frames from video
        # ffmpeg_command = ["ffmpeg", "-i", sys.argv[1], "-loglevel error -stats", {os.path.join(temp_dir.name, 'frame_%07d.png')}]
        subprocess.run(["ffmpeg", "-i", sys.argv[1], "-loglevel", "error", "-stats", os.path.join(temp_dir.name, 'frame_%07d.png')])
        # Process images with model
        frame_paths = listdir_fullpath(temp_dir.name)
        # print(frame_paths)
        # batches = [*divide_chunks(frame_paths, 16)]
        for path_chunk in tqdm(frame_paths):
            imgs = Image.open(path_chunk)
            checkpoint = torch.load('./generator_release.pth', map_location='cpu')
            G = Generator().to('cpu')
            G.load_state_dict(checkpoint['g_state_dict'])
            transformer = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor()
                ])
            pseudo_batched_img = transformer(imgs)
            pseudo_batched_img = pseudo_batched_img[None]
            imgs = G(pseudo_batched_img)
            # for path, img in zip(path_chunk, imgs):
            result = transforms.ToPILImage()(imgs[0]).convert('RGB')
            result.save(path_chunk)
        # Get video frame rate
        frame_rate = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-of", "default=noprint_wrappers=1:nokey=1", "-show_entries", "stream=r_frame_rate" , sys.argv[1]])
        # frame_rate = eval(frame_rate.split()[0]) # Dirty eval
        # Combine frames with original audio
        subprocess.run(["ffmpeg", "-y", "-r", frame_rate, "-i" ,os.path.join(temp_dir.name, 'frame_%07d.png') , "-i", sys.argv[1], "-map", "0:v", "-map", "1:a?", "-loglevel", "error", "-stats", "test_out.mp4"])


