import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
import os
from os import listdir
from torchvision.datasets.utils import list_dir, list_files
#import matplotlib.pyplot as plt
import glob
import cv2

def ValidateImages(dataset_path, bad_folder_name = 'bad_images', move_bad_to_bad_folder = True, print_bad_filenames = False, image_format = '.png'):
  good_count = 0
  bad_count = 0

  if(move_bad_to_bad_folder): # Create folder for bad images
    if not os.path.exists(dataset_path+'/'+bad_folder_name):
          os.mkdir(dataset_path+'/'+bad_folder_name)

  for filename in listdir(dataset_path):
    if filename.endswith(image_format):
      try:
        img = Image.open(dataset_path+'/'+filename) # Open the image file
        img.verify() # Verify that it is, in fact an image
        good_count = good_count + 1
      except (IOError, SyntaxError) as e:
        if(move_bad_to_bad_folder):
          shutil.move(os.path.join(dataset_path, filename), os.path.join(dataset_path+'/'+bad_folder_name, filename))
          
        if (print_bad_filenames): print('Bad file:', filename) # Print out the names of corrupt files
        bad_count = bad_count + 1

  print('Good: ', good_count)
  print('Bad: ', bad_count)
  print()
  if(move_bad_to_bad_folder): print('Bads moved from dataset folder to ' + bad_folder_name + ' folder.')
  print('Done!')


def pad_images_to_same_size(images, force_squared):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)

    if(force_squared):
      if(width_max > height_max):
        height_max = width_max
      else:
        width_max = height_max

    print('Padding to (' + str(height_max) + ', ' + str(width_max) + ')')

    images_padded = []
    for img in images:
        #print(img)
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)
        #assert img_padded.shape[:2] == (height_max, width_max)
        images_padded.append(img_padded)

    return images_padded


def PadImagesAndSave(images, image_names, dest_folder, force_squared, image_format = '.png'):
  images_padded = []
  
  if not os.path.exists(dest_folder):
    os.mkdir(dest_folder)

  padded_images = pad_images_to_same_size(images, force_squared)

  print(len(padded_images))

  i = 0
  
  for img in padded_images:
    #print(img.shape)
    path = os.path.join(dataset_padded_path, image_names[i])
    cv2.imwrite(path , img)
    i = i + 1
    cv2.waitKey(0)

  print('Done!')
  return padded_images


def pad_images_to_same_size(images, force_squared, max_size = 0):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)

    if(force_squared):
      if(width_max > height_max):
        height_max = width_max
      else:
        width_max = height_max

    if(max_size > 0):
      if(height_max > max_size): height_max = max_size
      if(width_max > max_size): width_max = max_size

    print('Padding to (' + str(height_max) + ', ' + str(width_max) + ')')

    images_padded = []
    for img in images:
        #print(img)
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)
        #assert img_padded.shape[:2] == (height_max, width_max)
        images_padded.append(img_padded)

    return images_padded

def PadImagesAndSave(images, image_names, dest_folder, force_squared, image_format = '.png', max_size = 0):
  images_padded = []
  
  if not os.path.exists(dest_folder):
    os.mkdir(dest_folder)

  padded_images = pad_images_to_same_size(images, force_squared, max_size)

  print(len(padded_images))

  i = 0
  
  for img in padded_images:
    #print(img.shape)
    path = os.path.join(dataset_padded_path, image_names[i])
    cv2.imwrite(path , img)
    i = i + 1
    cv2.waitKey(0)

  print('Done!')
  return padded_images


class Chromosomes(Dataset):
    def __init__(self, 
                 root='/content/data', 
                 subset='train', 
                 im_transform=None, 
                 loader=pil_loader):
        self.root = root
        self.im_transform = im_transform
        self.loader = loader

        # locate the subset
        self.subdir = os.path.join(self.root, subset)

        # list the images in the image folder
        self.file_list = list_files(self.subdir, '.png', prefix=False)
        self.file_list.sort()
            
    def __getitem__(self, index):

        # get the image
        im_filename = self.file_list[index]

        # get the full image path
        im_filename = os.path.join(self.subdir, im_filename)

        # load the image
        im = self.loader(im_filename)

        # apply image transform if specified
        if self.im_transform is not None:
            im = self.im_transform(im)

        return im
    
    def __len__(self):
        # return the number of entries in the dataset
        return len(self.file_list)





dataroot = "./datasets/all"
dataset_padded_path = "./datasets/padded"
# Number of workers for dataloader
workers = 4
# Batch size during training
batch_size = 256
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 1


###Printing an image
#img_path = dataroot + '/' + '1_1_G134510K6.png'
##pyImage(img_path)

#anImage = Image.open(img_path)
#print(anImage.size)
#plt.imshow(anImage)


ValidateImages(dataroot)


# Reading images
files = glob.glob(os.path.join(dataroot, '*.png'))
images = []
image_names = []
for f in files:
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img)
    #print(os.path.basename(f))
    image_names.append(os.path.basename(f))

p_images = PadImagesAndSave(images, image_names, dataset_padded_path, True, 128)



'''
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.Grayscale(num_output_channels=nc),
                               transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
'''




