#Import all the packages
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import h5py
import numpy as np
import time
from pandas import DataFrame

#Dataset handling
class TEMImageDataset(Dataset):
  """Creates a Dataset object that takes in a h5 file with all the training images"""

  def __init__(self, image_filepath, img_key, labels_filepath, lbl_key):
    """ Args:
      image_filepath (string): path to the h5 file with the images
      img_key (string): the "key" for the images in the h5 file
      labels_filepath (string): path to the h5 file with the maps/labels
      lbl_key (string): the "key" for the labels in the h5 file
    """
    self.image = h5py.File(image_filepath,'r')[img_key][:,:,:,:]
    self.labels = h5py.File(labels_filepath,'r')[lbl_key][:,:,:,:]
  
  def __len__(self):
    return self.image.shape[0]
  
  def __getitem__(self, idx):
    #If the provided idx is a tensor, then convert it to a list
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    images = self.image[idx] #Note that these output np arrays, not tensors
    labels = self.labels[idx]
    sample = {'image': images, 'label': labels}
    
    return sample


#Model classes
class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size1, kernel_size2):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size1, padding=(kernel_size1-1)//2)
    self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size2, padding=(kernel_size2-1)//2)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.conv_add = nn.Conv2d(in_channels,out_channels,kernel_size = 1) #Expands/contracts input image in the filter dimension for the ReLU skip connection

    #Copied from torchvision to correctly initialize layers
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
  
  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.bn2(out)
    skip = self.conv_add(x)
    out += skip
    out = self.relu2(out)
    return out

class UNet(nn.Module): 
  def __init__(self, num_blocks,filter_sequence,max_pool_sequence, num_classes = 2):
    #num_blocks: number of residual blocks in network
    #filter_sequence: list of filter sizes
   
    super(UNet, self).__init__()
    self.downs = nn.ModuleList()
    self.ups = nn.ModuleList()
   
    self.pools = nn.ModuleList()
    self.upsamples = nn.ModuleList()
    

    # creates down and pooling layers
    in_channels = 1
    for i in range(num_blocks):
        self.downs.append(ResidualBlock(in_channels,filter_sequence[i],3,3))
        in_channels = filter_sequence[i]
        self.pools.append(nn.MaxPool2d(kernel_size = max_pool_sequence[i], stride = max_pool_sequence[i]))



    # creates up and upsampling layers
    for i in reversed(range(num_blocks)):
        self.ups.append(ResidualBlock(filter_sequence[i]+filter_sequence[i+1],filter_sequence[i],3,3)) #The 2*filters in the input channels refers to the extra channels from the concat layer
        self.upsamples.append(nn.Upsample(scale_factor=max_pool_sequence[i]))

    # "bottleneck" or middle part at bottom of U
    self.bottleneck = ResidualBlock(filter_sequence[num_blocks-1],filter_sequence[num_blocks],3,3)

    # final convolution with 1x1 kernel
    self.final_conv = nn.Conv2d(filter_sequence[0], num_classes, kernel_size = 1)

    self.num_blocks = num_blocks

  
  def forward(self, x):
  
    skips = [] # empty array to store skip connections

    for i in range(self.num_blocks):
        x = self.downs[i](x)
        skips.append(x)
        x = self.pools[i](x)

    x = self.bottleneck(x)
    skips = skips[::-1] # reverse skips array because we want to work with latest one first

    for idx in range(self.num_blocks):
      x = self.upsamples[idx](x)
      skip = skips[idx]
      concat_skip = torch.cat((skip,x),dim=1)
      x = self.ups[idx](concat_skip)

    out = self.final_conv(x)

    return out

#Training definitions
def train_loop(dataloader, model, loss_fn, optimizer, device):
  avg_loss = 0
  dice_score = 0
  hard_dice_score = 0
  size = len(dataloader.dataset)
  model.train()
  for batch_i, sample_batch in enumerate(dataloader):
    #Move the images and labels to the GPU
    images = sample_batch['image'].to(device)
    labels = sample_batch['label'].type(torch.LongTensor).to(device)

    #Compute the prediction (i.e. forward pass)
    pred = model(images)
    
    #Compute the loss
    loss = loss_fn(pred,labels[:,0,:,:]) #can't have dimensions in the channel index

    #Backpropagation
    #First, reset the gradients of the model parameters
    optimizer.zero_grad()
    #Back prop the prediction loss
    loss.backward()
    #Adjust the parameters using the gradients collected
    optimizer.step()

    #Log the loss and dice score
    avg_loss += loss.item()*sample_batch['image'].shape[0] #Undo the normalization by mini-batch size
    dice_score += compute_dice(pred, labels, device).item()
    hard_dice_score += compute_hard_dice(pred, labels, device).item()

  #After going through all the batches, compute the average loss and dice score
  return avg_loss/size, dice_score/size, hard_dice_score/size

def test_loop(dataloader, model, loss_fn, optimizer, device):
  avg_loss = 0
  dice_score = 0
  hard_dice_score = 0
  size = len(dataloader.dataset)
  model.eval()
  with torch.no_grad():
    for batch_i, sample_batch in enumerate(dataloader):
      #Move the images and labels to the GPU
      images = sample_batch['image'].to(device)
      labels = sample_batch['label'].type(torch.LongTensor).to(device)
      #Compute the prediction (i.e. forward pass)
      pred = model(images)
      #Compute the loss
      loss = loss_fn(pred,labels[:,0,:,:]) 
      #Log the loss and dice score
      avg_loss += loss.item()*sample_batch['image'].shape[0] #Undo the normalization by mini-batch size
      dice_score += compute_dice(pred, labels, device).item()
      hard_dice_score += compute_hard_dice(pred, labels, device).item()

  #After going through all the batches, compute the average loss and dice score
  return avg_loss/size, dice_score/size, hard_dice_score/size


def compute_dice(y_pred, y_truth, device1, smooth=1):
  #Need to first convert the ground truth into one-hot labels
  #Create a new one_hot tensor on the device
  one_hot = torch.zeros_like(y_pred, device=device1)
  one_hot.scatter_(1,y_truth,1) #first argument is dimension that needs to "expand", second is the indicies to go to, third is the value to place at those locations
  sm_layer = nn.Softmax2d()
  y_pred = sm_layer(y_pred) #normalize the class score
  intersection = torch.sum(y_pred[:,1,:,:] * one_hot[:,1,:,:], dim=(1,2)) #elemental multiplication and then sum, but only for one of the predictions, gives a N-sized tensor
  return torch.sum(torch.div(2. * intersection + smooth , torch.sum(one_hot[:,1,:,:], dim=(1,2)) + torch.sum(y_pred[:,1,:,:],dim=(1,2)) + smooth))

def compute_hard_dice(y_pred, y_truth, device1, smooth=1):
  #Need to first convert the ground truth into one-hot labels
  #Create a new one_hot tensor on the device
  one_hot = torch.zeros_like(y_pred, device=device1)
  one_hot.scatter_(1,y_truth,1) #first argument is dimension that needs to "expand", second is the indicies to go to, third is the value to place at those locations
  sm_layer = nn.Softmax2d()
  y_pred = torch.round(sm_layer(y_pred)) #normalize the class score
  intersection = torch.sum(y_pred[:,1,:,:] * one_hot[:,1,:,:], dim=(1,2)) #elemental multiplication and then sum, but only for one of the predictions, gives a N-sized tensor
  return torch.sum(torch.div(2. * intersection + smooth , torch.sum(one_hot[:,1,:,:], dim=(1,2)) + torch.sum(y_pred[:,1,:,:],dim=(1,2)) + smooth))


#Start of code 

#Import the data
train_img_filepath = 'Au_Training_DiAugment_Norm_PyTorch_Images.h5'
train_lbl_filepath = 'Au_Training_DiAugment_Norm_PyTorch_Labels.h5'
train_dataset = TEMImageDataset(image_filepath=train_img_filepath,img_key='images',labels_filepath=train_lbl_filepath,lbl_key = 'labels')

valid_img_filepath = 'Au_Validation_DiAugment_Norm_PyTorch_Images.h5'
valid_lbl_filepath = 'Au_Validation_DiAugment_Norm_PyTorch_Labels.h5'
valid_dataset = TEMImageDataset(image_filepath=valid_img_filepath,img_key='images',labels_filepath=valid_lbl_filepath,lbl_key = 'labels')

#Specify training parameters
learning_rate = 1e-4
b_size = 32
epochs = 100
#save_directory = 'b2_MaxPoolSequence_DiAugment_Au/'

#Specify model parameters
num_blocks = 3
filter_sequence = [4,8,16,32,64,128,256,512]
pool_sequence = [2,2,4] # Sets the max pooling kernel size. The length of this list has to be equal to num_blocks
pool_str = ''.join(str(i) for i in pool_sequence)

#Globally setting the random seed
seed_list = [752, 2245, 140, 9213, 956]
i=1
for seed_num in seed_list:
     torch.manual_seed(seed_num)

     #Setup the data loader
     train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, num_workers=4)
     valid_dataloader = DataLoader(valid_dataset, batch_size=b_size, shuffle=True, num_workers=4)

     model = UNet(num_blocks,filter_sequence,pool_sequence)
     loss_fn = nn.CrossEntropyLoss()
     optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

     #Setup GPU and move model
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     model.to(device)

     #Set the logging variables
     history = {'loss': [], 'dice_coef': [], 'hard_dice_coef': [], 'val_loss': [], 'val_dice_coef': [], 'val_hard_dice_coef': [], 'time': []}

     #Run the training
     for t in range(epochs):
       start_time=time.time()
       avg_loss, avg_dice, avg_hard_dice = train_loop(train_dataloader, model, loss_fn, optimizer, device)
       history['loss'].append(avg_loss)
       history['dice_coef'].append(avg_dice)
       history['hard_dice_coef'].append(avg_hard_dice)
       val_loss, val_dice, val_hard_dice = test_loop(valid_dataloader, model, loss_fn, optimizer, device)
       history['val_loss'].append(val_loss)
       history['val_dice_coef'].append(val_dice)
       history['val_hard_dice_coef'].append(val_hard_dice)  
       end_time = time.time()
       history['time'].append(end_time-start_time)
  

     #Save model and history file
     torch.save(model, 'UNet_b3_Au_m'+pool_str+'_run'+str(i)+'_model.h5')
     history_dataframe = DataFrame(history)
     hist_filename ='UNet_b3_Au_m'+pool_str+'_run'+str(i)+'_trainingLossHistory.csv'
     with open(hist_filename, mode='w') as f:
       history_dataframe.to_csv(f)
     
     i+=1

