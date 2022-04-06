import numpy as np

def xray_correct_threshold(image,threshold=3000):
    # Removes x-ray outliers by finding pixels that are above a certain threshold above the median
    if type(image) is not np.ndarray:
        raise TypeError('Input must be numpy ndarray.')
    image[image<0] = 0
    bad_loc = np.argwhere(image > (np.median(image)+threshold))
    bad_loc_vals = image[image > (np.median(image)+threshold)]
    print('Number of detected x-rays: '+ str(len(bad_loc)))
    
    #Organize pixels by intensity
    sorted_ind = np.argsort(bad_loc_vals) #indices of lowest -> highest bad pixels
    bad_loc_sorted = bad_loc[sorted_ind,:] 
    bad_loc_sorted = np.flip(bad_loc_sorted,axis=0) #Reverse the order so it goes highest -> lowest
    for loc in bad_loc_sorted:
        if loc[0]+1 > (image.shape[0]-1) or loc[0]-1 < 0 or loc[1]+1 > (image.shape[1]-1) or loc[1] - 1 < 0:
            #If x-ray is in the corners, let the pixel value be the mean
            new_pixel_int = image.mean()
        else:
            #Otherwise take average of all 8 pixels surrounding the pixel in question
            neighbor_sum = np.sum(image[loc[0]-1,(loc[1]-1):(loc[1]+2)])+ image[loc[0],loc[1]-1] + image[loc[0],loc[1]+1] + np.sum(image[loc[0]+1,(loc[1]-1):(loc[1]+2)])
            new_pixel_int = neighbor_sum /8
        image[loc[0],loc[1]] = new_pixel_int
    #Repeat once more if there are still outliers
    bad_loc = np.argwhere(image > (np.median(image)+threshold))
    bad_loc_vals = image[image > (np.median(image)+threshold)]
    print('Number of detected x-rays: '+ str(len(bad_loc)))
    if len(bad_loc) > 0:
        sorted_ind = np.argsort(bad_loc_vals) #indices of lowest -> highest bad pixels
        bad_loc_sorted = bad_loc[sorted_ind,:] 
        bad_loc_sorted = np.flip(bad_loc_sorted,axis=0) #Reverse the order so it goes highest -> lowest
        for loc in bad_loc_sorted:
            if loc[0]+1 > (image.shape[0]-1) or loc[0]-1 < 0 or loc[1]+1 > (image.shape[1]-1) or loc[1] - 1 < 0:
                #If x-ray is in the corners, let the pixel value be the mean
                new_pixel_int = image.mean()
            else:
                #Otherwise take average of all 8 pixels surrounding the pixel in question
                neighbor_sum = np.sum(image[loc[0]-1,(loc[1]-1):(loc[1]+2)])+ image[loc[0],loc[1]-1] + image[loc[0],loc[1]+1] + np.sum(image[loc[0]+1,(loc[1]-1):(loc[1]+2)])
                new_pixel_int = neighbor_sum /8
            image[loc[0],loc[1]] = new_pixel_int
    
    return image

def create_patches(images,labels,patch_size=512):
    #Takes a list of images and splits it into patches
    #patch_size is the size of the patch (assume it will evenly go into the image)
    num_images = len(images)
    img_patches = []
    lbl_patches = []
    for i in range(num_images):
        img_to_split = images[i]
        lbl_to_split = labels[i]
        num_patch_y = img_to_split.shape[0]//patch_size
        num_patch_x = img_to_split.shape[1]//patch_size
        
        for j in range(num_patch_y):
            for k in range(num_patch_x):
                img_patches.append(img_to_split[j*patch_size:(j+1)*patch_size,k*patch_size:(k+1)*patch_size])
                lbl_patches.append(lbl_to_split[j*patch_size:(j+1)*patch_size,k*patch_size:(k+1)*patch_size])
    return img_patches, lbl_patches


def remove_bkgd_patches(imgs,lbls,threshold=100):
    #First convert to numpy array
    lbls = np.asarray(lbls)
    imgs = np.asarray(imgs)
    #Sum up all of the non-background pixels per image
    lbled_pixels = np.sum(lbls,axis=(1,2))
    #Only take patches with more than threshold pixels of "particle"
    lbls = lbls[lbled_pixels>threshold]
    imgs = imgs[lbled_pixels>threshold]
    return imgs, lbls

def standardize(data_array):
    #Takes in a data array and sets its mean to 0 and std to 1
    mean = np.mean(data_array)
    std = np.std(data_array)
    return (data_array-mean)/std

def write_h5(dataset, h5_name, label):
    #Saves a h5 file
    hf = h5py.File(h5_name, 'w')
    hf.create_dataset(label, data=dataset)
    hf.close()
    
def tvt_creation(imgs, lbls, split, seed):
    #Split the dataset
    valid_start = split[0]*imgs.shape[0]//100
    test_start = valid_start + split[1]*imgs.shape[0]//100

    train_dataset = imgs[:valid_start,:,:]
    train_labels = lbls[:valid_start,:,:] 

    valid_dataset = imgs[valid_start:test_start,:,:]
    valid_labels = lbls[valid_start:test_start,:,:]

    test_dataset = imgs[test_start:,:,:]
    test_labels = lbls[test_start:,:,:]

    print("Size of non-augmented training set: "+ str(train_dataset.shape))
    print("Size of non-augmented validation set: "+ str(valid_dataset.shape))
    print("Size of non-augmented test set: "+ str(test_dataset.shape))
    
    #Expand dimensions
    train_dataset = np.expand_dims(train_dataset, axis=3)
    train_labels = np.expand_dims(train_labels, axis=3)

    valid_dataset = np.expand_dims(valid_dataset, axis=3)
    valid_labels = np.expand_dims(valid_labels, axis=3)

    test_dataset = np.expand_dims(test_dataset, axis=3)
    test_labels = np.expand_dims(test_labels, axis=3)
    
    #Augment and Shuffle
    train_dataset, train_labels = dihedral_augmentation(train_dataset,train_labels)
    train_dataset, train_labels = shuffle_dataset(train_dataset, train_labels, seed)

    valid_dataset, valid_labels = dihedral_augmentation(valid_dataset,valid_labels)
    valid_dataset, valid_labels = shuffle_dataset(valid_dataset, valid_labels, seed+1)

    test_dataset, test_labels = dihedral_augmentation(test_dataset,test_labels)
    test_dataset, test_labels = shuffle_dataset(test_dataset, test_labels, seed+10)

    #Convert to PyTorch format
    train_dataset = pyTorch_format(train_dataset)
    train_labels = pyTorch_format(train_labels)

    valid_dataset = pyTorch_format(valid_dataset)
    valid_labels = pyTorch_format(valid_labels)

    test_dataset = pyTorch_format(test_dataset)
    test_labels = pyTorch_format(test_labels)
    
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def pyTorch_format(data_array):
    #Takes in a (N,H,W,C) numpy data array in float64 data type
    #Converts to a (N,C,H,W) numpy data array in float32 data type for PyTorch processing
    data_array = data_array.transpose((0,3,1,2))
    return np.float32(data_array)


def shuffle_dataset(data_array, data_labels, seed):
    #Takes a (N,H,W,C) dataset and labels and the randomization seed
    #Shuffles accordingly
    #Returns shuffled arrays
    np.random.seed(seed)
    new_index = np.arange(0,data_array.shape[0],1)
    np.random.shuffle(new_index)
    return data_array[new_index,:,:,:], data_labels[new_index,:,:,:]

def dihedral_augmentation(data_array,data_labels):
  #Takes in a (N,H,W,C) data array and its labels
  #Performs 90, 180, and 270 degree rotations, vertical flip, and horizontal flip. 
  #Returns data array and the expanded labels, though nothing is shuffled
    data_list = []
    label_list = []
    for i in range(data_array.shape[0]):
        image = data_array[i,:,:,:]
        label = data_labels[i,:,:,:]
        data_list.append(image)
        label_list.append(label)
        data_list.append(np.rot90(image,k=1))
        label_list.append(np.rot90(label,k=1))
        data_list.append(np.rot90(image,k=2))
        label_list.append(np.rot90(label,k=2))
        data_list.append(np.rot90(image,k=3))
        label_list.append(np.rot90(label,k=3))
        data_list.append(np.fliplr(image))
        label_list.append(np.fliplr(label))
        data_list.append(np.rot90(np.fliplr(image),k=1))
        label_list.append(np.rot90(np.fliplr(label),k=1))
        data_list.append(np.rot90(np.fliplr(image),k=2))
        label_list.append(np.rot90(np.fliplr(label),k=2))
        data_list.append(np.rot90(np.fliplr(image),k=3))
        label_list.append(np.rot90(np.fliplr(label),k=3))
    return np.asarray(data_list),np.asarray(label_list)
