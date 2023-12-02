import numpy as np
from qiskit.utils import algorithm_globals

## import for MNIST data 
from scipy.ndimage import uniform_filter
import torch
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def generate_dataset(num_images, image_height, image_width, stroke_length):
    images = []
    labels = []

    # calculate the size of image and number of possible combinations of stroke in the image
    if (stroke_length > image_width or stroke_length > image_height):
        print("Stroke too long!")
        return images, labels
    elif (stroke_length < 2 or image_width < 2 or image_height < 2):
        print("Either stroke too short or image too small")
        return images, labels
    image_size = image_width * image_height
    number_of_patterns = (image_width - stroke_length + 1) * (image_height - stroke_length + 1)

    # list of all possible stroke patterns
    fwd_array = np.zeros((number_of_patterns, image_size))
    bkwd_array = np.zeros((number_of_patterns, image_size))

    # generate all forward slash patterns
    j = 0
    for i in range(0, image_size):
        if ((i % image_width) >= (stroke_length - 1) and (image_height - (i // image_width)) >= stroke_length):
            for k in range(0, stroke_length):
                fwd_array[j][i + k*(image_width - 1)] = np.pi / 2
            j += 1

    # generate all backward slash patterns
    j = 0
    for i in range(0, image_size):
        if ((i % image_width) <= (image_width - stroke_length) and (image_height - (i // image_width)) >= stroke_length):
            for k in range(0, stroke_length):
                bkwd_array[j][i + k*(image_width + 1)] = np.pi / 2
            j += 1
    
    # Test print all the combinations
    # print("forward strocks:")
    # for ind_array in fwd_array:
    #     ind_array = ind_array.reshape(4,3)
    #     print(ind_array)
    #     print("\n")

    # print("backward strocks:")
    # for ind_array in bkwd_array:
    #     ind_array = ind_array.reshape(4,3)
    #     print(ind_array)
    #     print("\n")

    for n in range(num_images):
        # rng = algorithm_globals.random.integers(0, 2)
        # if rng == 0:
        #     labels.append(-1)
        #     random_image = algorithm_globals.random.integers(0, number_of_patterns)
        #     images.append(np.array(fwd_array[random_image]))
        # elif rng == 1:
        #     labels.append(1)
        #     random_image = algorithm_globals.random.integers(0, number_of_patterns)
        #     images.append(np.array(bkwd_array[random_image]))

        if (n < num_images / 2):
            # first half are 0's
            labels.append(-1)
            random_image = algorithm_globals.random.integers(0, number_of_patterns)
            images.append(np.array(fwd_array[random_image]))
        else:
            # second half are 1's
            labels.append(1)
            random_image = algorithm_globals.random.integers(0, number_of_patterns)
            images.append(np.array(bkwd_array[random_image]))

        # Create noise
        for i in range(image_width*image_height):
            if images[-1][i] == 0:
                images[-1][i] = algorithm_globals.random.uniform(0, np.pi / 4)
    return np.array(images), np.array(labels)

def import_MNIST_Torch(num_samples, final_img_size):
    # extract the dataset from pytorch API
    # Set train shuffle seed (for reproducibility)
    manual_seed(42)

    batch_size = 1
    n_samples = num_samples  # We will concentrate on the first num_samples samples of each pattern

    # Use pre-defined torchvision function to load MNIST train data
    X_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )

    # Filter out labels (originally 0-9), leaving only labels 0 and 1
    idx = np.append(
        np.where(X_train.targets == 0)[0][:n_samples], np.where(X_train.targets == 1)[0][:n_samples]
    )
    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    # process the data
    # Define the blur kernel size
    original_size = 28
    target_size = final_img_size
    kernel_size = original_size // target_size

    # data structure to store output
    out_data = np.empty((n_samples*2, target_size, target_size))
    out_labels = X_train.targets.numpy()

    for data_idx, data in enumerate(X_train.data):
        # reshape and cast to numpy array
        img = data.reshape(28,28).numpy()

        # Apply uniform blur to the image (traditional CNN method)
        blurred_img = uniform_filter(img, size=kernel_size)

        # Subsample the image by selecting every 'kernel_size'-th pixel
        # shrunken_img = blurred_img[::kernel_size, ::kernel_size]
        shrunken_img = blurred_img[:target_size * kernel_size, :target_size * kernel_size]
        shrunken_img = shrunken_img.reshape((target_size, kernel_size, target_size, kernel_size)).mean(3).mean(1)

        # convert back to tensor and store it back
        # tensor_img = torch.from_numpy(shrunken_img)

        # add it back to X_train.data
        out_data[data_idx] = shrunken_img

    # Define torch dataloader with filtered data
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

    return out_data, out_labels
