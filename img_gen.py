import numpy as np
from qiskit.utils import algorithm_globals

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
        rng = algorithm_globals.random.integers(0, 2)
        if rng == 0:
            labels.append(-1)
            random_image = algorithm_globals.random.integers(0, number_of_patterns)
            images.append(np.array(fwd_array[random_image]))
        elif rng == 1:
            labels.append(1)
            random_image = algorithm_globals.random.integers(0, number_of_patterns)
            images.append(np.array(bkwd_array[random_image]))

        # Create noise
        for i in range(image_width):
            if images[-1][i] == 0:
                images[-1][i] = algorithm_globals.random.uniform(0, np.pi / 4)
                
    return np.array(images), np.array(labels)