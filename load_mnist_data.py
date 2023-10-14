import struct # https://docs.python.org/3/library/struct.html#struct.unpack
import numpy as np

# loads the mnist image and label file for processing in ml
# output is:
#   the number digits
#   the number of rows per digit
#   the number of columns per digit
#   an array of normalized pixel data
#   an array of floating point label data
def load_mnist_float(image_file, label_file):
    image_data = open(image_file, "rb").read()
    image_labels = open(label_file, "rb").read()

    # initial position to read to for header
    imagePosition = 16
    labelPosition = 8

    # read big endian header
    (imageMagic, N, rows, columns) = struct.unpack(">iiii", image_data[:imagePosition])
    (labelMagic, numLabels) = struct.unpack(">ii", image_labels[:labelPosition])
    if (N != numLabels):
        print("number of labels does not correspond to digits")

    pixelsInImage =  rows * columns

    digits = []
    labels = []
    imageCount = 0
    while imageCount < N:
        # read a byte buffer for the label and then the image
        label = struct.unpack("B", image_labels[labelPosition:labelPosition+1])
        pixels = struct.unpack("B" * pixelsInImage, image_data[imagePosition: imagePosition + pixelsInImage])
        normPixels = []
        # advance the position
        imagePosition += rows * columns
        labelPosition += 1
        for pixel in pixels:
            normPixels.append(pixel/255)
        digits.append(normPixels)
        output = np.zeros(10)
        output[label[0]] = 1.0
        labels.append(output)
        imageCount += 1
    return (N, rows, columns, digits, labels)

# loads the mnist image and label file 
# output is:
#   The number of digits
#   the number of rows per digit
#   the number of columns per digit
#   an int array of digit pixel data
#   an array of integer labels
def load_mnist(image_file, label_file):
    image_data = open(image_file, "rb").read()
    image_labels = open(label_file, "rb").read()

    # initial position to read to for header
    imagePosition = 16
    labelPosition = 8

    # read big endian header
    (imageMagic, N, rows, columns) = struct.unpack(">iiii", image_data[:imagePosition])
    print('image magic num:', hex(imageMagic), 'N:', N, 'rows:', rows, 'columns:', columns)
    (labelMagic, numLabels) = struct.unpack(">ii", image_labels[:labelPosition])
    print('label magic num:', hex(labelMagic), 'numLabels:', numLabels)
    if (N != numLabels):
        print("number of labels does not correspond to digits")

    pixelsInImage =  rows * columns

    digits = []
    labels = []
    imageCount = 0
    while imageCount < N:
        # read a byte buffer for the label and then the image
        label = struct.unpack("B", image_labels[labelPosition:labelPosition+1])
        pixels = struct.unpack("B" * pixelsInImage, image_data[imagePosition: imagePosition + pixelsInImage])
        # advance the position
        imagePosition += rows * columns
        labelPosition += 1
        digits.append(pixels)
        labels.append(label[0])
        imageCount += 1
    return (N, rows, columns, digits, labels)

# Reads the label file
# Input:
#   The label file name
# Output:
#   A list of labels read from the file
def load_mnist_labels(label_file):
    image_labels = open(label_file, "rb").read()
    labelPosition = 8
    (labelMagic, numLabels) = struct.unpack(">ii", image_labels[:labelPosition])
    labelCount = 0
    labels = []
    while labelCount < numLabels:
        label = struct.unpack("B", image_labels[labelPosition:labelPosition+1])
        labelPosition += 1
        labelCount += 1
        labels.append(label[0])
    return labels

# this method will write mnist data to file
#  images is the input  array of images to write
#  labels is the input array of image lables to write
#  count is the number of items to write to the file. expected to be <= sice of input arrays
#  image and label filnames are the files to output to.  overwrites existing data
def write_partial_mnist_data(images, labels, count, image_filename, label_filename):
    rows = 28
    columns = 28
    max = count
    image_buffer_pos = 16
    label_buffer_pos = 8
    image_buffer = bytearray(image_buffer_pos + max * rows * columns)
    label_buffer = bytearray(label_buffer_pos + max)

    struct.pack_into(">iiii", image_buffer, 0, 2051, max, 28, 28)
    struct.pack_into(">ii", label_buffer, 0, 2049, max)
    pixels_in_image = rows * columns
    for x in range(max):
        struct.pack_into("B" * pixels_in_image, image_buffer, image_buffer_pos, *images[x])
        struct.pack_into("B", label_buffer, label_buffer_pos, labels[x])
        image_buffer_pos += pixels_in_image
        label_buffer_pos += 1

    f=open(image_filename,"wb")
    f.write(image_buffer)
    f.close()

    f=open(label_filename,"wb")
    f.write(label_buffer)
    f.close()

def get_num_classes(train_labels):
    train_label_min = min(train_labels)
    train_label_max = max(train_labels)
    print("Class labels, min:", train_label_min, "max:", train_label_max)
    # assuming class labels are zero through train_label_max giving train_label_max + 1 classes
    return train_label_max + 1


# loads the mnist image and label file for processing in ml
# output is:
#   the number digits
#   the number of rows per digit
#   the number of columns per digit
#   an array of normalized pixel data
#   an array of floating point label data
# TODO: pass range to normaloze on...
def load_mnist_float_tanh(image_file, label_file):
    image_data = open(image_file, "rb").read()
    image_labels = open(label_file, "rb").read()

    # initial position to read to for header
    imagePosition = 16
    labelPosition = 8

    # read big endian header
    (imageMagic, N, rows, columns) = struct.unpack(">iiii", image_data[:imagePosition])
    (labelMagic, numLabels) = struct.unpack(">ii", image_labels[:labelPosition])
    if (N != numLabels):
        print("number of labels does not correspond to digits")

    pixelsInImage =  rows * columns

    digits = []
    labels = []
    imageCount = 0
    while imageCount < N:
        # read a byte buffer for the label and then the image
        label = struct.unpack("B", image_labels[labelPosition:labelPosition+1])
        pixels = struct.unpack("B" * pixelsInImage, image_data[imagePosition: imagePosition + pixelsInImage])
        normPixels = []
        # advance the position
        imagePosition += rows * columns
        labelPosition += 1
        for pixel in pixels:
            normPixels.append(pixel/255 * 2.0 - 1.0)
        digits.append(normPixels)
        output = np.zeros(10)
        output[label[0]] = 1.0
        labels.append(output)
        imageCount += 1
    return (N, rows, columns, digits, labels)