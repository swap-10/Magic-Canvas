import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("iam_model.h5") 

def flip_and_rotate(image):
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

vocab = ['[UNK]',
 '!',
 '"',
 '#',
 '&',
 "'",
 '(',
 ')',
 '*',
 '+',
 ',',
 '-',
 '.',
 '/',
 '0',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '9',
 ':',
 ';',
 '?',
 'A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'J',
 'K',
 'L',
 'M',
 'N',
 'O',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'U',
 'V',
 'W',
 'X',
 'Y',
 'Z',
 'a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'j',
 'k',
 'l',
 'm',
 'n',
 'o',
 'p',
 'q',
 'r',
 's',
 't',
 'u',
 'v',
 'w',
 'x',
 'y',
 'z']

num_to_char = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None, invert=True)

def decode_batch_predictions(pred):
    max_len = 21
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

img = cv2.imread("test_sure.jpg", cv2.IMREAD_GRAYSCALE)
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_OTSU)
img = img[..., np.newaxis]
print(img.shape)
img = tf.image.resize_with_pad(img, 32, 128)
img = tf.image.flip_left_right(img)
img = tf.transpose(img, perm=[1, 0, 2])
img = np.rot90(img)
img = np.rot90(img)

'''
img = flip_and_rotate(img)
img = cv2.resize(img, (128, 32))
img = img.reshape((128, 32, 1))
'''
img = img / 255.0
plt.imshow(img, cmap='gray')
plt.show()
imgs = np.expand_dims(img, 0)

result = model.predict(imgs, batch_size=imgs.shape[0])
result_texts = decode_batch_predictions(result)
# class_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# print(class_labels[np.argmax(result)])
print(result_texts)