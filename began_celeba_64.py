import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from began_celeba import CelebABEGAN
from utils import celeba_64, ensure_exists


class CelebABEGAN2(CelebABEGAN):
    def __init__(self, img_rows, img_cols, img_channels, h, z, n):
        super(CelebABEGAN2, self).__init__(img_rows, img_cols, img_channels, h, z, n)

    def decoder(self, h):
        embedding_shape = (h,)
        noise = tf.keras.layers.Input(shape=embedding_shape, name="h", dtype="float32")

        hid = tf.keras.layers.Dense(8 * 8 * self.n, input_shape=embedding_shape)(noise)
        h0 = tf.keras.layers.Reshape((8, 8, self.n))(hid)

        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(h0)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 8, 8, n)

        h0 = tf.keras.layers.UpSampling2D((2, 2))(h0)
        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Concatenate()([hid, h0])  # skip connection 1

        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 16, 16, n)

        h0 = tf.keras.layers.UpSampling2D((2, 2))(h0)
        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Concatenate()([hid, h0])  # skip connection 2

        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 32, 32, n)

        h0 = tf.keras.layers.UpSampling2D((2, 2))(h0)
        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Concatenate()([hid, h0])  # skip connection 3

        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 64, 64, n)

        img = tf.keras.layers.Conv2D(self.channels, kernel_size=3, padding='same')(hid)
        # (None, 64, 64, 3)

        model = tf.keras.models.Model(inputs=noise, outputs=img)
        model.summary()
        return model

    def encoder(self):
        img = tf.keras.layers.Input(shape=self.img_shape, name="image", dtype="float32")

        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(img)
        hid = tf.keras.layers.Activation('elu')(hid)

        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 64, 64, n)
        hid = tf.keras.layers.Conv2D(2 * self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)

        hid = tf.keras.layers.Conv2D(2 * self.n, kernel_size=3, padding='same', strides=2)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 32, 32, 2n)
        hid = tf.keras.layers.Conv2D(3 * self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)

        hid = tf.keras.layers.Conv2D(3 * self.n, kernel_size=3, padding='same', strides=2)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 16, 16, 3n)
        hid = tf.keras.layers.Conv2D(4 * self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)

        hid = tf.keras.layers.Conv2D(4 * self.n, kernel_size=3, padding='same', strides=2)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(4 * self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 8, 8, 4n)

        hid = tf.keras.layers.Flatten()(hid)
        enc_img = tf.keras.layers.Dense(self.h)(hid)

        model = tf.keras.models.Model(inputs=img, outputs=enc_img)
        model.summary()
        return model


if __name__ == '__main__':
    tf.enable_eager_execution()
    gan = CelebABEGAN2(img_rows=64, img_cols=64, img_channels=3, h=64, z=64, n=64)

    x = celeba_64(50000)

    gan.train(x, epochs=1000001, k_lambda=0.001, gamma=0.75, batch_size=16, sample_interval=500,
              sample_path="samples/celeba_began_64", save_interval=5000)
    gan.save("models/celeba_began_64")
