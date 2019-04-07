import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ganbase import GANBase
from utils import ensure_exists


class CDCGAN(GANBase):
    def __init__(self, img_rows, img_cols, img_channels, img_label_size, noise_size, generator=None,
                 discriminator=None):
        """
        A conditional GAN.
        :param img_rows: The number of rows of the image.
        :param img_cols: The number of cols of the image.
        :param img_channels: The number of channels of the image.
        :param img_label_size: The number of labels.
        :param noise_size: The dimensionality of the noise.
        :param generator: A precompiled generator model. (Will create if not passed)
        :param discriminator: A precompiled discriminator model. (Will create if not passed)
        """
        super(CDCGAN, self).__init__(img_rows, img_cols, img_channels, img_label_size, noise_size, generator,
                                     discriminator)

    def build_generator(self):
        # Use ConvTranspose to generate images
        noise_shape = (self.noise_size,)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(3 * 3 * 256, input_shape=noise_shape))
        model.add(tf.keras.layers.Reshape((3, 3, 256)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        # (None, 3, 3, 256)

        model.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        # (None, 3, 3, 128)

        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        # (None, 7, 7, 64)

        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        # (None, 14, 14, 32)

        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2DTranspose(self.channels, kernel_size=3, padding='same', activation='tanh'))
        # (None, 28, 28, 1)
        model.summary()

        noise = tf.keras.layers.Input(shape=noise_shape, name="noise")
        img_label = tf.keras.layers.Input(shape=(1,), dtype="int32", name="label")

        # incorporate label by multiplying noise data by embedding
        label_embedding = tf.keras.layers.Embedding(self.img_label_size, self.noise_size)(img_label)
        label_embedding = tf.keras.layers.Flatten()(label_embedding)
        augmented_noise = tf.keras.layers.Multiply()([noise, label_embedding])

        img = model(augmented_noise)

        return tf.keras.models.Model(inputs=[noise, img_label], outputs=img)

    def build_discriminator(self):
        # Use convolutional network to improve discriminator
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        # (None, 14, 14, 32)

        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        # (None, 8, 8, 64)

        model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        # (None, 4, 4, 128)

        model.add(tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        # (None, 4, 4, 256)
        model.add(tf.keras.layers.Flatten())

        # output
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.summary()

        img = tf.keras.layers.Input(shape=self.img_shape, name="image")
        img_label = tf.keras.layers.Input(shape=(1,), dtype="int32", name="label")

        # incorporate label by multiplying image data by embedding, then reshaping and passing to CNN
        label_embedding = tf.keras.layers.Embedding(self.img_label_size, np.prod(self.img_shape))(img_label)
        flat_img = tf.keras.layers.Flatten(input_shape=self.img_shape)(img)
        label_embedding = tf.keras.layers.Flatten()(label_embedding)
        augmented_img = tf.keras.layers.Multiply()([flat_img, label_embedding])
        augmented_img = tf.keras.layers.Reshape(self.img_shape)(augmented_img)

        validity = model(augmented_img)

        return tf.keras.models.Model(inputs=[img, img_label], outputs=[validity])

    def build_combined(self):
        # Required compiled G and D
        z = tf.keras.layers.Input(shape=(self.noise_size,), name="noise")
        img_label = tf.keras.layers.Input(shape=(1,), name="label")
        img = self.generator([z, img_label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator([img, img_label])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        return tf.keras.models.Model(inputs=[z, img_label], outputs=valid)

    def train(self, x, y, epochs, batch_size=128, save_interval=50, sample_path="samples/conditional_dcgan"):
        """
        Trains the GAN.
        :param x: The training data.
        :param y: The labels for the training data.
        :param epochs: The number of epochs to train.
        :param batch_size: The size of an epoch.
        :param save_interval: How often to save sample images.
        :param sample_path: Where to save sample images.
        """
        ensure_exists(sample_path)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, x.shape[0], half_batch)
            imgs = x[idx]
            labels = y[idx]

            # Generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, self.noise_size))
            gen_imgs = self.generator.predict({"noise": noise, "label": labels})

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch({"image": imgs, "label": labels}, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch({"image": gen_imgs, "label": labels},
                                                            np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator on random labels
            noise = np.random.normal(0, 1, (batch_size, self.noise_size))
            idx = np.random.randint(0, x.shape[0], batch_size)
            labels = y[idx]

            g_loss = self.combined.train_on_batch({"noise": noise, "label": labels}, valid_y)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_sample(epoch, sample_path)

    def save_sample(self, epoch, path):
        r, c = self.img_label_size, 5
        noise = np.random.normal(0, 1, (r * c, self.noise_size))
        labels = []
        for row in range(r):
            for col in range(c):
                labels.append(row)
        labels = np.array(labels)
        gen_imgs = self.generator.predict({"noise": noise, "label": labels})

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"{path}/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = CDCGAN(img_rows=28, img_cols=28, img_channels=1, img_label_size=10, noise_size=100)

    # Load MNIST number dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    # add channel axis
    x_train = np.expand_dims(x_train, axis=3)

    gan.train(x_train, y_train, epochs=10001, batch_size=32, save_interval=200,
              sample_path="samples/conditional_dcgan_upsampling_mnist")
    gan.save("models/conditional_dcgan_upsampling_mnist")
