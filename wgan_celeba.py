import json
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.merge import _Merge

from utils import celeba, ensure_exists


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs):
        alpha = tf.keras.backend.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class CelebAWGAN:
    def __init__(self, img_rows, img_cols, img_channels, noise_size, generator=None, critic=None):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = img_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noise_size = noise_size
        self.n_critic = 5

        optimizer = tf.keras.optimizers.RMSprop(lr=0.00005)

        if generator and critic:
            self.generator_model = generator
            self.critic_model = critic
        else:
            # Build the generator and critic
            self.generator = self.build_generator()
            self.critic = self.build_critic()

            # -------------------------------
            # Construct Computational Graph
            #       for the Critic
            # -------------------------------

            # Freeze generator's layers while training critic
            self.generator.trainable = False

            # Image input (real sample)
            real_img = tf.keras.layers.Input(shape=self.img_shape)

            # Noise input
            z_disc = tf.keras.layers.Input(shape=(self.noise_size,))
            # Generate image based of noise (fake sample)
            fake_img = self.generator(z_disc)

            # Discriminator determines validity of the real and fake images
            fake = self.critic(fake_img)
            valid = self.critic(real_img)

            # Construct weighted average between real and fake images
            interpolated_img = RandomWeightedAverage()([real_img, fake_img])
            # Determine validity of weighted sample
            validity_interpolated = self.critic(interpolated_img)

            # Use Python partial to provide loss function with additional
            # 'averaged_samples' argument
            partial_gp_loss = partial(self.gradient_penalty_loss,
                                      averaged_samples=interpolated_img)
            partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

            self.critic_model = tf.keras.models.Model(inputs=[real_img, z_disc],
                                                      outputs=[valid, fake, validity_interpolated])
            self.critic_model.compile(loss=[self.wasserstein_loss,
                                            self.wasserstein_loss,
                                            partial_gp_loss],
                                      optimizer=optimizer,
                                      loss_weights=[1, 1, 10])
            # -------------------------------
            # Construct Computational Graph
            #         for Generator
            # -------------------------------

            # For the generator we freeze the critic's layers
            self.critic.trainable = False
            self.generator.trainable = True

            # Sampled noise for input to generator
            z_gen = tf.keras.layers.Input(shape=(100,))
            # Generate images based of noise
            img = self.generator(z_gen)
            # Discriminator determines validity
            valid = self.critic(img)
            # Defines generator model
            self.generator_model = tf.keras.models.Model(z_gen, valid)
            self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    @classmethod
    def load(cls, path):
        path = path.rstrip('/')
        with open(f"{path}/config.json") as f:
            config = json.load(f)
        generator = tf.keras.models.load_model(f"{path}/g.h5")
        critic = tf.keras.models.load_model(f"{path}/d.h5")
        return cls(config['rows'], config['cols'], config['chans'],
                   config['noise_size'], generator=generator, critic=critic)

    def save(self, path):
        """Saves the GAN to a folder."""
        path = path.rstrip('/')
        ensure_exists(path)
        config = {
            "rows": self.img_rows,
            "cols": self.img_cols,
            "chans": self.channels,
            "noise_size": self.noise_size
        }
        with open(f"{path}/config.json", 'w') as f:
            json.dump(config, f)
        self.generator_model.save(f"{path}/g.h5")
        self.critic_model.save(f"{path}/d.h5")
        try:
            self.save_summary(path)
        except:
            pass

    def save_summary(self, path):
        path = path.rstrip('/')
        ensure_exists(path)
        with open(f'{path}/summary.txt', 'w') as f:
            def write_to_summary_file(text):
                f.write(f"{text}\n")

            self.generator_model.summary(print_fn=write_to_summary_file)
            self.critic_model.summary(print_fn=write_to_summary_file)
        tf.keras.utils.plot_model(self.critic_model, to_file=f"{path}/d.png")
        tf.keras.utils.plot_model(self.generator_model, to_file=f"{path}/g.png")

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = tf.keras.backend.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = tf.keras.backend.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = tf.keras.backend.sum(gradients_sqr,
                                                 axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = tf.keras.backend.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = tf.keras.backend.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return tf.keras.backend.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return tf.keras.backend.mean(y_true * y_pred)

    def build_generator(self):
        # Use ConvTranspose to generate images
        noise_shape = (self.noise_size,)
        noise = tf.keras.layers.Input(shape=noise_shape, name="noise")

        hid = tf.keras.layers.Dense(8 * 8 * 512, activation='relu', input_shape=noise_shape)(noise)
        hid = tf.keras.layers.Reshape((8, 8, 512))(hid)
        # (None, 8, 8, 512)

        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Conv2D(256, kernel_size=4, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('relu')(hid)
        # (None, 16, 16, 256)

        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Conv2D(128, kernel_size=4, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('relu')(hid)
        # (None, 32, 32, 128)

        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Conv2D(64, kernel_size=4, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('relu')(hid)
        # (None, 64, 64, 64)

        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Conv2D(32, kernel_size=4, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('relu')(hid)
        # (None, 128, 128, 32)

        img = tf.keras.layers.Conv2D(self.channels, kernel_size=4, padding='same', activation='tanh')(hid)
        # (None, 128, 128, 3)

        model = tf.keras.models.Model(inputs=noise, outputs=img)
        model.summary()
        return model

    def build_critic(self):
        img = tf.keras.layers.Input(shape=self.img_shape, name="image")

        # Use convolutional network to improve discriminator
        hid = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same')(img)
        hid = tf.keras.layers.LeakyReLU(alpha=0.2)(hid)
        hid = tf.keras.layers.Dropout(0.25)(hid)
        # (None, 64, 64, 32)

        hid = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.LeakyReLU(alpha=0.2)(hid)
        hid = tf.keras.layers.Dropout(0.25)(hid)
        # (None, 32, 32, 64)

        hid = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.LeakyReLU(alpha=0.2)(hid)
        hid = tf.keras.layers.Dropout(0.25)(hid)
        # (None, 16, 16, 128)

        hid = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.LeakyReLU(alpha=0.2)(hid)
        hid = tf.keras.layers.Dropout(0.25)(hid)
        # (None, 8, 8, 128)

        hid = tf.keras.layers.Flatten()(hid)
        validity = tf.keras.layers.Dense(1, activation='sigmoid')(hid)

        model = tf.keras.models.Model(inputs=img, outputs=validity)
        model.summary()
        return model

    def train(self, x, epochs, batch_size=128, sample_interval=50, sample_path="samples/unknown",
              starting_epoch=0, save_interval=5000):
        ensure_exists(sample_path)
        noise = None  # to make pycharm happy
        d_loss = None

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))

        epoch = starting_epoch
        while epoch < epochs:
            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, x.shape[0], batch_size)
                imgs = x[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.noise_size))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                          [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_sample(epoch, sample_path)

            # save model
            if epoch % save_interval == 0:
                self.save(f"temp/{epoch}")
            epoch += 1

    def save_sample(self, epoch, path):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.noise_size))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(12, 9.5))
        fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"{path}/{epoch}.png", dpi=100)
        plt.close()


if __name__ == '__main__':
    gan = CelebAWGAN(img_rows=128, img_cols=128, img_channels=3, noise_size=100)

    x = celeba(30000)

    gan.train(x, epochs=50001, batch_size=32, sample_interval=200,
              sample_path="samples/celeba_wgan", save_interval=2000)
    gan.save("models/celeba_wgan")
