import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import celeba, ensure_exists


class CelebADCGAN:
    def __init__(self, img_rows, img_cols, img_channels, noise_size, generator=None, discriminator=None):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = img_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noise_size = noise_size

        optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

        if discriminator is None:
            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='binary_crossentropy',
                                       optimizer=optimizer,
                                       metrics=['accuracy'])
        else:
            self.discriminator = discriminator  # load model from saved

        if generator is None:
            # Build and compile the generator
            self.generator = self.build_generator()
            self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        else:
            self.generator = generator

        # Build and compile the combined model
        self.combined = self.build_combined()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    @classmethod
    def load(cls, path):
        path = path.rstrip('/')
        with open(f"{path}/config.json") as f:
            config = json.load(f)
        generator = tf.keras.models.load_model(f"{path}/g.h5")
        discriminator = tf.keras.models.load_model(f"{path}/d.h5")
        return cls(config['rows'], config['cols'], config['chans'],
                   config['noise_size'], generator=generator, discriminator=discriminator)

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
        self.generator.save(f"{path}/g.h5")
        self.discriminator.save(f"{path}/d.h5")
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

            self.generator.summary(print_fn=write_to_summary_file)
            self.discriminator.summary(print_fn=write_to_summary_file)
            self.combined.summary(print_fn=write_to_summary_file)
        tf.keras.utils.plot_model(self.discriminator, to_file=f"{path}/d.png")
        tf.keras.utils.plot_model(self.generator, to_file=f"{path}/g.png")

    def build_generator(self):
        # Use ConvTranspose to generate images
        noise_shape = (self.noise_size,)
        noise = tf.keras.layers.Input(shape=noise_shape, name="noise")

        hid = tf.keras.layers.Dense(8 * 8 * 512, activation='relu', input_shape=noise_shape)(noise)
        hid = tf.keras.layers.Reshape((8, 8, 512))(hid)
        # (None, 8, 8, 512)

        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('relu')(hid)
        # (None, 16, 16, 256)

        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('relu')(hid)
        # (None, 32, 32, 128)

        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('relu')(hid)
        # (None, 64, 64, 64)

        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('relu')(hid)
        # (None, 128, 128, 32)

        img = tf.keras.layers.Conv2D(self.channels, kernel_size=3, padding='same', activation='tanh')(hid)
        # (None, 128, 128, 3)

        model = tf.keras.models.Model(inputs=noise, outputs=img)
        model.summary()
        return model

    def build_discriminator(self):
        img = tf.keras.layers.Input(shape=self.img_shape, name="image")

        # Use convolutional network to improve discriminator
        hid = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same')(img)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
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

        model = tf.keras.models.Model(inputs=img, outputs=[validity])
        model.summary()
        return model

    def build_combined(self):
        # Required compiled G and D
        z = tf.keras.layers.Input(shape=(self.noise_size,), name="noise")
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        return tf.keras.models.Model(inputs=z, outputs=valid)

    def train(self, x, epochs, batch_size=128, sample_interval=50, sample_path="samples/unknown",
              starting_epoch=0, save_interval=5000):
        ensure_exists(sample_path)

        half_batch = int(batch_size / 2)
        exp_replay = []

        epoch = starting_epoch
        while epoch < epochs:
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, x.shape[0], half_batch)
            imgs = x[idx]

            # Generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, self.noise_size))
            gen_imgs = self.generator.predict(noise)

            # save one random generated image for experience replay
            r_idx = np.random.randint(0, half_batch)
            exp_replay.append(gen_imgs[r_idx])

            # Train the discriminator
            # If we have enough points, do experience replay
            if len(exp_replay) == half_batch:
                generated_images = np.array(exp_replay)
                d_loss_replay = self.discriminator.train_on_batch(generated_images,
                                                                  np.zeros((half_batch, 1)))
                exp_replay = []
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,
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
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

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
    gan = CelebADCGAN(img_rows=128, img_cols=128, img_channels=3, noise_size=100)
    # gan = CelebADCGAN.load("temp/136000")

    x = celeba()

    gan.train(x, epochs=200001, batch_size=64, sample_interval=200,
              sample_path="samples/celeba", save_interval=2000)
    gan.save("models/celeba_200k")
