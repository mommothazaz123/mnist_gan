"""
Modified GAN, based on basic GAN found at
https://github.com/eriklindernoren/Keras-GAN
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import ensure_exists


class ConditionalGAN:
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
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = img_channels
        self.img_label_size = img_label_size
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
        return cls(config['rows'], config['cols'], config['chans'], config['labels'], config['noise_size'],
                   generator=generator, discriminator=discriminator)

    def save(self, path):
        """Saves the GAN to a folder."""
        path = path.rstrip('/')
        ensure_exists(path)
        config = {
            "rows": self.img_rows,
            "cols": self.img_cols,
            "chans": self.channels,
            "labels": self.img_label_size,
            "noise_size": self.noise_size
        }
        with open(f"{path}/config.json", 'w') as f:
            json.dump(config, f)
        self.generator.save(f"{path}/g.h5")
        self.discriminator.save(f"{path}/d.h5")
        self.save_summary(path)

    def save_summary(self, path):
        path = path.rstrip('/')
        ensure_exists(path)
        with open(f'{path}/summary.txt', 'w') as f:
            def write_to_summary_file(text):
                f.write(f"{text}\n")

            self.generator.summary(print_fn=write_to_summary_file)
            self.discriminator.summary(print_fn=write_to_summary_file)
            self.combined.summary(print_fn=write_to_summary_file)
        tf.keras.utils.plot_model(self.combined, to_file=f"{path}/model.png", expand_nested=True)

    def build_generator(self):

        noise_shape = (self.noise_size,)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(256, input_shape=noise_shape))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Dense(1024))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(tf.keras.layers.Reshape(self.img_shape))
        model.summary()

        noise = tf.keras.layers.Input(shape=noise_shape, name="noise")
        img_label = tf.keras.layers.Input(shape=(1,), dtype="int32", name="label")

        # img_label = tf.keras.layers.Input(shape=(self.img_label_size,), name="label")
        # augmented_noise = tf.keras.layers.Concatenate()([noise, img_label])

        # incorporate label by multiplying noise data by embedding
        label_embedding = tf.keras.layers.Embedding(self.img_label_size, self.noise_size)(img_label)
        label_embedding = tf.keras.layers.Flatten()(label_embedding)
        augmented_noise = tf.keras.layers.Multiply()([noise, label_embedding])

        img = model(augmented_noise)

        return tf.keras.models.Model(inputs=[noise, img_label], outputs=img)

    def build_discriminator(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(512, input_shape=(np.prod(self.img_shape),)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.summary()

        img = tf.keras.layers.Input(shape=self.img_shape, name="image")
        img_label = tf.keras.layers.Input(shape=(1,), dtype="int32", name="label")

        # flat_img = tf.keras.layers.Flatten(input_shape=self.img_shape)(img)
        # img_label = tf.keras.layers.Input(shape=(self.img_label_size,), name="label")
        # augmented_img = tf.keras.layers.Concatenate()([flat_img, img_label])

        # incorporate label by multiplying image data by embedding
        label_embedding = tf.keras.layers.Embedding(self.img_label_size, np.prod(self.img_shape))(img_label)
        flat_img = tf.keras.layers.Flatten(input_shape=self.img_shape)(img)
        label_embedding = tf.keras.layers.Flatten()(label_embedding)
        augmented_img = tf.keras.layers.Multiply()([flat_img, label_embedding])

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

    def train(self, x, y, epochs, batch_size=128, save_interval=50, sample_path="gan/conditional"):
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
    gan = ConditionalGAN(img_rows=28, img_cols=28, img_channels=1, img_label_size=10, noise_size=100)

    # Load MNIST number dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    # add channel axis
    x_train = np.expand_dims(x_train, axis=3)

    gan.train(x_train, y_train, epochs=30001, batch_size=32, save_interval=200,
              sample_path="samples/embedding_conditional_mnist")
    gan.save("models/embedding_conditional_mnist")
