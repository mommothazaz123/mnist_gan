"""
Modified MNIST GAN, based on basic GAN found at
https://github.com/eriklindernoren/Keras-GAN
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist


class GAN:
    def __init__(self, img_rows, img_cols, img_channels, img_label_size):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = img_channels
        self.img_label_size = img_label_size
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = tf.keras.layers.Input(shape=(100,), name="noise")
        img_label = tf.keras.layers.Input(shape=(self.img_label_size,), name="label")
        # augmented_noise = tf.keras.layers.Concatenate()([z, img_label])
        img = self.generator([z, img_label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        # img = tf.keras.layers.Flatten(input_shape=self.img_shape)(img)
        # augmented_img = tf.keras.layers.Concatenate()([img, img_label])
        valid = self.discriminator([img, img_label])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = tf.keras.models.Model([z, img_label], valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(256, input_shape=np.add(noise_shape, (self.img_label_size,))))
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
        img_label = tf.keras.layers.Input(shape=(self.img_label_size,), name="label")
        augmented_noise = tf.keras.layers.Concatenate()([noise, img_label])
        img = model(augmented_noise)

        return tf.keras.models.Model(inputs=[noise, img_label], outputs=img)

    def build_discriminator(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(512, input_shape=(np.prod(self.img_shape) + self.img_label_size,)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.summary()

        img = tf.keras.layers.Input(shape=self.img_shape, name="image")
        flat_img = tf.keras.layers.Flatten(input_shape=self.img_shape)(img)
        img_label = tf.keras.layers.Input(shape=(self.img_label_size,), name="label")
        augmented_img = tf.keras.layers.Concatenate()([flat_img, img_label])

        validity = model(augmented_img)

        return tf.keras.models.Model(inputs=[img, img_label], outputs=[validity])

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (x_train, y_train), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = np.expand_dims(x_train, axis=3)

        # Use one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            imgs = x_train[idx]
            labels = y_train[idx]

            # Generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 100))
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
            noise = np.random.normal(0, 1, (batch_size, 100))
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            labels = y_train[idx]

            g_loss = self.combined.train_on_batch({"noise": noise, "label": labels}, valid_y)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 10, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        labels = []
        for row in range(r):
            for col in range(c):
                label = tf.keras.utils.to_categorical(row, num_classes=self.img_label_size)
                labels.append(label)
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
        fig.savefig("gan/conditional/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN(28, 28, 1, 10)
    gan.train(epochs=30000, batch_size=32, save_interval=200)
