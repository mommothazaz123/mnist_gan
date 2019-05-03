import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import celeba_32, ensure_exists


class CelebABEGAN:
    def __init__(self, img_rows, img_cols, img_channels, h, z, n):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = img_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.h = h
        self.z = z
        self.n = n

        # for training
        self.k = 0

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build and compile the generator
        self.generator = self.build_generator()

        # Build and compile the combined model
        self.combined = self.build_combined()

    def load(self, path):
        path = path.rstrip('/')
        self.generator.load_weights(f"{path}/g.h5")
        self.discriminator.load_weights(f"{path}/d.h5")

    def save(self, path):
        """Saves the GAN to a folder."""
        path = path.rstrip('/')
        ensure_exists(path)
        self.generator.save_weights(f"{path}/g.h5", save_format='h5')
        self.discriminator.save_weights(f"{path}/d.h5", save_format='h5')
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

    def decoder(self, h):
        embedding_shape = (h,)
        noise = tf.keras.layers.Input(shape=embedding_shape, name="h", dtype="float32")

        hid = tf.keras.layers.Dense(8 * 8 * self.n, input_shape=embedding_shape)(noise)
        hid = tf.keras.layers.Reshape((8, 8, self.n))(hid)

        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 8, 8, n)

        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 16, 16, n)

        hid = tf.keras.layers.UpSampling2D((2, 2))(hid)
        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 32, 32, n)

        img = tf.keras.layers.Conv2D(self.channels, kernel_size=3, padding='same', activation='tanh')(hid)
        # (None, 32, 32, 3)

        model = tf.keras.models.Model(inputs=noise, outputs=img)
        model.summary()
        return model

    def encoder(self):
        img = tf.keras.layers.Input(shape=self.img_shape, name="image", dtype="float32")

        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(img)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 32, 32, n)

        hid = tf.keras.layers.Conv2D(2 * self.n, kernel_size=3, padding='same', strides=2)(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(2 * self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 16, 16, 2n)

        hid = tf.keras.layers.Conv2D(3 * self.n, kernel_size=3, padding='same', strides=2)(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(3 * self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(3 * self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.BatchNormalization(momentum=0.8)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 8, 8, 3n)

        hid = tf.keras.layers.Flatten()(hid)
        enc_img = tf.keras.layers.Dense(self.h, activation='tanh')(hid)

        model = tf.keras.models.Model(inputs=img, outputs=enc_img)
        model.summary()
        return model

    def autoencoder(self):
        img_in = tf.keras.layers.Input(shape=self.img_shape, name="img_in")

        hid = self.encoder()(img_in)
        img_out = self.decoder(self.h)(hid)

        return tf.keras.models.Model(img_in, img_out)

    def build_generator(self):
        m = self.decoder(self.z)
        return m

    def build_discriminator(self):
        m = self.autoencoder()
        return m

    def build_combined(self):
        # discrim should be compiled before calling this
        z = tf.keras.layers.Input(shape=(self.z,), name="noise")
        img = self.generator(z)
        valid = self.discriminator(img)
        return tf.keras.models.Model(inputs=z, outputs=valid)

    def l1loss(self, model, x, y):
        y_ = model(x)
        return tf.reduce_mean(tf.abs(y_ - y))

    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.l1loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def d_grad(self, model, real_in, real_out, gen_in, gen_out):
        with tf.GradientTape() as tape:
            real_loss = self.l1loss(model, real_in, real_out)
            gen_loss = -self.k * self.l1loss(model, gen_in, gen_out)
            loss_value = real_loss + gen_loss
        return real_loss, gen_loss, loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train(self, x, epochs, k_lambda, gamma, batch_size=16, sample_interval=50, sample_path="samples/unknown",
              starting_epoch=0, save_interval=5000):
        ensure_exists(sample_path)

        epoch = starting_epoch
        while epoch < epochs:
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, x.shape[0], batch_size)
            real = x[idx]

            # Generate a batch of new images
            noise = np.random.uniform(-1, 1, (batch_size, self.z))
            noise = noise.astype("float32")
            fake = self.generator(noise)

            # train
            d_loss_real, d_loss_gen, d_loss, d_grads = self.d_grad(self.discriminator, real, real, fake, fake)
            # print(d_loss_real)
            # print(d_loss_fake)
            self.optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

            # ---------------------
            #  Train Generator
            # ---------------------
            # Generate a batch of new images
            noise = np.random.uniform(-1, 1, (2 * batch_size, self.z))
            noise = noise.astype("float32")
            fake = self.generator(noise)

            # Train the generator on generated images?
            self.discriminator.trainable = False
            g_loss, g_grads = self.grad(self.combined, noise, fake)
            self.optimizer.apply_gradients(zip(g_grads, self.combined.trainable_variables))
            self.discriminator.trainable = True

            # ---------------------
            #  Update k
            # ---------------------
            # maintains the diversity ratio of the network - balances goals of
            # autoencoding and discriminating images
            self.k = self.k + k_lambda * (gamma * d_loss_real - g_loss)
            self.k = min(max(self.k, 0), 1)

            # ---------------------
            #  LR Decay
            # ---------------------
            if epoch % 30000 == 0:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001 * pow(0.5, epoch // 30000))

            # ---------------------
            #  Status report
            # ---------------------
            # calculate convergence factor
            m_global = d_loss + np.abs(gamma * d_loss_real - g_loss)
            # Plot the progress
            print("%d [M: %f] [D loss: %f = %f + %f] [G loss: %f] [K: %f]" % (
                epoch, m_global, d_loss.numpy(), d_loss_real.numpy(), d_loss_gen.numpy(), g_loss.numpy(),
                self.k))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_sample(epoch, sample_path, x)

            # save model
            if epoch % save_interval == 0:
                self.save(f"temp/{epoch}")
            epoch += 1

    def save_sample(self, epoch, path, x):
        r, c = 5, 5

        idx = np.random.randint(0, x.shape[0], r * c)
        real = x[idx]
        real_imgs = self.discriminator(real)

        noise = np.random.uniform(-1, 1, (r * c, self.z))
        gen_imgs = self.generator(noise)

        combined_images = self.combined(noise)

        def save(imgs, fpath):
            # Rescale images 0 - 1
            imgs = 0.5 * imgs + 0.5

            fig, axs = plt.subplots(r, c, figsize=(self.img_cols / 10, self.img_rows / 12))
            fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(imgs[cnt, :, :, :])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(fpath, dpi=100)
            plt.close()

        save(gen_imgs, f"{path}/g-{epoch}.png")
        save(real_imgs, f"{path}/d-{epoch}.png")
        save(combined_images, f"{path}/c-{epoch}.png")


if __name__ == '__main__':
    tf.enable_eager_execution()
    gan = CelebABEGAN(img_rows=32, img_cols=32, img_channels=3, h=64, z=64, n=64)

    x = celeba_32(260000)

    gan.train(x, epochs=1000001, k_lambda=0.001, gamma=0.75, batch_size=16, sample_interval=500,
              sample_path="samples/celeba_began_batchnorm_32", save_interval=5000)
    gan.save("models/celeba_began_batchnorm_32")
