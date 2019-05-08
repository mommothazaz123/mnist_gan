import json

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

        self.optimizer = None

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build and compile the generator
        self.generator = self.build_generator()

        # Build and compile the combined model
        self.combined = self.build_combined()

    def load(self, path):
        path = path.rstrip('/')
        input(f"You are loading weights from {path}. Press enter to continue.")
        with open(f"{path}/config.json") as f:
            self.k = json.load(f)['k']
        self.generator.load_weights(f"{path}/g.h5")
        self.discriminator.load_weights(f"{path}/d.h5")

    def save(self, path):
        """Saves the GAN to a folder."""
        path = path.rstrip('/')
        ensure_exists(path)
        with open(f"{path}/config.json", 'w') as f:
            json.dump({"k": float(self.k)}, f)
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
        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 32, 32, n)

        img = tf.keras.layers.Conv2D(self.channels, kernel_size=3, padding='same')(hid)
        # (None, 32, 32, 3)

        model = tf.keras.models.Model(inputs=noise, outputs=img)
        model.summary()
        return model

    def encoder(self):
        img = tf.keras.layers.Input(shape=self.img_shape, name="image", dtype="float32")

        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(img)
        hid = tf.keras.layers.Activation('elu')(hid)

        hid = tf.keras.layers.Conv2D(self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 32, 32, n)
        hid = tf.keras.layers.Conv2D(2 * self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)

        hid = tf.keras.layers.Conv2D(2 * self.n, kernel_size=3, padding='same', strides=2)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 16, 16, 2n)
        hid = tf.keras.layers.Conv2D(3 * self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)

        hid = tf.keras.layers.Conv2D(3 * self.n, kernel_size=3, padding='same', strides=2)(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        hid = tf.keras.layers.Conv2D(3 * self.n, kernel_size=3, padding='same')(hid)
        hid = tf.keras.layers.Activation('elu')(hid)
        # (None, 8, 8, 3n)

        hid = tf.keras.layers.Flatten()(hid)
        enc_img = tf.keras.layers.Dense(self.h)(hid)

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
        z = tf.keras.layers.Input(shape=(self.z,), name="noise")
        img = self.generator(z)
        d_img = self.discriminator(img)
        return tf.keras.models.Model(inputs=z, outputs=d_img)

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
        steps_per_epoch = len(x) // batch_size
        decay_every = 16000
        initial_lr = 0.0001

        lr = initial_lr * pow(0.5, epoch // decay_every)
        print(f"Initialized initial learning rate at {lr}")
        self.optimizer = tf.train.AdamOptimizer(lr)

        while epoch < epochs:
            # ---------------------
            #  Calculate gradients on batch
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, x.shape[0], batch_size)
            real = x[idx]

            # Generate a batch of new images
            noise = np.random.uniform(-1., 1., (batch_size, self.z))
            noise = noise.astype("float32")
            fake = self.generator(noise)

            # Train the generator on generated images?
            self.discriminator.trainable = False
            g_loss, g_grads = self.grad(self.combined, noise, fake)
            self.discriminator.trainable = True

            # train
            d_loss_real, d_loss_gen, d_loss, d_grads = self.d_grad(self.discriminator, real, real, fake, fake)

            # ---------------------
            #  Apply Gradients
            # ---------------------
            self.discriminator.trainable = False
            self.optimizer.apply_gradients(zip(g_grads, self.combined.trainable_variables))
            self.discriminator.trainable = True
            self.optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

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
            if epoch % decay_every == 0:
                lr = initial_lr * pow(0.5, epoch // decay_every)
                print(f"Decaying learning rate to {lr}")

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

        noise = np.random.uniform(-1., 1., (r * c, self.z))
        noise = noise.astype("float32")
        gen_imgs = self.generator(noise)

        combined_images = self.combined(noise)

        def save(imgs, fpath):
            # Rescale images 0 - 1
            imgs = 0.5 * imgs + 0.5
            imgs = np.clip(imgs, 0, 1)

            fig, axs = plt.subplots(r, c, figsize=(self.img_cols * c / 100, self.img_rows * r / 100))
            fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, top=1, bottom=0)
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

    def tests(self):
        import copy
        self.discriminator.trainable = False
        print(self.combined.trainable_variables == self.generator.trainable_variables)

        # Generate a batch of new images
        noise = np.random.uniform(-1., 1., (16, self.z))
        noise = noise.astype("float32")
        fake = self.generator(noise)
        old_fake = copy.deepcopy(fake)

        # train
        d_loss, d_grads = self.grad(self.discriminator, fake, fake)

        print(fake == old_fake)


if __name__ == '__main__':
    tf.enable_eager_execution()
    gan = CelebABEGAN(img_rows=32, img_cols=32, img_channels=3, h=64, z=64, n=64)
    # gan.tests()

    x = celeba_32(50000)

    gan.train(x, epochs=1000001, k_lambda=0.001, gamma=0.5, batch_size=16, sample_interval=500,
              sample_path="samples/celeba_began_32_2", save_interval=5000)
    gan.save("models/celeba_began_32_2")
