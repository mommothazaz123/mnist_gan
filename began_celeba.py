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
        self.k = tf.Variable(0., trainable=False)

        self.optimizer = None

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build and compile the generator
        self.generator = self.build_generator()

    def load(self, path):
        path = path.rstrip('/')
        input(f"You are loading weights from {path}. Press enter to continue.")
        with open(f"{path}/config.json") as f:
            self.k = tf.Variable(json.load(f)['k'], trainable=False)
        self.generator.load_weights(f"{path}/g.h5")
        self.discriminator.load_weights(f"{path}/d.h5")

    def save(self, path):
        """Saves the GAN to a folder."""
        path = path.rstrip('/')
        ensure_exists(path)
        with open(f"{path}/config.json", 'w') as f:
            json.dump({"k": float(self.k.numpy())}, f)
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
        tf.keras.utils.plot_model(self.discriminator, to_file=f"{path}/d.png")
        tf.keras.utils.plot_model(self.generator, to_file=f"{path}/g.png")

    def decoder(self, h):
        embedding_shape = (h,)
        noise = tf.keras.layers.Input(shape=embedding_shape, dtype="float32")

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
        img = tf.keras.layers.Input(shape=self.img_shape, dtype="float32")

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

    def l1loss(self, x, y):
        return tf.reduce_mean(tf.abs(y - x))

    def train(self, x, epochs, k_lambda, gamma, batch_size=16, sample_interval=50, sample_path="samples/unknown",
              starting_epoch=0, save_interval=5000):
        ensure_exists(sample_path)

        epoch = starting_epoch
        steps_per_epoch = len(x) // batch_size
        decay_every = 16000
        initial_lr = 0.0001

        step = tf.Variable(0)

        lr = initial_lr * pow(0.5, epoch // decay_every)
        print(f"Initialized initial learning rate at {lr}")
        self.optimizer = tf.train.AdamOptimizer(lr)

        # set up train operations
        # dataset
        def datagen():
            while True:
                idx = np.random.randint(0, x.shape[0], batch_size)
                imgs = x[idx]
                yield imgs

        dataset = tf.data.Dataset.from_generator(datagen, tf.float32, tf.TensorShape(
            (batch_size, self.img_rows, self.img_cols, self.channels))).repeat()

        # generator
        noise = tf.random_uniform((x.shape[0], batch_size), -1.0, 1.0)
        fake = self.generator(noise)
        g_loss = self.l1loss(fake, self.discriminator(fake))

        # discriminator
        real = dataset.get_next()
        d_loss_real = self.l1loss(real, self.discriminator(real))
        d_loss_gen = g_loss
        d_loss = d_loss_real - self.k * d_loss_gen

        # training
        g_opt = self.optimizer.minimize(g_loss, var_list=self.generator.trainable_variables)
        d_opt = self.optimizer.minimize(d_loss, global_step=step, var_list=self.discriminator.trainable_variables)

        # updating
        balance = gamma * d_loss_real - g_loss
        measure = d_loss_real + tf.abs(balance)
        with tf.control_dependencies([g_opt, d_opt]):
            k_update = tf.assign(self.k, tf.clip_by_value(self.k + k_lambda * balance, 0, 1))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            while epoch < epochs:
                result = sess.run({
                    "k_update": k_update,
                    "m": measure,
                    "g_loss": g_loss,
                    "d_loss_real": d_loss_real,
                    "d_loss_fake": d_loss_gen,
                    "d_loss": d_loss,
                    "k": self.k
                })

                # ---------------------
                #  LR Decay
                # ---------------------
                if epoch % decay_every == 0:
                    lr = initial_lr * pow(0.5, epoch // decay_every)
                    print(f"Decaying learning rate to {lr}")
                    self.optimizer = tf.train.AdamOptimizer(lr)

                # ---------------------
                #  Status report
                # ---------------------
                # calculate convergence factor
                # Plot the progress
                print("%d [M: %f] [D loss: %f = %f + %f] [G loss: %f] [K: %f]" % (
                    epoch, result['m'], result['d_loss'], result['d_loss_real'], result['d_loss_fake'],
                    result['g_loss'], result['k']))

                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    self.save_sample(epoch, sample_path, x, sess)

                # save model
                if epoch % save_interval == 0:
                    self.save(f"temp/{epoch}")
                epoch += 1

    def save_sample(self, epoch, path, x, sess):
        r, c = 5, 5

        idx = np.random.randint(0, x.shape[0], r * c)
        real = x[idx]
        real_imgs = self.discriminator(real)

        noise = np.random.uniform(-1., 1., (r * c, self.z))
        noise = noise.astype("float32")
        gen_imgs = self.generator(noise)

        combined_images = self.discriminator(gen_imgs)

        def save(imgs, fpath):
            # Rescale images 0 - 1
            # imgs = 0.5 * imgs + 0.5
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

        save(sess.run(gen_imgs), f"{path}/g-{epoch}.png")
        save(sess.run(real_imgs), f"{path}/d-{epoch}.png")
        save(sess.run(combined_images), f"{path}/c-{epoch}.png")


if __name__ == '__main__':
    gan = CelebABEGAN(img_rows=32, img_cols=32, img_channels=3, h=64, z=64, n=64)
    # gan.tests()

    x = celeba_32(50000, True)

    gan.train(x, epochs=1000001, k_lambda=0.001, gamma=0.5, batch_size=16, sample_interval=500,
              sample_path="samples/celeba_began_32_2", save_interval=5000)
    gan.save("models/celeba_began_32_2")
