import json

import tensorflow as tf

from utils import ensure_exists


class GANBase:
    def __init__(self, img_rows, img_cols, img_channels, img_label_size, noise_size, generator=None,
                 discriminator=None):
        """
        An ABC for GANs.
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
        """To be overriden in subclasses"""
        raise NotImplemented()

    def build_discriminator(self):
        """To be overriden in subclasses"""
        raise NotImplemented()

    def build_combined(self):
        """To be overriden in subclasses"""
        raise NotImplemented()

    def train(self, x, y, epochs, batch_size=128, sample_interval=50, sample_path="samples/unknown"):
        """To be overriden in subclasses"""
        raise NotImplemented()

    def save_sample(self, epoch, path):
        """To be overriden in subclasses"""
        raise NotImplemented()
