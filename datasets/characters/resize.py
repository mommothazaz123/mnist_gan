from PIL import Image
import glob, os


def resize(size, out_folder):
    fails = 0
    for infile in glob.glob("raw/*.*"):
        file, ext = os.path.splitext(infile)
        try:
            im = Image.open(infile).convert("RGBA")
        except OSError:
            print(f"Failed: {infile}")
            fails += 1
            continue

        width, height = im.size
        is_taller = height >= width
        if is_taller:
            box = (0, 0, width, width)
        else:
            box = (width / 2 - height / 2, 0, width / 2 + height / 2, height)
        im = im.crop(box)

        try:
            im.getchannel("A")
            background = Image.new("RGB", im.size, (255, 255, 255))
            background.paste(im, mask=im.split()[-1])
            im = background
        except ValueError:
            pass

        im = im.resize(size, Image.ANTIALIAS)
        im.save(f"{out_folder}/" + file.split('\\')[-1] + ".png", "PNG")
        print(infile)

    print(f"Done - {fails} fails")


resize((64, 64), 'img_64')
resize((128, 128), 'img_128')
