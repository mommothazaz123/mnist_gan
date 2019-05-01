from PIL import Image
import glob, os

size = 32, 32

for infile in glob.glob("raw/*.jpg"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im = im.crop((0, 20, 178, 198))
    im.thumbnail(size, Image.ANTIALIAS)
    im.save("img_32/" + file.split('\\')[-1] + ".png", "PNG")
    print(infile)
