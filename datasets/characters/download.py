import collections
import json
import re
import shutil
import time

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/60.0.3112.113 Safari/537.36"
}

with open('characters.json') as f:
    characters = json.load(f)

start = time.time()

num_characters_with_image = 0
failed = 0
responses = collections.Counter()
filetypes = collections.Counter()
downloaded = set()
duplicates = 0
for i, character in enumerate(characters):
    image = character['stats']['image']
    match = re.match(r"(https?://)?([/.\w\s-]*)\.(jpg|gif|png|jpeg|bmp|webp)", image)
    if match:
        if image in downloaded:
            print(f"Skipping duplicate {image}...\n")
            duplicates += 1
            continue
        else:
            downloaded.add(image)
        print(f"Getting {image}...")
        try:
            img = requests.get(image, stream=True, headers=HEADERS)
        except Exception as e:
            print(f"Failed: {e}")
            failed += 1
            responses[None] += 1
            print()
            continue
        print(img.status_code)
        responses[img.status_code] += 1
        if img.status_code == 200:
            num_characters_with_image += 1
            filetypes[match.group(3)] += 1
            img.raw.decode_content = True
            with open(f'raw/{num_characters_with_image}.{match.group(3)}', 'wb') as out_file:
                shutil.copyfileobj(img.raw, out_file)
        else:
            failed += 1
        print()
    elif image:
        print(f"Unknown image URL: {image}")
        print()

end = time.time()
print(f"Done! Downloaded {num_characters_with_image} images with {failed} failures and {duplicates} duplicates "
      f"in {end - start}s.")
print(responses)
print(filetypes)
