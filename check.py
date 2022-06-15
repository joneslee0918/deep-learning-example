import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type=int, default=[216 * 2, 384 * 2], help="the image size, eg. [216,384]")
opt = parser.parse_args()
print(type(opt.image_size))

# import os
# refs = os.listdir("./sample_videos/ref/v04")
# refs.sort()
# print(refs)
# for ref_name in refs:
#     print(ref_name)