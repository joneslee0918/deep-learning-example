import os
refs = os.listdir("./sample_videos/ref/v04")
refs.sort()
print(refs)
for ref_name in refs:
    print(ref_name)