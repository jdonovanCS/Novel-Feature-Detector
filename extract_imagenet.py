import tarfile
import os

print(os.listdir('data/imagenet/train'))

for f in [x for x in os.listdir('./data/imagenet/train/') if ('.tar' in x)]:
    print(f)
    my_tar = tarfile.open('data/imagenet/train/{}'.format(f))
    my_tar.extractall('data/imagenet/train/{}'.format(f.split('.')[0]))
    my_tar.close()

