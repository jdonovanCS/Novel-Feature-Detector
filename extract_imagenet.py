import tarfile
import os

print(os.listdir('data/imagenet/train'))
count=0
for f in [x for x in os.listdir('./data/imagenet/train/') if os.path.isdir('./data/imagenet/train/'+x)]:
    print(f)
    for f2 in os.listdir('data/imagenet/train/{}'.format(f)):
        count+=1

print(count)

    # my_tar = tarfile.open('data/imagenet/val/{}'.format(f))
    # my_tar.extractall('data/imagenet/val/{}'.format(f.split('.')[0]))
    # my_tar.close()

