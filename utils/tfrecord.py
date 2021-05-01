import torch

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TFRDataloader():
    def _parse_image_function(self,example_proto):
        parsed_example = tf.io.parse_single_example(example_proto,{'image': tf.io.FixedLenFeature([], tf.string)})

        image = tf.image.decode_jpeg(parsed_example['image'], channels=3)
        image = tf.cast(image, tf.float32) / 255.0

        if self.size is not None:
            s=tf.shape(image)
            minsize=tf.minimum(s[0],s[1])
            image = tf.image.resize_with_crop_or_pad(image,minsize, minsize)
            image=tf.image.resize(image,[self.size,self.size])
        image=tf.transpose(image,[2,0,1])
        return image

    def __init__(self, path,epoch,batch,size=None):
        self.path=path
        self.size=size
        self.batch=batch
        self.epoch=epoch
        self.tfdataset = tf.data.TFRecordDataset(path)\
            .map(self._parse_image_function)\
            .repeat(epoch)\
            .prefetch(5)\
            .batch(batch)\
            .as_numpy_iterator()
    def init(self):
        self.__init__(self.path,epoch=self.epoch,batch=self.batch,size=self.size)

    def __iter__(self):
        return self

    def __next__(self):
        return torch.from_numpy(next(self.tfdataset))
    def __len__(self):
        return 202589//self.batch

if __name__=='__main__':
    from PIL import Image
    from torchvision.transforms import ToPILImage
    # tfdataset=tf.data.TFRecordDataset(path).map(_parse_image_function).as_numpy_iterator()
    path = '/home/hokusei/Downloads/celeba.tfrecord'
    for i, x in enumerate(TFRDataloader(path=path, size=128,epoch=1,batch=128)):
        print(i,x.shape)
        x=x[0]
        #ToPILImage()(x).show()
        #exit()
