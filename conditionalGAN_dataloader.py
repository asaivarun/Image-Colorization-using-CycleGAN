import scipy
from glob import glob
import numpy as np

np.random.seed(1234)

class DataLoader:
    def __init__(self, dataset_name, img_res):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
#        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        path_A = glob('./datasets/%s/%sB/*' % (self.dataset_name, data_type))
        path_B = glob('./datasets/%s/%sA/*' % (self.dataset_name, data_type))
        print("L19, LOAD DATA : PathA images count: {} PathB images count: {}".format(len(path_A), len(path_B)))
       
        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
#        path_A = np.random.choice(path_A, size=batch_size)
#        path_B = np.random.choice(path_B,size=batch_size)

        random_order=np.random.choice(len(path_A), batch_size,replace=False)
#        path_A = np.random.choice(path_A, total_samples, replace=False)
#        path_B = np.random.choice(path_B, total_samples, replace=False)
    
        path_A=np.asarray(path_A)
        path_B=np.asarray(path_B)
        path_A=path_A[random_order]
        path_B=path_B[random_order]
#        print(path_A[1:10])
#        print(path_B[1:10])


#        batch_images = np.random.choice(path, size=batch_size)
#
#        imgs = []
#        for img_path in batch_images:41
#            img = self.imread(img_path)
#            if not is_testing:
#                img = scipy.misc.imresize(img, self.img_res)
#
#                if np.random.random() > 0.5:
#                    img = np.fliplr(img)
#            else:
#                img = scipy.misc.imresize(img, self.img_res)
#            imgs.append(img)
#
#        imgs = np.array(imgs) / 127.5 - 1.
#
#        return imgs
        imgs_A = []
        imgs_B = []
        for img_A, img_B in zip(path_A, path_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

    
                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)
    
                # If training => do random flip
                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
    
                imgs_A.append(img_A)
                imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B
        
        

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob('./datasets/%s/%sB/*' % (self.dataset_name, data_type))
        path_B = glob('./datasets/%s/%sA/*' % (self.dataset_name, data_type))
        print("L73, LOAD_BATCH: PathA images count: {} PathB images count: {}".format(len(path_A), len(path_B)))
        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        random_order=np.random.choice(len(path_A), total_samples,replace=False)
#        path_A = np.random.choice(path_A, total_samples, replace=False)
#        path_B = np.random.choice(path_B, total_samples, replace=False)
    
        path_A=np.asarray(path_A)
        path_B=np.asarray(path_B)
        path_A=path_A[random_order]
        path_B=path_B[random_order]
#        print(path_A[1:10])
#        print(path_B[1:10])
        for i in range(self.n_batches - 1):
            batch_A = path_A[i * batch_size:(i + 1) * batch_size]
            batch_B = path_B[i * batch_size:(i + 1) * batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A) / 127.5 - 1.
            imgs_B = np.array(imgs_B) / 127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img / 127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)


#Reference : https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/data_loader.py
