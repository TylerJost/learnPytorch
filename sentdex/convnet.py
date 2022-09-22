import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# %matplotlib inline
REBUILD_DATA=False


# +
class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    
    catcount = 0
    dogcount = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    # This conversion is not completely necessary for CNN
                    # However, color is not really relevant to if something is a cat/dog
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # Resize
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # We want the image, as well as a ones-hot array
                    # [cat, dog]
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                # Apparently there are some weird images that cannot be loaded (?)
                except Exception as e:
                    print(path)
                    os.remove(path)
#                     print('There is an error!:')
#                     print(str(e))
                    pass
        np.random.seed(1234)
        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)
        print(f'Cats: {self.catcount}   Dogs: {self.dogcount}')

if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()
# -

training_data = np.load('training_data.npy', allow_pickle=True)
plt.imshow(training_data[1][0], cmap='gray')
plt.show()
