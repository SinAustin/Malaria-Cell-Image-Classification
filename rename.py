import os
path = '/home/i/Documents/Malaria Cell Image Classification/cell_images/tt'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, 'paras.' +str(i)+'.jpg'))
    i = i+1
