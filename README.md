# Malaria-Cell-Image-Classification
### Classifying Malaria infecting based on red blood cell images.
### My CNN has a 92% classification Accuracy

### Data Acquisition and Prep.

The cell images used in this case are from the U.C Nation Library of Medicine. https://lhncbc.nlm.nih.gov/publication/pub9932.

Initially the Data was seperated into folders as parasitised and uninfected. Each containing 13,780 images. The images where not labeled so i wrote a python script to rename the images as paras.#.jpg(parasitised) or uninf.#.jpg(uninfected) i.e-

os.rename(os.path.join(path, file), os.path.join(path, 'paras.' +str(i)+'.jpg'))
i = i+1

Then I seperated the folders into train and test folders. Train containing 13,000 of each parasitised and uninifected cell images totaling 26,000 cell images. and test containing the left over 780 cell images of each folder totaling 1560 cell images.