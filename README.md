# leaf_classification

## Generate corrupted leaf images
To generate corrupted images: <br>
`cd corrupted_test_leaf` <br>
`python3 make_corrupted_leaf.py`  <br>
Select the type of corruptions in `make_corrupted_leaf.py` (lne 692-710) <br>
The size of the corrupted images is 2048x1024 and the images are stored under directory *corrupted_test_leaf* <br>
There are 5 severities for each corruption   

## Leaf end-classifiers
Use the architecture from [Deep Learning for Classification and Severity Estimation of Coffee Leaf Biotic Stress](https://github.com/esgario/lara2018/tree/master/classifier) <br>
Add their architecture in `end_classifiers/model.py` (line 202) and change the current model to *model_3* in `end_classifiers/model.py` (line 156) <br>
To train the model, run `python3 main.py --train_classifier True`

