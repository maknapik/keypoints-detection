# keypoints-detection

### Directory structure
__Keypoints__ directory should be placed in main directory of the project (where __model*.py__ files are located).

### Scripts running
To start training for given model run the script by command: __./model1.py__. The same rules concern another models.

### Models
There are following models available:
- model1 (the simplest one)
- model2 (first convolution approach)
- model3 (data generator)
- model4 (learning rate scheduler)
- model5 (dropout added)
- model6 (more neurons in the end)
- model7 (early stopping)
- model8 (knowledg transfer)

- modelX (another convolution model)
- modelXX (convolution model with data generator, scheduler and early stopping)

### Utilities
__./visualize.py__ - plot 16 images with predicted keypoints. To choose model edit the code at line 31.
