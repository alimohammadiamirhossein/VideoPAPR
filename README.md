### How to use:

Follow installation instructions from: https://github.com/zvict/papr

## Blender usage:

Load the model and textures onto the blender windows application, then in the scripting tab run the camera creation.
Then adjust the main directory in the camera render file and run the render. This should give you 100 pngs per frame for 40 frames, adjust values according to your model.
Run the json creation script to create proper json data to work with the train.py, adjust threshold and data_save_path according to what data you're generating (test, val, train).
Move images to the correct folder, repeat for each of test, val, and train.

OPTIONAL: Adjust blender compositing to create depth renders and run the code again while using json_creation_depth instead of json_creation to generate depth maps for test folder.

## Video Training:

If you're running the video's first frame, please replicate the yml with 'video' as your type, and the corresponding elements to match your data set up (i.e. data path in the correct position etc.)
If you're running the subsequent frames please add the args '--outframes [your frame here]'.