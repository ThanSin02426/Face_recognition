Face Recognition with LBPH and OpenCV
This project is a simple, real-time face recognition system. It uses OpenCV and the Local Binary Patterns Histograms (LBPH) algorithm to train a model that can identify individuals from a webcam feed. It can even recognise an person through image or reflection.
Prerequisites 
Python 3.x
OpenCV: pip install opencv-contrib-python
Getting Started!!

Follow these steps to set up the project and train your face recognition model.

Step 1:  Run the gather_data.py script:
This will allow you to simply lable and insert an image in the dataset.
(And use labeling wisely and in order).
The script will automatically create a new folder inside dataset/ with that person's id and capture 10 images.
Repeat for every person you want to include in your model.

Step 2: Train the Recognizer
Once you have gathered enough images, run the training script.
Execute train_model.py:
The script will process the images from your dataset folder and train the LBPH model.
After training is complete, it will save the model to a trainer.yml file 

Next Step
After training, you can proceed to the final step of real-time recognition. By running recognize.py
