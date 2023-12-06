from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf
from PIL import Image, ImageOps  # Install pillow instead of PIL
import cv2

class VGG_CLIPS:

    def __init__(self):
        self.model = None
        self.batch_size = 8
        self.train_dataset = None

    def load_data(self):

        self.train_dataset = image_dataset_from_directory("train_images", image_size=(700, 270), batch_size=self.batch_size)
        
    def load_model(self):
        self.model = load_model('vgg16_clips.keras')
    
    def train(self):

        self.load_data()
        # size of image 700x370
        vgg = VGG16(input_shape=[700,270,3], weights='imagenet', include_top = False)

        for layer in vgg.layers:
            layer.trainable = False
        x = Flatten()(vgg.output)
        output = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs = vgg.input, outputs = output)
        self.model.compile(
            loss = 'binary_crossentropy',
            optimizer= 'rmsprop',
            metrics= ['accuracy']
        )
        self.model.fit(self.train_dataset, epochs = 5)
        # Save the weights
        self.model.save('vgg16_clips.keras')
    
    def capture_image(self):

        cap = cv2.VideoCapture(0)  # Change to 0 if it is your laptop's camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            print("Cannot open camera")
            return None
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Can't receive frame. Exiting ...")
            return None
        cap.release()  # When everything done, release the capture
        # # Convert from BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the image
        # plt.imshow(frame_rgb)
        # plt.axis('off')  # Turn off axis numbers and labels
        # plt.show()

        return frame

    def preprocess_image(self, frame):
        # Convert the captured frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Define the crop dimensions
        left = 400 
        top = 450
        right = 1100
        bottom = 720
        # size of image 700x270
        cropped_img = image.crop((left, top, right, bottom))
        # Display the image
        # plt.imshow(cropped_img)
        # plt.axis('off')  # Turn off axis numbers and labels
        # plt.show()
            # Convert the PIL Image to a numpy array
        img_array = keras.preprocessing.image.img_to_array(cropped_img)

        # Resize the image using TensorFlow's resize function
        resized_img = tf.image.resize(img_array, [700, 270])
        return resized_img

    def predict(self):
        captured_image = self.capture_image()
        cropped_image = self.preprocess_image(captured_image)
        # img = keras.preprocessing.image.load_img(cropped_image, target_size=(700, 270))
        # img_array = keras.preprocessing.image.img_to_array(cropped_image)
        img_array = tf.expand_dims(cropped_image, 0)  # Create batch axis
        prediction = self.model.predict(img_array) #(1 = dog, 0 = cat)
        if prediction >= 0.7:
            return True
        else:
            return False