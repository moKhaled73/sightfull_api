from fastapi import APIRouter, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io



class DTWLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size: int = 32, reduction=tf.keras.losses.Reduction.AUTO, name="dtw_loss"):
        super(DTWLoss, self).__init__(name=name, reduction=reduction)  # ✅ Accept reduction argument
        self.batch_size = batch_size

    def call(self, y_true, y_pred):
        tmp = []
        for item in range(self.batch_size):
            tf.print(f'Working on batch: {item}')
            s = y_true[item, :]
            t = y_pred[item, :]
            n, m = len(s), len(t)
            dtw_matrix = tf.fill([n + 1, m + 1], np.inf)
            dtw_matrix = tf.tensor_scatter_nd_update(dtw_matrix, [[0, 0]], [0.0])

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = tf.abs(s[i - 1] - t[j - 1])
                    last_min = tf.reduce_min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
                    dtw_matrix = tf.tensor_scatter_nd_update(dtw_matrix, [[i, j]], [cost + last_min])

            tmp.append(dtw_matrix[n, m])

        return tf.reduce_mean(tmp)

    def get_config(self):  # ✅ Add get_config() for saving
        config = super().get_config()
        config.update({"batch_size": self.batch_size})
        return config


def generate_raw_predicted_results(prediction, width, height, image_name="img1.jpg", username="test"):
    scanpath = []

    for row in prediction:  # Each row contains [x, y, timestamp]
        x_scaled = width * row[0]   # Scale x coordinate
        y_scaled = height * row[1]  # Scale y coordinate
        timestamp = row[2]          # Keep timestamp as is

        # Append to scanpath as a dictionary
        scanpath.append([
             image_name,
             width,
             height,
             username,
             x_scaled,
             y_scaled,
             timestamp
        ])

    return scanpath

def subsample_seq(scanpath):
    final_scanpth = []
    i = 0
    for row in scanpath:
        if i % 2 == 1 and i < 30:
            final_scanpth.append(row)
        i += 1
    return final_scanpth


scanpath_weights = 'models/scanpath_model.h5'
scanpath_model = tf.keras.models.load_model(scanpath_weights, custom_objects={"DTWLoss": DTWLoss})

router = APIRouter()

# scanpath route
@router.post("/")
async def generate_scanpath(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 🔹 Store original width & height
        original_width, original_height = img.size

        # Resize the image to 224x224 before converting to numpy
        img = img.resize((224, 224))  
        img = np.array(img)  # Convert to numpy array

        # Ensure the image has the correct shape (224, 224, 3)
        height, width, channel = img.shape  
        
        # Expand dimensions to match model input (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.vgg19.preprocess_input(img)  # Apply preprocessing

        noise  = np.random.normal(0,3,img.shape)
        noisy_img = img + noise

        prediction = scanpath_model.predict(noisy_img)  

        prediction = prediction.reshape(32, 3)

        prediction = np.array(prediction, dtype=np.float32)

        scanpath = generate_raw_predicted_results(prediction, original_width, original_height)

        final_scanpath = subsample_seq(scanpath)

        # 🔹 Convert NumPy values to Python float before returning
        final_scanpath = [[str(item) if isinstance(item, str) else float(item) for item in row] for row in final_scanpath]
        
        return {
            "scanpath" : final_scanpath,
        }    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

