import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import numpy as np
import tensorflow as tf
import time

# Fetch the service account key JSON file contents
cred = credentials.Certificate("firebase_credentials.json")

# Initialize the Firebase app
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://temphum-4ede4-default-rtdb.firebaseio.com'
})
# Get a database reference to the root of the database
ref = db.reference('/')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="regModel.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

latest_temp = None
latest_humidity = None

while True:
    # Retrieve the latest temperature value from Firebase
    temp_ref = ref.child('Data').child('Temperature').child('value')
    # Retrieve the latest temperature value from Firebase
    new_temp_snap = temp_ref.order_by_key().limit_to_last(1).get()
    for key, value in new_temp_snap.items():
        new_temp = float(value)
         
    # Check if the latest temperature has changed
    if new_temp != latest_temp:
        latest_temp = new_temp
        print("Latest temperature reading:", latest_temp)

    # Use the temperature reading to predict humidity
    input_data = np.array([[new_temp]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    humidity_prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    print("Predicted humidity:", humidity_prediction)

time.sleep(25)