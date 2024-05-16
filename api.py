from dlibModel import DlibModel
import numpy as np
from db.conn import MongoConnection
from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
import dlib # type: ignore
from gevent.pywsgi import WSGIServer # type: ignore
import cv2 # type: ignore
import json

app = Flask(__name__)

CORS(app)
dlibModel = DlibModel()

mongoConn = MongoConnection()

users, index = mongoConn.get_all()

constraint = 0.038

@app.route('/verify', methods=['POST'])
def verify():
    if len(users) == 0:
        return jsonify({"status": "error", "message": "No face detected"})
    
    print("Request received")
    image_files = []
    # Check if an 'image' part is present in the request files
    if 'img' not in request.files:
        return jsonify({"status": "error", "message": "No image provided"}), 400

    # Get the image file from the request
    image_files = request.files.getlist('img')
    if len(image_files) == 0:
        return jsonify({"status": "error", "message": "No image provided"}), 400
    # Read the image file

    img_read = image_files[0].read()

    imgS = cv2.imdecode(np.frombuffer(img_read, np.uint8), cv2.IMREAD_COLOR)

    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    

    face_locations, _ = dlibModel.getFace(imgS)

    for face_location in face_locations:
        landmarks = dlibModel.face_landmarks(imgS, face_location)

        normalize_img = dlibModel.normalization(landmarks, new_img=imgS)

        face_chip = dlib.get_face_chip(normalize_img, landmarks)

        compared_embedding = np.array(dlibModel.face_encoder.compute_face_descriptor(face_chip), dtype=np.float32)

        k = 1 
        if len(users) < 5:
            k = len(users)
        else:
            k = 5
        labels, distances = index.knn_query(compared_embedding, k=k)

        for i in range(len(labels[0])):
            distance = distances[0][i]
            print("Distance: ", distance)
            if distance < constraint:
                user = users[labels[0][i]]
                return jsonify({"status": "success", "message": user})
            
            # else:
                # return jsonify({"status": "error", "message": f"Faces are not of the same person"})
            # print("User: ", users[labels[0]])
    return jsonify({"status": "error", "message": "No face detected"})

@app.route('/add', methods=['POST'])
def add():
    if 'img' not in request.files:
      return jsonify({"status": "error", "message": "No image provided"}), 400
    # Get the image file from the request
    image_files = request.files.getlist('img')
    user = request.form['user']

    userJson = json.loads(user)

    embeddings = []
    for image in image_files:
        # Read the image file
        img_read = image.read()

        imgS = cv2.imdecode(np.frombuffer(img_read, np.uint8), cv2.IMREAD_COLOR)

        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        face_locations, _ = dlibModel.getFace(imgS)
                    
        for face_location in face_locations:
            landmarks = dlibModel.face_landmarks(imgS, face_location)

            normalize_img = dlibModel.normalization(landmarks, new_img=imgS)

            face_chip = dlib.get_face_chip(normalize_img, landmarks)

            embedding = np.array(dlibModel.face_encoder.compute_face_descriptor(face_chip))
            embeddings.append(embedding.tolist())

    mongoConn.insert(userJson, embeddings)

    balance = mongoConn.getBalance(userJson["fullName"])
    if balance is None:
        mongoConn.insertBalance(userJson["fullName"], 500000)

    reset()
    return jsonify({"status": "success", "message": "User added successfully"})

@app.route('/fine_tune', methods=['POST']) 
def fine_tune_constraint():
    global constraint
    constraint = float(request.form['constraint'])
    return jsonify({"status": "success", "message": "Constraint updated successfully"})

@app.route("/balance", methods=["POST"])
def balance():
    user = request.form["userName"]
    balance = mongoConn.getBalance(user)
    return jsonify({"status": "success", "message": balance})

@app.route("/invoice", methods=["POST"])
def invoice():
    user = request.form["userName"]
    invoices = mongoConn.getAllInvoiceOfUser(user)
    return jsonify({"status": "success", "message": invoices})

@app.route("/all_invoices", methods=["GET"])
def all_invoices():
    invoices = mongoConn.getAllInvoice()
    return jsonify({"status": "success", "message": invoices})

def reset():
    global users, index
    users, index = mongoConn.get_all()

if __name__ == "__main__":
    # app.run(debug=False, port=8083)
    print("Starting server...")
    http_server = WSGIServer(('', 8080), app)
    http_server.serve_forever()