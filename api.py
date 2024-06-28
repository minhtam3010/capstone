from dlibModel import DlibModel
import numpy as np
from db.conn import MongoConnection
from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
import dlib # type: ignore
from gevent.pywsgi import WSGIServer # type: ignore
import cv2 # type: ignore
import json
import redis # type: ignore

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Set the maximum request size to 32 MB

CORS(app)
dlibModel = DlibModel()

mongoConn = MongoConnection()

r = redis.Redis(
    host="redis-10826.c252.ap-southeast-1-1.ec2.redns.redis-cloud.com", port=10826,
    username="default", # use your Redis user. More info https://redis.io/docs/latest/operate/oss_and_stack/management/security/acl/
    password="qRfi4voTGJQmjk5k5wBU8FI94Ajflys3", # use your Redis password
)

users, index = mongoConn.get_all()

# constraint = 0.038
constraint = 0.042

@app.route('/verify', methods=['POST'])
def verify():
    if len(users) == 0:
        return jsonify({"status": "error", "message": "No face detected"}), 400

    print("Request received")

    # Check if 'img' part is present in the request files
    if 'img' not in request.files:
        return jsonify({"status": "error", "message": "No image provided"}), 400

    # Get the image file from the request
    image_files = request.files.getlist('img')
    if len(image_files) == 0:
        return jsonify({"status": "error", "message": "No image provided"}), 400

    # Read the image file
    img_read = image_files[0].read()

    imgS = cv2.imdecode(np.frombuffer(img_read, np.uint8), cv2.IMREAD_COLOR)

    if imgS is None:
        return jsonify({"status": "error", "message": "Invalid image"}), 400

    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    try:
        # Get the face locations
        face_locations, _ = dlibModel.getFace(imgS)
    except Exception as e:
        return jsonify({"status": "error", "message": "Error in face detection", "error": str(e)}), 500

    for face_location in face_locations:
        try:
            landmarks = dlibModel.face_landmarks(imgS, face_location)
            normalize_img = dlibModel.normalization(landmarks, new_img=imgS)
            face_chip = dlib.get_face_chip(normalize_img, landmarks)
            compared_embedding = np.array(dlibModel.face_encoder.compute_face_descriptor(face_chip), dtype=np.float32)
        except Exception as e:
            return jsonify({"status": "error", "message": "Error in processing face", "error": str(e)}), 500

        k = min(len(users), 1)
        labels, distances = index.knn_query(compared_embedding, k=k)

        for i in range(len(labels[0])):
            distance = distances[0][i]
            user = users[labels[0][i]]
            if distance < constraint:
                return jsonify({"status": "success", "message": user})

    return jsonify({"status": "error", "message": "No face detected"}), 200

@app.route('/add', methods=['POST'])
def add():
    global r
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

    r.set("update", "true")
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

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    if username == "admin" and password == "faceid":
        user = {'fullName': 'Le Minh Tam', 'gender': 'maleTeacher', 'vietnameseName': 'Lê Minh Tâm'}
        return jsonify({"status": "success", "message": user}), 200

    return jsonify({"status": "error", "message": "No face detected"}), 200

def reset():
    global users, index
    users, index = mongoConn.get_all()

if __name__ == "__main__":
    # app.run(debug=False, port=8083)
    print("Starting server...")
    http_server = WSGIServer(('', 8080), app)
    http_server.serve_forever()