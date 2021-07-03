from flask import request, render_template, redirect, url_for, flash, jsonify, session, Response
from werkzeug.utils import secure_filename
import os
from app import app

import app.CONFIG as CONFIG
from app.utils.register import Capture_Images, register_capture_images_, process_existing_images
from app.utils.inference import Infernce, inference_webcam
from app.utils.delete import delete_person
from app.utils.utils import show_dataset


@app.route('/')
def index():
    return render_template('index.html', status=session)

@app.route('/register_capture_images', methods=['GET', 'POST'])
def register_capture_images():
    if request.method == 'POST':
        name = request.form['known_face_name']
        cam = Capture_Images(name)
        return Response(register_capture_images_(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register_existing_images', methods=['GET', 'POST'])
def register_existing_images_():
    if request.method == 'POST':

        files = request.files.getlist("known_face_img")
        print(len(files), CONFIG.TRAINING_IMAGES)

        if len(files) < CONFIG.TRAINING_IMAGES:
            session['status'] = 'failed'
            flash(f"# of images required: {CONFIG.TRAINING_IMAGES} but selected only {len(files)}")
            return redirect(url_for('index'))

        for file in files:
            # change the name of the image before saving into the specified directory
            file.save(os.path.join(CONFIG.TEMP_FILES_PATH, secure_filename(file.filename)))

        session['status'] = 'warning'
        name = request.form['known_face_name']
        status = process_existing_images(name)
        if status["status"]:
            session['status'] = 'success'
            flash(status["message"])
        else:
            session['status'] = 'failed'
            flash(status["message"])
    return redirect(url_for('index'))

@app.route('/inference_webcam', methods=["GET", "POST"])
def _inference_webcam():
    if request.method == 'POST':
        cam = Infernce(resize_scale=CONFIG.RESIZE_SCALE)
        
        return Response(inference_webcam(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/delete_person', methods=["GET", "POST"])
def delete():
    if request.method == 'POST':
        name = request.form['del_face_name']
        res = delete_person(name)
        if res["status"]:
            session['status'] = 'success'
            flash(res["message"])
        else:
            session['status'] = 'failed'
            flash(res["message"])

    return redirect(url_for('index'))

@app.route('/registered_people', methods=['POST', 'GET'])
def registered_people():
    data = show_dataset()
    return jsonify(data)