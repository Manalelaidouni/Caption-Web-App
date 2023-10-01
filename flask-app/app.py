from flask import Flask
from flask import request, flash, redirect, url_for, jsonify, render_template, Markup
from werkzeug.utils import secure_filename
import os
import sys
import pickle
import imghdr
import torch
from PIL import Image, ImageOps
sys.path.append("..") 

from running_wandb import yaml_load_with_wandb
from utils import load_checkpoint, inference_beam_search
from model import Encoder, Decoder



app = Flask(__name__, template_folder='../templates', static_folder='../static')
upload_folder = '../static/uploads'
app.config['UPLOAD_FOLDER'] = upload_folder 
app.secret_key = "it's a secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}



# get config yaml & setup wandb function
print('Loading config ...')
cfg = yaml_load_with_wandb("config_defaults.yaml", config_path="../configurations", use_wandb=False)



#  load saved vocab
print('Loading processing tools ...')
vb = open("../vocab.pickle",  'rb')
idx = open("../idx2token.pickle",  'rb')
vocab = pickle.load(vb)
idx2token = pickle.load(idx)


# setup models
print('Loading models ...')
device = torch.device("cpu")
encoder = Encoder(cfg, device).to(device)
decoder = Decoder(cfg, vocab, device).to(device)


# load model
print('Loading checkpoint ...')
encoder, decoder, _, _, _ = load_checkpoint(cfg, encoder, decoder, checkpoint_path="../checkpoint")
print('Loading checkpoint done ...')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            print('no')
            return redirect(request.url)
        
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            print('no selectd')
            flash('You have not submitted a file.')
            return redirect(request.url)

        # save the image in "image" folder
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
        else:
            print('Not allowed file type')
            flash('Not allowed file type, the following image formats are allowed: png, jpg, jpeg, gif.')
            return redirect(request.url)
    
        img_path = f"../static/uploads/{file.filename}" 
        caption = inference_beam_search(encoder, decoder, idx2token, vocab, cfg, device, image_path=img_path)
        return render_template('index_pred.html', model_output=caption, image_path=img_path)

    



if __name__ == '__main__':
    #app.run(debug=True, use_reloader=False)
    app.run(host= '0.0.0.0', port=5000)
