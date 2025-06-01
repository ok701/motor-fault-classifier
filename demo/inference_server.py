import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import torch

from flask_cors import CORS

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Initialize Flask web server for AI inference
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# ====== Hyperparameters and Model/Dataset Classes ======
import numpy as np
import pandas as pd
import torch.nn as nn

# Data preprocessing parameters
win_size = 2560
step = 1280
seq_len = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset: creates sliding window sequence samples from 2-channel CSV data
class MiniSeqDataset(torch.utils.data.Dataset):
    def __init__(self, arr, label, win_size=2560, step=1280, seq_len=8, per_sample_norm=True):
        self.per_sample_norm = per_sample_norm
        self.win_size = win_size
        self.seq_len = seq_len
        self.samples = []
        # Create overlapping windows and sequences from the input array
        wins = [arr[s:s+win_size].T for s in range(0, len(arr) - win_size + 1, step)]
        for i in range(0, len(wins) - seq_len + 1):
            seq = np.stack(wins[i:i+seq_len])
            self.samples.append((seq, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.tensor(x, dtype=torch.float32)
        # Optionally normalize each sample
        if self.per_sample_norm:
            mean = x.mean(dim=(0,2), keepdim=True)
            std  = x.std (dim=(0,2), keepdim=True) + 1e-6
            x = (x - mean) / std
        return x, torch.tensor(y, dtype=torch.long)

# CNN-LSTM model for multi-class classification of motor signals
class CNN_LSTM(nn.Module):
    def __init__(self, cnn_out=32, lstm_hidden=64, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2,16,7,padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1))
        self.lstm = nn.LSTM(cnn_out, lstm_hidden, 2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden,64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes))
    def forward(self, x):
        # x shape: (batch, seq_len, channels, win_size)
        b,s,c,w = x.size()
        feat = self.cnn(x.view(b*s,c,w)).squeeze(-1)     # (b*s, 32)
        lstm_out,_ = self.lstm(feat.view(b,s,-1))
        return self.fc(lstm_out[:,-1])


# ====== Load trained model (load once and reuse for all requests) ======
model_path = './model/best_cnn_lstm.pth'
model = CNN_LSTM(num_classes=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# ====== Inference function: gets the latest uploaded CSV and predicts its class ======
def get_ai_model_prediction():
    # Find the most recently uploaded CSV file
    file_list = os.listdir(UPLOAD_FOLDER)
    if not file_list:
        return -1
    file_list = [f for f in file_list if f.endswith('.csv')]
    if not file_list:
        return -1
    file_list.sort(key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)
    file_path = os.path.join(UPLOAD_FOLDER, file_list[0])

    # Read CSV file and create dataset for model input
    test_arr = pd.read_csv(file_path, header=None).values
    test_ds = MiniSeqDataset(test_arr, label=0, win_size=win_size, step=step, seq_len=seq_len, per_sample_norm=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    # Predict class for each sequence in the file
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_ld:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
    all_preds = np.array(all_preds)

    # Return most common predicted class (majority vote), or -1 if no predictions
    if len(all_preds) == 0:
        return -1
    most_common = np.bincount(all_preds).argmax()
    print('Estimated Value:', most_common)
    return int(most_common)


# ====== Flask HTTP routes ======
# API: Predict motor status by running model inference on uploaded file
@app.route('/predict_motor_status', methods=['GET'])
def predict_motor_status():
    status_value = get_ai_model_prediction()
    return jsonify({"motorStatus": status_value})

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper to check allowed file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page: file upload form (HTML)
@app.route('/')
def index():
    return render_template('upload.html')

# Endpoint for manual file upload via form
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return '파일 부분이 없습니다.'
    file = request.files['file']
    if file.filename == '':
        return '선택된 파일이 없습니다.'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return f'파일 "{filename}"이 성공적으로 업로드되었습니다.'
    return '허용되지 않는 파일 형식입니다.'

# Endpoint for automated file upload (e.g., from frontend script)
@app.route('/upload_auto', methods=['POST'])
def upload_file_auto():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        print(f"자동 업로드 완료: {save_path}") # Log upload in server console
        return jsonify({"message": f"File '{filename}' successfully uploaded automatically"}), 200
    else:
        return jsonify({"error": "File type not allowed or no file provided"}), 400

# Start Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)