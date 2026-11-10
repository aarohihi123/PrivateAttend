# PrivateAttend

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Consent-Aware On-Device Biometric Attendance with Open-Set Recognition, Liveness Detection and Privacy-Preserving Templates**

PrivateAttend is a secure, privacy-preserving biometric attendance framework engineered for institutional environments. By combining deep hyperspherical metric learning (ArcFace), presentation attack detection (PAD), calibrated open-set decision boundaries, and authenticated symmetric template encryption, PrivateAttend ensures robust student verification without compromising individual civil privacy or violating data protection regulations (such as GDPR Art. 9 & FERPA).

---

## Key Features

- **Presentation Attack Detection (PAD)**: Deep texture and frequency-domain analysis identifies and rejects 2D printed photographs, digital screen replays, and video spoofs before recognition.
- **Calibrated Open-Set Recognition**: Replaces naive closed-set nearest-neighbor matching with a rigorous dual-threshold decision partition (`KNOWN`, `UNKNOWN`, `UNCERTAIN`) preventing false enrollments of un-enrolled visitors.
- **Privacy-Preserving Biometric Templates**: Raw 512-dimensional facial embeddings are never stored in plaintext. Templates are encrypted at rest with AES-256-GCM / Fernet, decrypted ephemerally in-memory during verification, and scrubbed immediately afterward.
- **Student Metadata Separation**: Personally Identifiable Information (name, email, department) is relationally separated from biometric identifiers.
- **Consent-Aware Lifecycle & Instant Purge**: Students maintain complete control over biometric consent. Revoking consent triggers instant, verifiable deletion of biometric template records and logs an immutable audit event.
- **On-Device CPU Execution**: Highly optimized inference pipeline running on standard edge and workstation CPUs without mandatory GPU hardware.
- **Unified REST API & Interactive UI**: Fully functional FastAPI backend coupled with a multi-page Streamlit dashboard for real-time camera scanning, management, and audit inspection.

---

## System Architecture

```
[Camera Stream / Photo Upload]
               │
               ▼
   [Face Detection & Landmark Alignment] (OpenCV YuNet / Haar Fallback)
               │
               ▼
   [Image Quality Filter] (Blur, Illumination & Resolution Gates)
               │
               ▼
   [Liveness Verification] (Lightweight Texture CNN + Temporal Aggregation)
         │                     │
    [Spoof Rejection]     [Bona Fide Face]
                               │
                               ▼
   [Face Recognition Engine] (ArcFace 512-d Hyperspherical Normalization)
                               │
                               ▼
   [Open-Set Decision Boundary] (Dual-Threshold Calibration)
         ├── KNOWN (Similarity >= theta_known)
         ├── UNKNOWN (Similarity < theta_unknown) ──> Rejection
         └── UNCERTAIN (Ambiguous Margin) ─────────> Manual Review Required
                               │
                               ▼
   [Privacy & Verification Check] (Active Consent & Duplicate Session Gate)
                               │
                               ▼
   [SQLite Encrypted Database & Audit Logging]
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)
- Web camera (for live attendance mode)

### 1. Clone Repository & Setup Virtual Environment
```bash
cd PrivateAttend
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment & Master Key
PrivateAttend automatically manages encryption keys. To specify a custom master key, export the environment variable:
```bash
export PRIVATEATTEND_MASTER_KEY="your-32-byte-base64-encoded-key="
```
Or allow PrivateAttend to generate a local secure keyfile with `0600` permissions.

---

## Execution Instructions

### Launch the FastAPI Backend
Start the high-performance asynchronous REST service:
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at:  
`http://127.0.0.1:8000/docs`

### Launch the Streamlit Dashboard
Open the administrative web console:
```bash
streamlit run app.py --server.port 8501
```
Access the dashboard at `http://localhost:8501`.

---

## Dashboard Navigation

The Streamlit interface provides 9 cohesive views:
1. **Dashboard Overview**: Real-time KPI cards (enrolled students, active sessions, attendance counts) and daily attendance charts.
2. **Live Attendance**: Real-time webcam feed with live facial bounding boxes, liveness scoring, and automated attendance logging.
3. **Image-Based Attendance**: Drag-and-drop verification for single or batch photo attendance.
4. **Student Enrollment**: Student profile registration, explicit consent check, and multi-sample facial photo capture.
5. **Student Management**: Student directory with search, filter, manual attendance override, and profile deletion.
6. **Attendance History**: Session logs, record tables, and one-click CSV export.
7. **Analytics**: Confidence distributions, liveness scores, and open-set rejection statistics.
8. **Privacy and Consent**: Consent registry, instant consent revocation with template purge verification, and audit event explorer.
9. **System Settings**: Real-time threshold tuning ($\theta_{\text{known}}$, $\theta_{\text{unknown}}$, liveness threshold), detector backend selection, and health diagnostic display.

---

## API Usage Examples

### 1. Register a Student
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/students" \
     -H "Content-Type: application/json" \
     -d '{
       "student_id": "STU1001",
       "first_name": "Ada",
       "last_name": "Lovelace",
       "email": "ada@university.edu",
       "department": "Computer Science"
     }'
```

### 2. Record Biometric Consent
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/consent" \
     -H "Content-Type: application/json" \
     -d '{
       "student_id": "STU1001",
       "status": "GRANTED",
       "terms_version": "v1.0"
     }'
```

### 3. Enroll Multi-Sample Biometric Photos
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/biometrics/enroll" \
     -F "student_id=STU1001" \
     -F "files=@sample1.jpg" \
     -F "files=@sample2.jpg" \
     -F "files=@sample3.jpg"
```

### 4. Verify Attendance via Image
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/attendance/verify" \
     -F "session_id=1" \
     -F "file=@probe.jpg"
```

---

## Pretrained Weights & Fallback Modes

To run operational inference, the following weights can be placed in `weights/`:
- **Face Detector**: `weights/face_detection_yunet_2023mar.onnx` (OpenCV YuNet). If missing, the detector seamlessly falls back to OpenCV's built-in Haar Cascade detector.
- **ArcFace Extractor**: `weights/w600k_mbf.onnx` (InsightFace `buffalo_sc` recognition model; thresholds are calibrated on this model, so swapping it requires recalibration). Copy it from `~/.insightface/models/buffalo_sc/` after installing `insightface`.
- **LivenessNet**: `weights/liveness_cnn_v1.pth`.

If the ArcFace weights are missing, enrollment and verification fail with a clear `ModelNotLoadedError` (no silent synthetic embeddings). If the liveness weights are missing, liveness returns a neutral 0.50 score, so do not rely on PAD results until `liveness_cnn_v1.pth` is installed. If the YuNet weights are missing, the detector falls back to Haar, which gives weaker alignment.

---

## Dataset Integration: UMDFace Subset

PrivateAttend is built for integration with the upcoming UMDFace subset:
- **100 Identities**
- **9 Aligned Images per Identity (900 images total)**
- **112x112 Pixel Resolution**

### Ingestion Steps:
1. Place raw images in `data/raw/umd_face/`.
2. Run the preparation script:
   ```bash
   python scripts/prepare_dataset.py --input-dir data/raw/umd_face/ --output-dir data/processed/umd_face/
   ```
3. Generate disjoint evaluation manifests (Gallery, Validation Unknowns, Test Probes):
   ```bash
   python scripts/prepare_dataset.py --create-splits --gallery-size 60 --val-size 20 --test-size 20
   ```

---

## Experimental Evaluation

Run the evaluation harness to compute performance metrics and generate ablation plots:
```bash
python scripts/evaluate.py --run-all
```
Generated reports and publication-ready figures are written to:
- `results/metrics/`: JSON and CSV error rates (FAR, FRR, APCER, BPCER, ACER).
- `results/figures/`: High-resolution ROC curves, confusion matrices, and latency breakdown charts.

---

## System Limitations

1. **Illumination Requirements**: Face images with mean brightness below 40 or above 220 are rejected by the quality gate.
2. **Pose Variations**: Severe out-of-plane head rotations ($> 45^\circ$) trigger quality rejection.
3. **Untested Attack Types**: PAD is evaluated on printed paper photos and LCD/OLED replays; silicone 3D masks require specialized depth/IR sensors.
4. **Sharpness Gate**: `BLUR_THRESHOLD` (Laplacian variance, default 25) is resolution-dependent; tune it against your own capture device.
