# PrivateAttend: System & Research Specification

**Project Title**: Consent-Aware On-Device Biometric Attendance with Open-Set Recognition, Liveness Detection and Privacy-Preserving Templates  
**Architecture Classification**: High-Security On-Device Biometric Edge System  
**Document Status**: Specification & Theoretical Foundation (Pre-Dataset Release)

---

## 1. Problem Statement

Automated facial biometric attendance systems deployed in educational institutions and enterprise environments present severe vulnerabilities across four critical operational dimensions:

1. **Vulnerability to Presentation Attacks (Spoofing)**: Conventional 2D face recognition systems are vulnerable to physical presentation attacks, including high-resolution printed photographs, digital tablet/smartphone screen replays, and cut-out mask attacks.
2. **The Closed-Set Fallacy**: Commercial facial recognition pipelines frequently operate under a naive "closed-set" assumption, forcibly mapping any detected face to the nearest enrolled identity regardless of similarity quality, leading to catastrophic false attendances for unenrolled individuals or visitors.
3. **Biometric Template Insecurity & Irreversibility Violation**: Facial embeddings stored in plaintext in centralized relational databases violate fundamental data protection standards. Unlike passwords or cryptographic tokens, compromised biometric traits cannot be reissued or revoked if stolen.
4. **Consent & Regulatory Non-Compliance**: Modern privacy regulations (e.g., GDPR Articles 7 and 9, FERPA, CCPA) classify biometric vectors as Special Category Data requiring explicit informed consent, non-coercive alternative attendance options, auditable lifecycle tracking, and instantaneous data purging upon consent revocation.

---

## 2. Research Motivation & Theoretical Foundations

Biometric verification differs fundamentally from standard image classification. In attendance monitoring:
- Enrolled individuals form a dynamic gallery $\mathcal{G} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ of known identities.
- The operational probe distribution contains both known subjects $\mathcal{Y}_{\text{known}}$ and an infinite open universe of unknown subjects $\mathcal{Y}_{\text{unknown}}$.
- Stored representations must balance matching fidelity in latent angular space against cryptographic confidentiality in physical storage.

### 2.1 Deep Metric Learning via Additive Angular Margin (ArcFace)
Rather than cross-entropy loss over a closed class set, feature extraction leverages hyperspherical metric learning:
$$\mathcal{L}_{\text{ArcFace}} = -\frac{1}{M} \sum_{i=1}^M \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos\theta_j}}$$
where:
- $s$ is the hypersphere radius scale factor (typically $s = 64.0$).
- $m$ is the additive angular margin penalty (typically $m = 0.50$ radians).
- $\theta_j$ is the angle between weight vector $\mathbf{w}_j$ and normalized feature embedding $\mathbf{e}_i = \frac{f(\mathbf{x}_i)}{\|f(\mathbf{x}_i)\|_2} \in \mathbb{R}^{512}$.

On-device inference projects normalized facial crops into the unit hypersphere $\mathbb{S}^{511}$, ensuring that cosine similarity directly measures geodesic distance on the manifold:
$$\text{Sim}(\mathbf{e}_1, \mathbf{e}_2) = \mathbf{e}_1^\top \mathbf{e}_2 = \cos(\theta)$$

### 2.2 Open-Set Decision Boundaries
Traditional closed-set systems assign $\hat{y} = \arg\max_{j} \text{Sim}(\mathbf{e}, \mathbf{T}_j)$. In PrivateAttend, an explicit open-set decision partition is enforced:
$$\mathcal{D}(\mathbf{e}, \mathcal{G}) = \begin{cases} 
\text{KNOWN}(j^*) & \text{if } S^* \ge \theta_{\text{known}} \text{ and } (S^* - S_2^*) \ge \delta_{\text{margin}} \\
\text{UNCERTAIN} & \text{if } \theta_{\text{unknown}} \le S^* < \theta_{\text{known}} \text{ or } (S^* - S_2^*) < \delta_{\text{margin}} \\
\text{UNKNOWN} & \text{if } S^* < \theta_{\text{unknown}}
\end{cases}$$
where:
- $S^* = \max_j \text{Sim}(\mathbf{e}, \mathbf{T}_j)$ is the top-1 similarity score.
- $S_2^*$ is the top-2 runner-up similarity score among enrolled identities.
- $\theta_{\text{known}}$ is calibrated strictly on validation data to achieve $\text{FAR} \le \alpha$ (e.g., $\alpha = 0.01$).
- $\theta_{\text{unknown}}$ establishes the rejection floor.
- $\delta_{\text{margin}}$ prevents misidentifications between closely resembling gallery identities.

---

## 3. System Architecture & Component Interactions

```
+-----------------------------------------------------------------------------------+
|                                 PrivateAttend                                     |
+-----------------------------------------------------------------------------------+
|  [Presentation Layer]                                                            |
|    - Streamlit Administrative Web Dashboard (app.py)                              |
|    - Real-Time Camera Feed / Single & Batch Image Processor                       |
+-----------------------------------------------------------------------------------+
|  [Application & API Layer]                                                        |
|    - FastAPI REST Service (api.py)                                                |
|    - JWT Administrative Auth & Role-Based Access Control                          |
+-----------------------------------------------------------------------------------+
|  [Core Biometric Engine] (face_engine.py / attendance.py)                         |
|    1. Face Detection & Alignment (modules/detection.py & preprocessing.py)         |
|       * OpenCV YuNet (cv2.FaceDetectorYN) / Haar Fallback                         |
|       * 5-Point Landmark Similarity Alignment to 112x112 canonical crop           |
|       * Laplacian Variance Blur & Illumination Quality Gates                      |
|    2. Presentation Attack Detection (liveness.py)                                 |
|       * Lightweight Texture CNN (Spatial Frequency & Chromatic Anomaly Analysis)  |
|       * Multi-Frame Sliding Window Temporal Aggregation                           |
|       * Rejection of Print 2D, Screen Replay, & Cutout Spoofs                     |
|    3. Face Recognition & Metric Extraction (modules/recognition.py & models.py)   |
|       * ArcFace 512-d Hyperspherical Normalization (||e||_2 = 1.0)                |
|       * Multi-sample Quality-Weighted Template Aggregation                        |
|    4. Open-Set Decision Engine (evaluation.py & modules/recognition.py)           |
|       * Dual-Threshold Calibration (Known / Uncertain / Unknown)                  |
+-----------------------------------------------------------------------------------+
|  [Security, Privacy & Persistence Layer]                                          |
|    1. Cryptographic Template Protector (privacy.py)                               |
|       * Symmetric Encryption at Rest: AES-256-GCM / Authenticated Fernet          |
|       * Local Master Key Management (chmod 600 keyfile / ENV var)                 |
|       * Ephemeral Decryption: Plaintext embeddings exist in memory only during    |
|         matching and are scrubbed immediately afterward                           |
|    2. Consent Management & Metadata Isolation (privacy.py & database.py)          |
|       * Relational separation between PII (Student details) and Template UUIDs    |
|       * Instantaneous Cryptographic Erasure: Revoking consent purges template     |
|    3. SQLite Persistence & Audit Logging (database.py)                            |
|       * ACID Transactions, WAL Mode, Unique Constraints                           |
|       * Append-Only Security & Operational Event Audit Trail                      |
+-----------------------------------------------------------------------------------+
```

---

## 4. Technology Stack & Operational Environment

- **Language Runtime**: Python 3.11+ (Engineered and tested on Python 3.14).
- **Core Numerical**: NumPy $\ge 1.25.0$, SciPy $\ge 1.11.0$, Pandas $\ge 2.0.0$.
- **Computer Vision**: OpenCV (`cv2`) $\ge 4.8.0$ (FaceDetectorYN & Affine Transforms).
- **Deep Learning**: PyTorch $\ge 2.1.0$, Torchvision $\ge 0.16.0$, ONNX Runtime $\ge 1.16.0$.
- **Biometric Security & Cryptography**: `cryptography` $\ge 41.0.0$ (Fernet / AES-256-GCM, PBKDF2HMAC).
- **Relational Persistence**: SQLite 3 with Write-Ahead Logging (WAL) and foreign key enforcement.
- **REST Backend**: FastAPI $\ge 0.104.0$, Starlette, Pydantic v2, Uvicorn.
- **User Interface**: Streamlit $\ge 1.28.0$.
- **Benchmarking & Testing**: Pytest $\ge 7.4.0$, Matplotlib $\ge 3.8.0$, Scikit-learn $\ge 1.3.0$, psutil.
- **Inference Profile**: Standard x86_64 CPU-first execution; optional CUDA / ONNX Runtime provider acceleration.

---

## 5. Dataset Requirements & Integration Specification

### 5.1 UMDFace Benchmark Subset Specification
PrivateAttend is pre-architected to ingest the target UMDFace subset:
- **Total Identities ($N_{\text{id}}$)**: 100 distinct human identities.
- **Samples per Identity ($K_{\text{samples}}$)**: 9 facial images per identity.
- **Total Dataset Size**: 900 images.
- **Format & Dimensions**: $112 \times 112$ pixels, 3-channel RGB, eye-aligned canonical crop.
- **Covariates**: Real-world expression variations (neutral, smile, open mouth), minor illumination shifts, and natural age progression.

### 5.2 Open-Set Partitioning Protocol
To guarantee zero data leakage and rigorous open-set evaluation:
1. **Gallery Enrolled Set ($\mathcal{G}$)**: 60 identities ($60\%$), using 5 images per identity for multi-sample template building.
2. **Known Probe Test Set ($\mathcal{P}_{\text{known}}$)**: Same 60 identities, remaining 4 images per identity ($240$ test probes).
3. **Validation Calibration Set ($\mathcal{V}_{\text{unknown}}$)**: 20 disjoint identities ($20\% \times 9 = 180$ images) used strictly to calibrate $\theta_{\text{known}}$ and $\theta_{\text{unknown}}$ at target validation FAR.
4. **Unknown Probe Test Set ($\mathcal{P}_{\text{unknown}}$)**: 20 unseen identities ($20\% \times 9 = 180$ images) strictly reserved for open-set test evaluation.

*Critical Scientific Constraint*: Test data $\mathcal{P}_{\text{unknown}}$ must NEVER be used to calibrate thresholds.

---

## 6. Liveness & Presentation Attack Detection Methodology

### 6.1 Attack Typology
The presentation attack detection (PAD) subsystem detects:
1. **Printed 2D Photographs**: Matte and glossy paper reflections, lack of micro-vascular texture, print screening artifacts.
2. **Screen Replay Attacks**: High-frequency moiré patterns, screen glare, color gamut compression from LCD/OLED mobile screens.
3. **Video Presentation Attacks**: Frame-rate jitter and physical device border reflections.

### 6.2 PAD Pipeline & Anti-Blinking Disclaimer
- **Architecture**: Lightweight Texture Convolutional Network (`LivenessNet`) processing $112 \times 112$ normalized RGB facial crops.
- **Inference**: Computes bona fide probability $p_{\text{live}} \in [0.0, 1.0]$.
- **Temporal Aggregation**: For video/camera streams, scores are aggregated across a sliding temporal window of $W$ frames:
  $$\bar{p}_{\text{live}}(t) = \frac{1}{W} \sum_{k=0}^{W-1} p_{\text{live}}(t - k)$$
  with variance check $\text{Var}(p) \le \sigma_{\text{threshold}}^2$ to ensure score stability.
- **Strict Methodological Rule**: Blinking alone is explicitly rejected as a sufficient proof of liveness because printed photos can be manipulated (cutouts) and video replays inherently feature natural blinking.

---

## 7. Privacy, Biometric Security & Regulatory Compliance

### 7.1 Cryptographic Storage of Biometric Templates
- Raw 512-dimensional float32 embeddings ($\sim 2048$ bytes) are serialized into binary representation.
- Authenticated Symmetric Encryption: Encrypted using AES-256 in CBC/GCM mode via `cryptography.fernet.Fernet` with high-entropy keys.
- Key Material: Stored outside the relational database in a dedicated key store with restricted filesystem permissions (`chmod 600`), or injected via the `PRIVATEATTEND_MASTER_KEY` environment variable.

### 7.2 Conceptual Distinction: Encryption vs Cancelable Biometrics
PrivateAttend maintains clear scientific terminology in all documentation and code:
- **Encryption at Rest**: Reversible cryptographic protection. Decrypted ephemerally in-memory during matching. Protects against database exfiltration and offline disk theft.
- **Cancelable Biometrics / Fuzzy Commitments**: Irreversible transformations (non-invertible salting or bio-hashing) allowing matching directly in the transformed domain. (Distinguished in spec; not conflated with AES).
- **Prohibited Claim**: Standard cryptographic hashing (e.g., SHA-256) is never claimed to protect embeddings, because hashing destroys metric distance relationships ($\|\mathbf{e}_1 - \mathbf{e}_2\|$), making biometric comparison impossible.

### 7.3 Consent Lifecycle & Instant Template Purging
- Every enrolled student maintains an auditable consent state: `GRANTED`, `WITHDRAWN`, `EXPIRED`.
- When consent is revoked (`WITHDRAWN`):
  1. The student's encrypted biometric template record is immediately deleted from `biometric_templates`.
  2. In-memory template caches are invalidated.
  3. An immutable audit event (`CONSENT_REVOKED_TEMPLATE_PURGED`) is logged.
  4. Subsequent attendance queries for this student are flagged, requiring manual attendance verification.

---

## 8. Attendance Verification Pipeline

```
[Camera Frame / File Input]
           |
           v
[Face Detection (YuNet / Fallback)]
           |
      Faces > 0?  ---> NO  ---> [Return: NO_FACE_DETECTED]
           |
          YES
           |
           v
[Image Quality Gates (Sharpness >= 50, Illumination in [40, 220])]
           |
      Quality Pass? ---> NO  ---> [Return: LOW_QUALITY_IMAGE]
           |
          YES
           |
           v
[Liveness Verification (LivenessNet)]
           |
     p_live >= 0.50? ---> NO  ---> [Log Security Alert: PRESENTATION_ATTACK_DETECTED]
           |
          YES (Bona Fide)
           |
           v
[Feature Extraction (ArcFace 512-d L2-Normalized)]
           |
           v
[Open-Set Matching against Active Enrolled Templates]
           |
   Match Classification:
     * KNOWN (Sim >= theta_known)
     * UNKNOWN (Sim < theta_unknown) ---> [Return: UNKNOWN_IDENTITY_REJECTED]
     * UNCERTAIN (Ambiguous Margin)   ---> [Return: MANUAL_VERIFICATION_REQUIRED]
           |
        KNOWN
           |
           v
[Verify Student Consent Status == GRANTED]
           |
     Consent Valid? ---> NO  ---> [Return: CONSENT_REVOKED_BLOCKED]
           |
          YES
           |
           v
[Duplicate Attendance Check in Target Session]
           |
      Already Marked? ---> YES ---> [Return: DUPLICATE_ATTENDANCE_IGNORED]
           |
           NO
           |
           v
[Atomic Transaction: Insert Attendance Record & Log Audit Event]
           |
           v
[Return: ATTENDANCE_CONFIRMED]
```

---

## 9. Experimental Evaluation Framework

The evaluation engine calculates industry-standard metrics:
- **Biometric Recognition**: False Match Rate (FMR / FAR), False Non-Match Rate (FNMR / FRR), Receiver Operating Characteristic (ROC), Equal Error Rate (EER), Cumulative Match Characteristic (CMC).
- **Open-Set Identification**: Known Acceptance Rate (KAR), Unknown Rejection Rate (URR), Detection and Identification Rate (DIR) vs False Alarm Rate (FAR).
- **Presentation Attack Detection (ISO/IEC 30107-3)**:
  - Attack Presentation Classification Error Rate (APCER)
  - Bona Fide Presentation Classification Error Rate (BPCER)
  - Average Classification Error Rate: $\text{ACER} = \frac{\text{APCER} + \text{BPCER}}{2}$
  - Half Total Error Rate (HTER), Area Under ROC (ROC-AUC).
- **Privacy & System Benchmarking**:
  - Encryption and decryption latency per template ($\mu s$).
  - Ciphertext vs plaintext storage overhead ($\%$ expansion).
  - Component-level latency breakdown (Detection, Alignment, PAD, Embedding, Matching, Cryptography, Database).
  - Peak memory footprint (MB) and throughput (frames per second).

### 6 Systematic Ablation Studies:
1. Baseline Closed-Set Identification vs Open-Set Dual-Threshold Rejection.
2. Full Attendance Pipeline vs Attendance without Presentation Attack Detection.
3. Plaintext Embedding Storage vs Authenticated AES-256-GCM Template Protection.
4. Single-Image Enrollment vs Multi-Sample Quality-Weighted Average Template.
5. Static Single-Frame Liveness vs Multi-Frame Sliding Window Aggregation.
6. Zero-Retention Image Policy vs Temporary Verification Cache.
