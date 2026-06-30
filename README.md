# Face Authentication System

## Overview
Real-time biometric authentication system utilizing facial embeddings, FAISS vector similarity search, and live webcam inference for authorized identity verification.

## Features
- Real-time webcam authentication
- Facial enrollment system for first-time setup
- Face embedding generation using face_recognition
- FAISS vector similarity search
- Threshold-based authorization logic
- OpenCV visualization overlays
- Frame-skipping inference optimization for improved performance

## Tech Stack
- Python
- OpenCV
- face_recognition
- FAISS
- NumPy

## Architecture

Enrollment Phase
```
Webcam Capture
→ Face Image Storage
→ Face Embedding Generation
→ FAISS Index Creation
```

Authentication Phase
```
Webcam Frame
→ Face Detection
→ Face Embedding
→ FAISS Similarity Search
→ Authentication Decision
→ Visual Overlay
```

## Installation

Clone the repository:

```bash
git clone https://github.com/DukeOrji/face-auth.git
cd face-auth
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Start the authentication system:

```bash
python system.py
```

### First Run

If no enrolled faces are found:

1. The system automatically enters enrollment mode.
2. Five images of the user will be captured and saved.
3. Facial embeddings will be generated from the captured images.
4. Authentication mode will begin automatically.

### Authentication

- A green bounding box indicates an authorized user.
- A red bounding box indicates an unauthorized user.
- Authorization is determined using facial embedding similarity and FAISS vector search.

## Controls

| Key | Action |
|------|--------|
| P | Exit application |

## Project Status

Work in progress.

Planned features:
- Multiple authorized users
- User management interface
- Password-protected enrollment
- Access logging
- Identity labeling ("Authorized: Duke")
