# 📚 Face2Emo: <br> 안면 특징 추출을 통한 감정 인식 최적화 및 맞춤형 색상 추천

22기 박준영 | 23기 서민솔, 정준호

# 1. Overview
Face2Emo는 얼굴 이미지로부터 감정을 인식하고, 개인 고유 피부, 눈동자, 머리 톤을 반영하여 맞춤형 컬러를 추천하는 시스템입니다.

## Objectives
1. FER 성능 최적화
2. 클래스 불균형 해결
3. 감정 및 개인 신체 색상 특성을 기반으로 심리 보완 색상 도출

# 2. Architecture

## Overall Pipeline
1. Input Image
2. Face Detection
3. EfficientNet-B2 (FER)
4. Emotion Probability (7-class softmax)
5. Condition Score (Heuristic Mapping)
6. Personal Color Extraction (CIE LAB color space)
7. Color Synthesis

## Key Components
- Backbone: EfficientNet-B2
- Pretraining: AffectNet
- Landmark: MediaPipe (478 pts)
- Color Space: CIE LAB
- Clustering Algorithm: K-Means

# 3. Repository Structure
📦 Face2Emo

┣ 📂 Models

┣ 📂 datasets

┣ 📂 html

┣ 📂 images

┣ 📜 README.md

┗ 📜 requirements.txt

# 4. Dataset
- FER2013
- RAF-DB
- etc

## Data Cleaning
- Perceptual hashing → 중복 이미지 제거
- MTCNN face confidence < 0.9 제거

# 5. Installation
1. ...

# 6. Final Model

1. Model Architecture
- Compound Scaling Model (EfficientNet-B2, 9.9M parameters)
2. Transfer Learning
- AffectNet pre-trained
- 2-phase training
3. Optimization & Regularization
- Weighted Cross Entropy
- Label Smoothing

# 7. Color Matching
1. Emotional Color
- Softmax 확률 기반 Top-3 blending
- 감정-색 매핑

2. Personal Color
- MediaPipe FaceLandmarker (478 pts)
- Skin / Hair / Iris 영역 추출
- RGB → CIE LAB 변환
- 16-Type Grid Classification

# 8. Results
# 9. Demo

