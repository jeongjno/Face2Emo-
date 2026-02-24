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
Input Image
   ↓
Face Detection
   ↓
EfficientNet-B2 (FER)
   ↓
Emotion Probability (7-class softmax)
   ↓
Condition Score (Heuristic Mapping)
   ↓
Personal Color Extraction (CIE LAB color space)
   ↓
Color Synthesis

## Key Components
- Backbone: EfficientNet-B2
- Pretraining: AffectNet
- Landmark: MediaPipe (478 pts)
- Color Space: CIE LAB
- Clustering Algorithm: K-Means

# 3. Dataset
- FER2013
- RAF-DB
- etc

## Data Cleaning
- Perceptual hashing → 중복 이미지 제거
- MTCNN face confidence < 0.9 제거

# 4. Fianl Model

## 1. ModelArchitecture
- Compound Scaling Model (EfficientNet-B2, 9.9M parameters)
## 2. Transfer Learning
- AffectNet pre-trained
- 2-phase training
## 3. Optimization & Regularization
- Weighted Cross Entropy
- Label Smoothing

# 5. Color Matching
# 6. Results
# 7. Demo
# 8. Installation
# 9. Folder Structure
