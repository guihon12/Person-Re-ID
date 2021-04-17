# Person-Re-ID

environments
- python 3.7, tensorflow 1.13.1, cuda 10.0, cudnn 7.6.4, cv2(opencv) 4.4.0.46

google drive
- https://drive.google.com/file/d/1LPseGhwLIVaGKCrJUXx7QDxTL1ocnX2h/view?usp=sharing

Training
- src/train_tripletloss.py

Evaluation
1. extract embedding vector from person images
  -> src/gen_npy.py
2. evaluate
  -> src/evaluate.py
