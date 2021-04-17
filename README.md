# Person-Re-ID

environments
- python 3.7, tensorflow 1.13.1, cuda 10.0, cudnn 7.6.4, cv2(opencv) 4.4.0.46

google drive
- https://drive.google.com/file/d/1LPseGhwLIVaGKCrJUXx7QDxTL1ocnX2h/view?usp=sharing

Training
- src/train_tripletloss.py

Evaluation
- src/gen_npy.py (extract embedding vectors from person images)
- src/evaluate.py (evaluate the model and perform the re-identification)
