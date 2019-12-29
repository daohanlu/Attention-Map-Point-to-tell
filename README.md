# Attention-Map-Point-to-tell
An object detection system that detects the category and location of an object that is being pointed at by the user's hand, based on the attention map calculated from a image-classification neural network.

Main programs are under cnn/Guided-Attention-Inference-Network.
crop_hands.py crops hands from a video by color masking and saves them to a folder
as pngs with metadata such as the location of the fingertip.
gt_transfer reads the VOC SBD dataset and overlays the png images of hands on top of objects of a class of interest. 

Machine Learning code based on code by github user @alokwhitewolf.
Machine Learning model inspired by [Tell Me Where to Look: Guided Attention Inference Network](https://arxiv.org/abs/1802.10171)
