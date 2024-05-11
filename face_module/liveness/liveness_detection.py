# Importing the required dependencies 
import cv2  # for video rendering 
import dlib  # for face and landmark detection 
import imutils 
# for calculating dist b/w the eye landmarks 
from scipy.spatial import distance as dist 
# to get the landmark ids of the left and right eyes 
# you can do this manually too 
from imutils import face_utils 
import numpy as np
  
# from imutils import 
  
def is_live(frames,n_frames,model_path):
    """parametros
        frames 24 ultimos frames em imagem de cinza cropado no rosto
    """
    if len (frames)<n_frames:
        return False
    
    blinks=[]
    for frame in frames:
        blinks.append(is_blink(frame))

    if has_true_and_false(blinks):
        True
    else:
        False
  

def is_blink(frame:np.ndarray,model_path:str):
    """
    Check if a image has blink
    Parameters
    frame: cropped image of a person in gray
    """
    shape=frame.shape

    # Eye landmarks 
    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 

    # Variables 
    blink_thresh = 0.45
    succ_frame = 2
    count_frame = 0

    # face Detection 
    detector = dlib.get_frontal_face_detector() 
    landmark_predict = dlib.shape_predictor( model_path) 

     # landmark detection 
    shape = landmark_predict(frame, (0,0,shape[0],shape[1])) 

    # converting the shape class directly 
    # to a list of (x,y) coordinates 
    shape = face_utils.shape_to_np(shape) 

    # parsing the landmarks list to extract 
    # lefteye and righteye landmarks--# 
    lefteye = shape[L_start: L_end] 
    righteye = shape[R_start:R_end] 

    # Calculate the EAR 
    left_EAR = calculate_EAR(lefteye) 
    right_EAR = calculate_EAR(righteye) 

    # Avg of left and right eye EAR 
    avg = (left_EAR+right_EAR)/2
    if avg < blink_thresh: 
        True
    else: 
        False
       

def has_true_and_false(lst):
    return any(lst) and any(not item for item in lst)


# defining a function to calculate the EAR 
def calculate_EAR(eye): 
  
    # calculate the vertical distances 
    y1 = dist.euclidean(eye[1], eye[5]) 
    y2 = dist.euclidean(eye[2], eye[4]) 
  
    # calculate the horizontal distance 
    x1 = dist.euclidean(eye[0], eye[3]) 
  
    # calculate the EAR 
    EAR = (y1+y2) / x1 
    return EAR 
  

if __name__=='__main__':
    model_path= 'shape_predictor_68_face_landmarks.dat'
    blink_image=cv2.imread()
    not_blink_image=cv2.imread()
    is_blink(blink_image)
    is_blink(not_blink_image)


