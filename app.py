import cv2
from time import time

import face_module.faceDetection
import face_module.identification
import face_module.identification.face_id

import os

cam = cv2.VideoCapture(0) 


# Pegando modelo de identificação
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
id_recognition_model_path=os.path.join(current_dir,r"face_module\identification\modelos\id_recognition_model.xml")
print(id_recognition_model_path)
recognizer=face_module.identification.face_id.get_recognizer(id_recognition_model_path)

#pegando labels de usuarios
labels_path=os.path.join(current_dir,'face_module\identification\labels_ids.json')
labels=face_module.identification.face_id.load_labels(labels_path)

# Lista de imagens para trakear piscadas
liveness_images={}  ####{'usuario 1':[img01,img02....]}
#   Numero de imagens salvas
image_treshold=24

while True:
    # Ler o frame da camera
    #Se o video finalizar recomeçar
    if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get( 
            cv2.CAP_PROP_FRAME_COUNT): 
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0) 
  
    else: 
        ret, frame = cam.read()

        
        #Prepara a imagem
        ### print(imagem.shape)
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces=face_module.faceDetection.alinha_face(gray)
    

        #Para cada uma das faces
        for face in faces:
            # faces.append({'image':cropped, 'coordinates':{'x':x,'y':y,'w':w,'h':h}})
            x=face['coordinates']['x']
            y=face['coordinates']['y']
            w=face['coordinates']['w']
            h=face['coordinates']['h']
            #Desenha um frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            #identifica pessoas
            id=face_module.identification.face_id.recognize_face(recognizer,face['image'],label=labels)
            name=id['name']
            confidence=id['confidence']
            live=True
            cv2.putText(frame, f"{name} - Confidence: {round(confidence,0)} Live: {live}", (x-int(w*0.5), y-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 0), 2) 

       
        # Display the frame with face detection
        cv2.imshow('Capture Faces', frame)
 
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def append_and_truncate(lst, item,size):
    lst.append(item)
    if len(lst) > size:
        del lst[0]