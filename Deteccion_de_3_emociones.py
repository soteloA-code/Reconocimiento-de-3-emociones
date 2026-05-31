#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:46:24 2024

@author: arman
"""
import numpy as np
import cv2
import mediapipe as mp
import math
from math import sqrt
import time
import torch
from torch import nn

class lstm_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=20, hidden_size=1400, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(1400, 3)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

def calculate_fps(img, pTime, pos=(50, 80), color=(0, 0, 0), scale=0, thickness=4):
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        pos,
        cv2.FONT_HERSHEY_PLAIN,
        scale,
        color,
        thickness,
    )
    return pTime, img

def obtener_dos_angulos(face_landmarks, p1, p2, p3):
    x1 = face_landmarks.landmark[p1].x
    y1 = face_landmarks.landmark[p1].y
    x2 = face_landmarks.landmark[p2].x
    y2 = face_landmarks.landmark[p2].y
    x3 = face_landmarks.landmark[p3].x
    y3 = face_landmarks.landmark[p3].y
    
    lado1 = sqrt((x2-x1)**2 + (y2-y1)**2)
    lado2 = sqrt((x3-x1)**2 + (y3-y1)**2)
    lado3 = sqrt((x3-x2)**2 + (y3-y2)**2)
    
    A = (lado1**2 + lado2**2 - lado3**2)/(2*lado1*lado2)
    anguloRA = math.acos(A)
    AG = math.degrees(anguloRA)
    
    B = (lado2**2 + lado3**2 - lado1**2)/(2*lado2*lado3)
    AnguloRB = math.acos(B)    
    BG = math.degrees(AnguloRB)
    
    return AG, BG

def obtener_predicciones(model, datos):
    # Convertir los datos a un tensor de PyTorch
    datos_tensor = torch.tensor(datos, dtype=torch.float32)

    # Realizar predicciones
    with torch.no_grad():
        predicciones = model(datos_tensor.unsqueeze(0))  # Agregar dimensión de lote

    # Aplicar función de activación softmax para obtener probabilidades
    probabilidades = torch.softmax(predicciones, dim=1)

    # Obtener la etiqueta con la mayor probabilidad
    etiqueta_predicha = torch.argmax(probabilidades).item()

    return etiqueta_predicha

def show_video(num_features, emocion, video_count):
    print("\r video number: {} con etiqueta :{}".format(video_count, emocion), end='')
    print(" ")
    cap = cv2.VideoCapture(0)
    CC= []
    CC1 = []
    CC2 = []
    CC3 = []
    CC4 = []
    CC5 = []
    CC6 = []
    CC7 = []
    CC8 = []
    CC9 = []
    CC10 = []
    frame_count = 0
    emocion_texto = ""  # Variable para almacenar la emoción detectada
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    
    try:
        pTime = time.time()

        with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as face_mesh:

            while cap.isOpened():
                print("\r Frame = {}".format(frame_count), end='')
                ret, img = cap.read()
                if not ret:
                    break
                pTime, img = calculate_fps(img, pTime)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False

                results = face_mesh.process(img)

                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:

                    for face_landmarks in results.multi_face_landmarks:

                        mp_drawing.draw_landmarks(
                            image=img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                        )
                        mp_drawing.draw_landmarks(
                            image=img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                        )


                        for i in range(num_features):
                            alpha, beta = obtener_dos_angulos(face_landmarks, *points[i])
                            if i == 0:
                                CC1.append([alpha, beta])
                            if i == 1:
                                CC2.append([alpha, beta])
                            if i == 2:
                                CC3.append([alpha, beta])
                            if i == 3:
                                CC4.append([alpha, beta])
                            if i == 4:
                                CC5.append([alpha, beta])
                            if i == 5:
                                CC6.append([alpha, beta])
                            if i == 6:
                                CC7.append([alpha, beta])
                            if i == 7:
                                CC8.append([alpha, beta])
                            if i == 8:
                                CC9.append([alpha, beta])
                            if i == 9:
                                CC10.append([alpha, beta])
                        
                        frame_count += 1
                        e=0
                        f=0
                        t=0
     
                        if  frame_count == 100:
                            CC_combined = np.concatenate((CC1, CC2, CC3,CC4,CC5,CC6,CC7,CC8,CC9,CC10), axis=1)
                            model = lstm_model()
                            model.load_state_dict(torch.load("datos.pth", map_location=torch.device('cpu'), weights_only=False))
                            model.eval()
                            etiqueta_predicha = obtener_predicciones(model, CC_combined)
                            if etiqueta_predicha == 0:
                                emocion_texto = "Enojo"
                                e += 1
                            elif etiqueta_predicha == 1:
                                emocion_texto = "Felicidad"
                                f +=1
                            elif etiqueta_predicha == 2:
                                emocion_texto = "Tristeza"
                                t += 1
                            frame_count = 0
                            CC1 = []
                            CC2 = []
                            CC3 = []
                            CC4 = []
                            CC5 = []
                            CC6 = []
                            CC7 = []
                            CC8 = []
                            CC9 = []
                            CC10= []
                            
                  
                            
                if img is not None:
                    cv2.putText(img, f"Emocion: {emocion_texto}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 150 , 30), 10)
                    cv2.imshow("Face Mesh", img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
    except Exception as e:
        print("Se produjo un error:", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()

points = [
    (306, 427, 434),  
    (287, 410, 432),  
    (25, 173, 255),
    (159, 145, 6),
    (223, 321, 168),
    (285, 386, 276),
    (159, 173, 203),
    (223, 198, 168),
    (159, 145, 197),
    (223, 391, 168) 
]

show_video(len(points), "emoción", 1)
