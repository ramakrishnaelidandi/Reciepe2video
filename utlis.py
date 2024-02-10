import torch
import cv2
import ffmpeg
import os
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
import time
import matplotlib.pyplot as plt
import yt_dlp
from sentence_transformers import SentenceTransformer,util
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NER model
tokenizer = AutoTokenizer.from_pretrained("Dizex/InstaFoodBERT-NER")
model_ner = AutoModelForTokenClassification.from_pretrained("Dizex/InstaFoodBERT-NER")
model_ner.to(device)
with torch.no_grad():
  pipe = pipeline("ner", model=model_ner, tokenizer=tokenizer, device = device)

def NER_classification(query):
  entities, curr_entity = [], ''
  for i in pipe(query):
    if i['entity'][:2] == 'B-':
      if curr_entity:
        entities.append(curr_entity)
      curr_entity = i['word']

    elif i['entity'][:2] == 'I-':
      if curr_entity:
        curr_entity += " " + i['word']

    else:
      entities.append(curr_entity)
      curr_entity = ""

  if curr_entity:
    entities.append(curr_entity)

  return query + ", " + ", ".join(entities)


## Text Similarity
model_name = "paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
model.to(device)

def similarity_mapping(reciepe, df):
  collection = []
  for text in reciepe:
    with torch.no_grad():
      input_emb = model.encode(NER_classification(str(text)), convert_to_tensor=True).to(device)

  # Calculate cosine similarities
    similarity_score = util.pytorch_cos_sim(input_emb, torch.stack(df.embeding.tolist()).to(device)).view([-1]).tolist()
    sorted_indices = sorted(range(len(similarity_score)), key = lambda i: similarity_score[i], reverse = True)
    collection.append([[i,similarity_score[i],df.sentence[i]] for i in sorted_indices[:5]])
  return collection


## video processing

def trim_and_concat_videos(ind, video_id, instruction, df):
    ydl_opts = {
        'format': 'best',
        'outtmpl': 'collected/video_' + str(ind) + '.mp4',
        'quiet': True
    }
    print('---------------------------------------------------------------------------------------------------------------------------------------')
    print(f'recipe step: {instruction}')
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([df.iloc[video_id]['url']])
    # print(f'recipe step: {instruction}')

    video_file = f'collected/video_{ind}.mp4'

    # Define the start and end times in seconds
    start_time = df.iloc[video_id]['segment'][0]
    end_time = df.iloc[video_id]['segment'][1]

    # Define output folder
    output_folder = 'collected/concat/'

    # Define output file path for trimmed video
    trimmed_output_file = f'{output_folder}clip_{ind}.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', video_file, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', trimmed_output_file, '-y'])

    # Check if there's an existing video in the output folder
    # existing_video = None
    # for file in os.listdir(output_folder):
    #     if file.endswith(".mp4"):
    #         existing_video = os.path.join(output_folder, file)
    #         break

    # if existing_video:
    #     # Concatenate the existing video and the current processed video using FFmpeg
    #     ffmpeg.input(existing_video).input(video_file, ss=start_time, to=end_time).output(trimmed_output_file).run(overwrite_output=True)
    # else:
    #     # Save the current processed video as is
    #     ffmpeg.input(video_file, ss=start_time, to=end_time).output(trimmed_output_file).run(overwrite_output=True)
    # if existing_video:
    #     # Concatenate the existing video and the current processed video using FFmpeg
    #     subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', existing_video, '-i', video_file, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', trimmed_output_file, '-y'])
    # else:
    #     # Save the current processed video as is
    #     subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', video_file, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', trimmed_output_file, '-y'])


## key frame extraction from video
      
def extract_frames(video_file, output_folder):

  cap = cv2.VideoCapture(video_file) 

  arr = np.empty((0, 1944), int)   #initializing 1944 dimensional array to store 'flattened' color histograms
  D=dict()   #to store the original frame (array)
  count=0    #counting the number of frames
  start_time = time.time()
  while cap.isOpened():
      # Read the video file.
      ret, frame = cap.read()
      # If we got frames.
      if ret == True:
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #since cv reads frame in bgr order so rearraning to get frames in rgb order
          D[count] = frame_rgb   #storing each frame (array) to D , so that we can identify key frames later 
          
          #dividing a frame into 3*3 i.e 9 blocks
          height, width, channels = frame_rgb.shape

          if height % 3 == 0:
              h_chunk = int(height/3)
          else:
              h_chunk = int(height/3) + 1

          if width % 3 == 0:
              w_chunk = int(width/3)
          else:
              w_chunk = int(width/3) + 1

          h=0
          w= 0 
          feature_vector = []
          for a in range(1,4):
              h_window = h_chunk*a
              for b in range(1,4):
                  frame = frame_rgb[h : h_window, w : w_chunk*b , :]
                  hist = cv2.calcHist(frame, [0, 1, 2], None, [6, 6, 6], [0, 256, 0, 256, 0, 256])#finding histograms for each block  
                  hist1= hist.flatten()  #flatten the hist to one-dimensinal vector 
                  feature_vector += list(hist1)
                  w = w_chunk*b
                  
              h = h_chunk*a
              w= 0
          arr =np.vstack((arr, feature_vector )) #appending each one-dimensinal vector to generate N*M matrix (where N is number of frames
            #and M is 1944) 
          count+=1
      else:
          break

  print((time.time() - start_time))

  final_arr = arr.transpose()
  A = csc_matrix(final_arr, dtype=float)

  #top 63 singular values from 76082 to 508
  u, s, vt = svds(A, k = 63)
  v1_t = vt.transpose()

  projections = v1_t @ np.diag(s) #the column vectors i.e the frame histogram data has been projected onto the orthonormal basis 
  #formed by vectors of the left singular matrix u .The coordinates of the frames in this space are given by v1_t @ np.diag(s)
  #So we can see that , now we need only 63 dimensions to represent each column/frame

  f=projections
  C = dict() #to store frames in respective cluster
  for i in range(f.shape[0]):
      C[i] = np.empty((0,63), int)
      
  #adding first two projected frames in first cluster i.e Initializaton    
  C[0] = np.vstack((C[0], f[0]))   
  C[0] = np.vstack((C[0], f[1]))

  E = dict() #to store centroids of each cluster
  for i in range(projections.shape[0]):
      E[i] = np.empty((0,63), int)
      
  E[0] = np.mean(C[0], axis=0) #finding centroid of C[0] cluster

  count = 0
  for i in range(2,f.shape[0]):
      similarity = np.dot(f[i], E[count])/( (np.dot(f[i],f[i]) **.5) * (np.dot(E[count], E[count]) ** .5)) #cosine similarity
      #this metric is used to quantify how similar is one vector to other. The maximum value is 1 which indicates they are same
      #and if the value is 0 which indicates they are orthogonal nothing is common between them.
      #Here we want to find similarity between each projected frame and last cluster formed chronologically. 
      
      
      if similarity < 0.9: #if the projected frame and last cluster formed  are not similar upto 0.9 cosine value then 
                          #we assign this data point to newly created cluster and find centroid 
                          #We checked other thresholds also like 0.85, 0.875, 0.95, 0.98
                          #but 0.9 looks okay because as we go below then we get many key-frames for similar event and 
                          #as we go above we have lesser number of key-frames thus missed some events. So, 0.9 seems optimal.
                          
          count+=1         
          C[count] = np.vstack((C[count], f[i])) 
          E[count] = np.mean(C[count], axis=0)   
      else:  #if they are similar then assign this data point to last cluster formed and update the centroid of the cluster
          C[count] = np.vstack((C[count], f[i])) 
          E[count] = np.mean(C[count], axis=0) 

  b = []
  for i in range(f.shape[0]):
      b.append(C[i].shape[0])
  last = b.index(0)
  b1=b[:last ]

  res = [idx for idx, val in enumerate(b1) if val >= 25] #so i am assuming any dense cluster with atleast 25 frames is eligible to 
  #make shot.
  GG = C #copying the elements of C to GG, the purpose of  the below code is to label each cluster so later 
  #it would be easier to identify frames in each cluster
  for i in range(last):
      p1= np.repeat(i, b1[i]).reshape(b1[i],1)
      GG[i] = np.hstack((GG[i],p1))

  F= np.empty((0,64), int) 
  for i in range(last):
      F = np.vstack((F,GG[i]))
  colnames = []
  for i in range(1, 65):
      col_name = "v" + str(i)
      colnames+= [col_name]
  print(colnames)

  df = pd.DataFrame(F, columns= colnames)
  df['v64']= df['v64'].astype(int)
  df1 =  df[df.v64.isin(res)]
  new = df1.groupby('v64').tail(1)['v64']
  new1 = new.index 
  for c in new1:
      frame_rgb1 = cv2.cvtColor(D[c], cv2.COLOR_RGB2BGR) #since cv consider image in BGR order
      frame_num_chr = str(c)
      file_name = 'frame'+ frame_num_chr +'.jpg'
      output_file_path = os.path.join(output_folder, file_name)
      cv2.imwrite(output_file_path, frame_rgb1)
    
# def display_output(ind, video_id,recipe,df):
#     ydl_opts = {
#                 'format': 'best',
#                 'outtmpl': '/DATA/elidandi_2211ai08/Reciepe2video/collected/video_'+str(ind)+'.mp4',  # Set the output file path and name
#                 'quiet': True
#                 }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([df.iloc[video_id]['url']])
#     print(f'recipe step: {recipe[ind]}')

#     video_file ='/DATA/elidandi_2211ai08/Reciepe2video/collected/video_'+str(ind)+'.mp4'  # Replace with the path to your video file

# # Define the start and end times in seconds
#     start_time = df.iloc[video_id]['segment'][0]
#     end_time = df.iloc[video_id]['segment'][1]    
#     num_frames_to_display = 5  # Adjust as needed

#     # Open the video file for reading
#     cap = cv2.VideoCapture(video_file)

#     # Get the frames per second (fps) of the video
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Calculate the start and end frames based on time range
#     start_frame = int(start_time * fps)
#     end_frame = int(end_time * fps)

#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

#     # Function to display frames evenly spaced between start and end times
#     def display_evenly_spaced_frames(video_capture, end_frame, num_frames_to_display):
#         frames = []

#         frame_interval = (end_frame - start_frame) // (num_frames_to_display - 1)
#         current_frame = start_frame

#         while video_capture.isOpened() and current_frame <= end_frame:
#             video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
#             ret, frame = video_capture.read()

#             if not ret:
#                 break

#             frames.append(frame)
#             current_frame += frame_interval

#         for i, frame in enumerate(frames, start=1):
#             plt.figure(figsize=(8, 6))
#             plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             plt.axis('off')
#             plt.title(f'Frame {i}')
#             plt.show()

#     # Call the function to display evenly spaced frames
#     display_evenly_spaced_frames(cap, end_frame, num_frames_to_display)

#     # Release the video capture object
#     cap.release()

