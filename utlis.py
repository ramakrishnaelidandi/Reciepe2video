import torch
import cv2
import ffmpeg
import os
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
        'outtmpl': '/DATA/elidandi_2211ai08/Reciepe2video/collected/video_' + str(ind) + '.mp4',
        'quiet': True
    }
    print('---------------------------------------------------------------------------------------------------------------------------------------')
    print(f'recipe step: {instruction}')
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([df.iloc[video_id]['url']])
    # print(f'recipe step: {instruction}')

    video_file = f'/DATA/elidandi_2211ai08/Reciepe2video/collected/video_{ind}.mp4'

    # Define the start and end times in seconds
    start_time = df.iloc[video_id]['segment'][0]
    end_time = df.iloc[video_id]['segment'][1]

    # Define output folder
    output_folder = '/DATA/elidandi_2211ai08/Reciepe2video/collected/trimmed/'

    # Define output file path for trimmed video
    trimmed_output_file = f'{output_folder}Recipe_video.mp4'

    # Check if there's an existing video in the output folder
    existing_video = None
    for file in os.listdir(output_folder):
        if file.endswith(".mp4"):
            existing_video = os.path.join(output_folder, file)
            break

    # if existing_video:
    #     # Concatenate the existing video and the current processed video using FFmpeg
    #     ffmpeg.input(existing_video).input(video_file, ss=start_time, to=end_time).output(trimmed_output_file).run(overwrite_output=True)
    # else:
    #     # Save the current processed video as is
    #     ffmpeg.input(video_file, ss=start_time, to=end_time).output(trimmed_output_file).run(overwrite_output=True)
    if existing_video:
        # Concatenate the existing video and the current processed video using FFmpeg
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', existing_video, '-i', video_file, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', trimmed_output_file, '-y'])
    else:
        # Save the current processed video as is
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', video_file, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', trimmed_output_file, '-y'])


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

