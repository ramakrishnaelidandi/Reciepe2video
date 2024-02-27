import torch
import os
import yt_dlp
import ffmpeg
import clip
import subprocess
import numpy as np
from PIL import Image
from key_frame_extraction import extract_frames
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer,util
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from stable_diffusion import generate_and_add_text_slide
from LLM import generate_keyphrases

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
sentence_model = SentenceTransformer(model_name)
sentence_model.to(device)

def similarity_mapping(reciepe, df):
  collection = []
  for text in reciepe:
    with torch.no_grad():
      input_emb = sentence_model.encode(NER_classification(str(text)), convert_to_tensor=True).to(device)

  # Calculate cosine similarities
    similarity_score = util.pytorch_cos_sim(input_emb, torch.stack(df.embeding.tolist()).to(device)).view([-1]).tolist()
    sorted_indices = sorted(range(len(similarity_score)), key = lambda i: similarity_score[i], reverse = True)
    collection.append([[i,similarity_score[i]] for i in sorted_indices[:5] if similarity_score[i] >= 0.8])
  return collection

## moving video_clips
def move_video_clips(source_file_path, recp_ind):

  # source_file_path = os.path.join('collected/trimmed',str(recp_ind) +'/'+ sorted(os.listdir(os.path.join('collected/trimmed',str(recp_ind))))[int(selected_ind)])

  # Destination directory path
  destination_directory = 'collected/concat'

  # Extract file name from source path
  file_name = os.path.basename(source_file_path)

  # Construct the destination file path
  destination_file_path = os.path.join(destination_directory, str(recp_ind) + file_name)

  # Move the file
  os.rename(source_file_path, destination_file_path)

## video processing
def trim_and_concat_videos(recp_ind, recipe, df, asset):
  
  #create folder for extracted video frames
  os.makedirs(os.path.join('collected/extracted_frames',str(recp_ind)))
  os.makedirs(os.path.join('collected/download',str(recp_ind)))
  os.makedirs(os.path.join('collected/trimmed',str(recp_ind)))
  # os.makedirs(os.path.join('collected/trimmed',str(recp_ind)))

  score_txt_file = os.path.join('collected/extracted_frames',str(recp_ind) +'/'+'score.txt')

  ranking_need = True

  for vid_ind, [video_id, score] in enumerate(asset):

  # Define the start and end times in seconds
    start_time = df.iloc[video_id]['segment'][0]
    end_time = df.iloc[video_id]['segment'][1]

  # skipping vides that are less than 3 seconds
    if score < 0.9 and (end_time - start_time) <= 5:
        continue
    
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join('collected/download',str(recp_ind) +'/'+ str(vid_ind)+'.mp4'),
        'quiet': True
        }
    
    try:
      with yt_dlp.YoutubeDL(ydl_opts) as ydl:
          ydl.download([df.iloc[video_id]['url']])

      video_file =  os.path.join('collected/download',str(recp_ind) +'/'+ str(vid_ind)+'.mp4')

      # Define output file path for trimmed video
      trimmed_output_file = os.path.join('collected/trimmed',str(recp_ind) +'/'+ str(vid_ind)+'.mp4')
      # subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', video_file, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', trimmed_output_file, '-y'])
      # subprocess.run(f'ffmpeg -hide_banner -loglevel error -i {video_file} -ss {str(start_time)} -to {str(end_time)} -an -c:v libx264 {trimmed_output_file} -y', shell = True)
      subprocess.run(f'ffmpeg -hide_banner -loglevel error -i {video_file} -ss {str(start_time)} -to {str(end_time)} -an -c copy {trimmed_output_file} -y', shell = True)
      # extract video frames
      os.makedirs(os.path.join('collected/extracted_frames',str(recp_ind) +'/'+ str(vid_ind)))
      extract_out_folder = os.path.join('collected/extracted_frames',str(recp_ind) +'/'+ str(vid_ind))

      # note the scores in the score_text file of each clip to be trimmed
      with open(score_txt_file, 'a') as file:
         file.write(str(score)+'\n')

      if score >= 0.9 and (end_time - start_time) >= 2:
        move_video_clips(trimmed_output_file, recp_ind)
        ranking_need = False

        break

      else:
        extract_frames(trimmed_output_file, extract_out_folder)

      # note the scores in the score_text file of each clip to be trimmed
      

    except yt_dlp.DownloadError as e:
      print(f'Error downloading video {recp_ind}/{vid_ind}: {e}')

  with open(score_txt_file, 'r') as file:
    score_text = file.read()
  
  if not score_text:
    # no video is downloaded successfully
    generate_and_add_text_slide(recipe[recp_ind], recp_ind)

  elif ranking_need :
    # ranking the trimmed clips
    os.remove(score_txt_file)

    key_phrases = generate_keyphrases(recipe[recp_ind])

    selected_ind = rank_assets(str(recp_ind), key_phrases)

    # Source file path
    source_file_path = os.path.join('collected/trimmed',str(recp_ind) +'/'+ sorted(os.listdir(os.path.join('collected/trimmed',str(recp_ind))))[int(selected_ind)])

    move_video_clips(source_file_path, recp_ind)
    




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



## ranking the assets CLIP and KL divergence
## load the model    
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def rank_assets(asset,key_phrases):
  tensor_list = []
  for folder in sorted([int(i) for i in os.listdir(os.path.join('collected/extracted_frames',asset))]):
    frames = []
    folder = str(folder)
    image = torch.stack([preprocess(Image.open(os.path.join(os.path.join(os.path.join('collected/extracted_frames',asset),folder), file)).resize((596, 437))) for file in os.listdir(os.path.join(os.path.join('collected/extracted_frames',asset),folder))]).to(device)
    tensor_list.append(image)
  text = clip.tokenize(key_phrases).to(device)

  prob_list = []
  with torch.no_grad():
    for ind in range(len(tensor_list)):
      logits_per_image, logits_per_text = clip_model(tensor_list[ind],text)
      probs = logits_per_image.softmax(dim = -1).cpu().numpy()
      prob_list.append(probs)

  mean_dist = []
  for j in range(len(key_phrases)):
    mean_probability = np.mean(prob_list[j], axis=0)
    # print("Mean Probability:", mean_probability)
    mean_dist.append(mean_probability)

  # def generate_uniform_probability_distribution(k):
  #   alpha = np.ones(k)
  #   dirichlet_sample = np.random.dirichlet(alpha)
  #   return dirichlet_sample
    
  ## generate_uniform_probability_distribution
  uniform_probability_distribution = np.random.dirichlet(np.ones(len(key_phrases)))
  # uniform_probability_distribution = np.ones(len(key_phrases)) * (1 / len(key_phrases))
  
  def compare_distributions(distributions, target_distribution):
    similarities = []
    for distribution in distributions:
        kl_divergence = entropy(target_distribution, distribution)
        similarities.append(kl_divergence)

    # most_similar_index = similarities.index(min(similarities))
    # return f"Distribution {most_similar_index + 1} is most similar to the target distribution."
    return int(similarities.index(min(similarities)))
  
  return compare_distributions(mean_dist, uniform_probability_distribution)

