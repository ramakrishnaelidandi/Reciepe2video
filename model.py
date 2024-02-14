import pickle
import os
import subprocess
from PIL import Image
from utlis import NER_classification, similarity_mapping, trim_and_concat_videos, extract_frames
from stable_diffusion import generate_and_add_text_slide

##load the datase
file_path = 'data/dataset.pkl'

try:
    with open(file_path, 'rb') as file:
        df = pickle.load(file)
    print("Successfully loaded the dataset.")
except FileNotFoundError:
    print(f"File not found at '{file_path}'. Please check the file path.")
except EOFError:
    print("EOFError: Ran out of input. The file may be empty or corrupted.")
except Exception as e:
    print(f"An error occurred: {str(e)}")

# print(df.columns)

# def trim_and_concat_videos(ind, video_id, instruction, df):
#     ydl_opts = {
#         'format': 'best',
#         'outtmpl': '/DATA/elidandi_2211ai08/Reciepe2video/collected/video_' + str(ind) + '.mp4',
#         'quiet': True
#     }
#     print('---------------------------------------------------------------------------------------------------------------------------------------')
#     print(f'recipe step: {instruction}')
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([df.iloc[video_id]['url']])

# def download_youtube_video(url, output_path='/DATA/elidandi_2211ai08/Reciepe2video/data'):
#     try:
#         ydl_opts = {
#             'format': 'bestvideo+bestaudio/best',
#             'outtmpl': f'{output_path}/%(title)s.%(ext)s',
#         }

#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             info_dict = ydl.extract_info(url, download=True)
#             print(f"Video '{info_dict['title']}' downloaded successfully.")
#     except Exception as e:
#         print(f"Error: {e}")

# # Example usage
# youtube_url = df.iloc[0]['url']
# download_youtube_video(youtube_url)


# print(NER_classification("spread butter on two slices of white bread"))

# recipe = [
#     "Place a rack in the middle of the oven and preheat it to 400°F.",
#     "In a large saucepan over medium heat, bring evaporated milk and whole milk to a bare simmer.",
#     "Whisk in garlic powder, onion powder, paprika, pepper, and 1 tsp. salt.",
#     "Working in batches, whisk in three fourths of the cheddar, then all of the cream cheese.",
#     "Meanwhile, bring a large pot of generously salted water to a boil (it should have a little less salt than seawater)."
# ]
    
recipe = ["Crack the eggs into a bowl and whisk until well combined",
          "Stir in the milk, salt, and pepper into the boil which contain crackked eggs in a bowl",
          "Heat a non-stick skillet over medium heat and add the butter or oil",
          "Pour the egg mixture into the skillet and tilt the pan to spread it evenly.",
          "Sprinkle the desired fillings over one half of the omelet.",
          "Using a spatula, fold the omelet in half and cook for another minute or two, or until the eggs are cooked through."
]

# print(similarity_mapping([recipe[2]],df))

######################################################################## generate a recipe video #########################################################################

def create_video_list_file(file_paths, output_file='videos.txt'):
    # Filter out files with '.ipynb' extension and '.ipynb_checkpoints' directory
    file_paths = [path for path in file_paths if not (path.endswith('.ipynb') or os.path.basename(path) == '.ipynb_checkpoints')]

    # Sort the file paths
    file_paths.sort()

    with open(output_file, 'w') as file:
        file.write('\n'.join([f"file '{path}'" for path in file_paths]))


def recipe2vid(recipe):
    # extract_frames_folder = 'collected/extracted_frames'
    for ind,asset in enumerate([j[0] for j in similarity_mapping(recipe,df)]):
        if asset[1] >= 0.80:
            trim_and_concat_videos(ind,asset[0],recipe[ind],df)
        else:
            generate_and_add_text_slide(recipe[ind],ind)

    directory_path = '/content/collected/concat'
    video_files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]
    create_video_list_file(video_files)
    subprocess.run("ffmpeg -f concat -safe 0 -i videos.txt -c copy output.mp4", shell=True)
    print('Your recipe video has been generated successfully')

recipe2vid(recipe)

### for deleting the files

def delete_all_files_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {filename}")
        except Exception as e:
            print(f"Error deleting file {filename}: {e}")

# delete_all_files_in_directory('collected/concat')
# delete_all_files_in_directory('/content/collected/generated')
# delete_all_files_in_directory('collected')





######################################################################## extract frames from a video #########################################################################

# for folder in os.listdir('collected/trimmed'):
#     for ind, file in enumerate(os.listdir(os.path.join('collected/trimmed',folder))):
#         os.makedirs(os.path.join(os.path.join('collected/extracted_frames',folder), str(ind)))
#         extract_frames(os.path.join(os.path.join('collected/trimmed',folder),file), os.path.join(os.path.join('collected/extracted_frames',folder), str(ind)))

# image = Image.open('collected/extracted_frames/frame179.jpg').convert('RGB')

# print(image)



## KL divergence 


######################################################################## to check for similarity of captions################################################
# captions = []
# for ind,asset in enumerate(similarity_mapping(["Stir in the milk, salt, and pepper into the boil which contain crackked eggs in a bowl"],df)[0]):
#     captions.append([df.iloc[asset[0]]['sentence'], df.iloc[asset[0]]['url']])
#     print(df.iloc[asset[0]]['url'])
# print(captions)
# print(similarity_mapping(["Stir in the milk, salt, and pepper into the boil which contain crackked eggs in a bowl"],df))