# Tasty Videos Dataset 

'ALL_RECIPES_without_videos' contains the recipe steps (recipe_encoded.xml), frames per second for the videos (fps.txt) and the annotations (csvalignment.dat) for each recipe in our dataset. 

'SPLITS' contains our dataset splits. 'split_2511' is the split set used in [1]. 'split_4022' contains the new split set for our extended dataset. 

'ALL_RECIPES.txt' contains the list of the names of all the recipes in our dataset. 

We do not directly share the recipe text and videos but encode the recipe text and provide the URLs to download the videos.

To decode the recipe text use the following script. 
python3 fs_decode_xml.py 
'id2word_tasty.txt' includes the encoding information. 

If you want to download the videos and extract frames run the following scripts.
python3 fs_download_videos.py
python3 fs_video2frames.py

To process the dataset and create the splits please use the following function:
python3 fs_create_splits.py
This script reads the recipe text and ingredient information in the 'recipe.xml' files. It also reads the annotation information in 'csvalignment.dat' and relates the recipe steps to the corresponding video frames. 

[1] Fadime Sener and Angela Yao, Zero-Shot Anticipation for Instructional Activities, The IEEE International Conference on Computer Vision (ICCV), 2019. 
 
  
