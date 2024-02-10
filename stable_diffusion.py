import torch
import os
import subprocess
import ffmpeg
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


model_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_attention_slicing()

# def generate(prompts,ind):
#     generator = [torch.Generator("cuda").manual_seed(6669) for _ in range(len(prompts))]
#     image = pipeline(prompt=prompts, generator=generator, num_inference_steps=30).images
#     image[0].save('/DATA/elidandi_2211ai08/Reciepe2video/collected/img_'+str(ind)+'.jpg', 'JPEG')
#     img = cv2.imread('/DATA/elidandi_2211ai08/Reciepe2video/collected/img_'+str(ind)+'.jpg')
    

def generate_and_add_text_slide(prompts, ind, duration_text=2, duration_image=3):
    """
    Generates a video with the input prompts and adds the input text to the video.
    :param prompts: The input prompts.
    :param ind: The index of the video.
    :param duration_text: The duration of the input text.
    """

    generator = [torch.Generator("cuda").manual_seed(6669) for _ in range(len([prompts]))]
    image = pipeline(prompt=[prompts], generator=generator, num_inference_steps=30).images
    image[0].save('/DATA/elidandi_2211ai08/Reciepe2video/collected/generated/img_'+str(ind)+'.jpg', 'JPEG')
    img_path = f'/DATA/elidandi_2211ai08/Reciepe2video/collected/img_{ind}.jpg'

    # Check if the input image file exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    
    output_video_path = f'collected/generated/video_{ind}.mp4'
    # Create a temporary text video with the input text
    temp_text_video_path = 'collected/temp_text_video.mp4'
    subprocess.run(['ffmpeg', '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100 -t {duration_text}', '-vf', f"drawtext=text='{prompts}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2", temp_text_video_path])

    # Create a temporary video file with the image
    temp_image_video_path = 'collected/temp_image_video.mp4'
    subprocess.run(['ffmpeg', '-loop', '1', '-i', f'{img_path}', '-c:v', 'libx264', '-t', f'{duration_image}', '-pix_fmt', 'yuv420p', temp_image_video_path])

    # Concatenate the text video and the image video
    subprocess.run(['ffmpeg', '-i', f'{temp_text_video_path}', '-i', f'{temp_image_video_path}', '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[outv]', '-map', '[outv]', '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast', '-c:a', 'aac', '-b:a', '192k', '-shortest', output_video_path])

    # Remove the temporary video files
    os.remove(temp_text_video_path)
    os.remove(temp_image_video_path)

    output_folder = 'collected/trimmed/'
    trimmed_output_file = f'{output_folder}Recipe_video.mp4'

    existing_video = None
    for file in os.listdir(output_folder):
        if file.endswith(".mp4"):
            existing_video = os.path.join(output_folder, file)
            break
    if existing_video:
        # Concatenate the existing video and the current processed video using FFmpeg
        ffmpeg.input(existing_video).input(output_video_path).output(trimmed_output_file).run(overwrite_output=True)
    else:
        # Save the current processed video as is
        ffmpeg.input(output_video_path).output(trimmed_output_file).run(overwrite_output=True)
