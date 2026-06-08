import torch
import PIL
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
from typing import List


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="xcltql666/Video-As-Prompt-Wan2.1-14B", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# dataset_snapshot_download("DiffSynth-Studio/example_video_dataset", allow_file_pattern="wanvap/*", local_dir="data/example_video_dataset")
ref_video_path = '/home/users/astar/cfar/stuchengyou/jcy/xcl/Magic-TryOn/temp/vap_train_data_clean/test/videos_ref/vivid_dresses_1148255_detail.mp4'
target_image_path = '/home/users/astar/cfar/stuchengyou/jcy/xcl/Magic-TryOn/temp/vap_train_data_clean/test/temp/videos_ref_edited_images/vivid_dresses_1148255_detail.png'

def select_frames(video_frames, num):
    idx = torch.linspace(0, len(video_frames) - 1, num).long().tolist()
    return [video_frames[i] for i in idx]

image = Image.open(target_image_path).convert("RGB")
ref_video = VideoData(ref_video_path, height=832, width=640)
ref_frames = select_frames(ref_video, num=49)

vap_prompt = "The model wearing a white midi dress with sheer black long sleeves and a side slit is showcasing the garment."
prompt = "The model wearing a white midi dress with sheer black long sleeves and a side slit is showcasing the garment."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    input_image=image,
    seed=42, tiled=True,
    height=832, width=640,
    num_frames=49,
    vap_video=ref_frames,
    vap_prompt=vap_prompt,
    negative_vap_prompt=negative_prompt,
)
save_video(video, "video_Video-As-Prompt-Wan2.1-14B.mp4", fps=25, quality=5)