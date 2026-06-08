import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="xcltql666/Video-As-Prompt-Wan2.1-14B", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
)
state_dict = load_state_dict("models/train_0301_new_stage_2/Video-As-Prompt-Wan2.1-14B_full/epoch-1.safetensors")
pipe.vap.load_state_dict(state_dict)

ref_video_path = '/home/users/astar/cfar/stuchengyou/jcy/xcl/Magic-TryOn/temp/vap_train_data_final/test/videos_ref/vivid_lower_body_1219220_detail.mp4'
target_image_path = '/home/users/astar/cfar/stuchengyou/jcy/xcl/Magic-TryOn/temp/vap_train_data_final/test/images/vivid_lower_body_1219220_detail.png'

image = Image.open(target_image_path).convert("RGB")
ref_video = VideoData(ref_video_path, height=832, width=640)
ref_frames = [ref_video[i] for i in range(77)]

vap_prompt = "The model wearing a [cloth] is showcasing the garment."
prompt = "The model with her face turned away, wearing a red short-sleeved top and wide-leg pink pants with a black side stripe pattern, and black high-heeled pumps, carrying a silver metallic handbag, is showcasing the overall look."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    input_image=image,
    seed=42, tiled=True,
    height=832, width=640,
    num_frames=77,
    vap_video=ref_frames,
    vap_prompt=vap_prompt,
    negative_vap_prompt=negative_prompt,
)
save_video(video, "video_Video-As-Prompt-Wan2.1-14B.mp4", fps=25, quality=5)
