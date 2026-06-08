import torch
import PIL
from PIL import Image
import os
import json
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from typing import List

# ================= 配置路径 =================
# 输入视频目录
video_dir = '/home/users/astar/cfar/stuchengyou/jcy/xcl/Magic-TryOn/temp/vap_train_data_final/test/split_videos_ref/part_1'
# 输入图片目录
image_dir = '/home/users/astar/cfar/stuchengyou/jcy/xcl/Magic-TryOn/temp/vap_train_data_final/test/images'
# Prompt JSON 文件路径
json_path = '/home/users/astar/cfar/stuchengyou/jcy/xcl/Magic-TryOn/temp/vap_train_data_clean/test/caption.json'
# 输出保存目录
output_dir = '/home/users/astar/cfar/stuchengyou/jcy/xcl/DiffSynth-Studio/tryon_result/res_baseline_0301'
os.makedirs(output_dir, exist_ok=True)
# ===========================================

# 加载模型 (只需加载一次)
print("正在加载模型...")
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
print("模型加载完成。")

# 加载 Prompt JSON 数据
print(f"正在读取 Prompt 文件: {json_path}")
with open(json_path, 'r', encoding='utf-8') as f:
    captions = json.load(f)

def select_frames(video_frames, num):
    idx = torch.linspace(0, len(video_frames) - 1, num).long().tolist()
    return [video_frames[i] for i in idx]

# 固定参数
fixed_vap_prompt = "The model wearing a [cloth] is showcasing the garment."
fixed_negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# 遍历视频目录进行批量推理
print(f"开始批量推理，输入目录: {video_dir}")
files = os.listdir(video_dir)
# 简单的进度计数
total_files = len([f for f in files if f.endswith('.mp4')])
processed_count = 0

for filename in files:
    if not filename.endswith('.mp4'):
        continue
    
    # 1. 确定文件路径
    video_path = os.path.join(video_dir, filename)
    
    # 对应的图片文件名 (假设除后缀外文件名完全一致)
    image_filename = os.path.splitext(filename)[0] + ".png"
    target_image_path = os.path.join(image_dir, image_filename)
    
    # 2. 检查图片是否存在
    if not os.path.exists(target_image_path):
        print(f"[警告] 跳过 {filename}: 对应的图片 {image_filename} 不存在于 {image_dir}")
        continue
        
    # 3. 获取 Prompt
    if filename in captions:
        current_prompt = captions[filename]
    else:
        print(f"[警告] 跳过 {filename}: 在 JSON 文件中未找到对应的 prompt key")
        current_prompt = fixed_vap_prompt

    processed_count += 1
    print(f"[{processed_count}/{total_files}] 正在处理: {filename}")
    print(f"   - 图片: {image_filename}")
    print(f"   - Prompt: {current_prompt[:50]}...") # 只打印前50个字符

    try:
        # 4. 数据预处理
        image = Image.open(target_image_path).convert("RGB")
        ref_video = VideoData(video_path, height=832, width=640)
        ref_frames = select_frames(ref_video, num=49)

        # 5. 推理
        video = pipe(
            prompt=current_prompt,
            negative_prompt=fixed_negative_prompt,
            input_image=image,
            seed=42, 
            tiled=True,
            height=832, 
            width=640,
            num_frames=49,
            vap_video=ref_frames,
            vap_prompt=fixed_vap_prompt,
            negative_vap_prompt=fixed_negative_prompt,
        )
        
        # 6. 保存结果
        output_filename = f"result_{filename}"
        save_path = os.path.join(output_dir, output_filename)
        save_video(video, save_path, fps=25, quality=5)
        print(f"   - 保存成功: {save_path}")
        
    except Exception as e:
        print(f"[错误] 处理 {filename} 时发生错误: {e}")

print("批量推理结束。")