import os
import json
import torch
import cv2
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from tqdm import tqdm

# ================= 配置路径 =================
REF_VIDEO_DIR = '/home/users/astar/cfar/stuchengyou/jcy/xcl/Magic-TryOn/temp/vap_train_data_final/test/split_videos_ref/part_0'
TARGET_IMAGE_DIR = '/home/users/astar/cfar/stuchengyou/jcy/xcl/Magic-TryOn/temp/vap_train_data_final/test/images'
JSON_PATH = '/home/users/astar/cfar/stuchengyou/jcy/xcl/Magic-TryOn/temp/vap_train_data_final/test/caption.json'
OUTPUT_DIR = '/home/users/astar/cfar/stuchengyou/jcy/xcl/DiffSynth-Studio/tryon_result/0301_new_epoch1_1'

# 固定的 Prompt
VAP_PROMPT = "The model wearing a [cloth] is showcasing the garment."
NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 模型加载 =================
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
)
state_dict = load_state_dict("/home/users/astar/cfar/stuchengyou/jcy/xcl/DiffSynth-Studio/models/train_0301_new_stage_2/Video-As-Prompt-Wan2.1-14B_full/epoch-1.safetensors")
pipe.vap.load_state_dict(state_dict)
print("模型加载完成。")

# ================= 加载 Prompt JSON =================
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    captions = json.load(f)

# ================= 批量推理逻辑 =================
video_files = [f for f in os.listdir(REF_VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
print(f"共发现 {len(video_files)} 个待处理视频。")

for video_filename in tqdm(video_files, desc="Processing"):
    if os.path.exists(os.path.join(OUTPUT_DIR, video_filename)):
        print(f"\n[跳过] 输出已存在: {video_filename}")
        continue
    try:
        # 1. 解析文件名和路径
        video_stem = os.path.splitext(video_filename)[0]
        ref_video_path = os.path.join(REF_VIDEO_DIR, video_filename)
        
        # 2. 寻找对应的图片
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            temp_path = os.path.join(TARGET_IMAGE_DIR, f"{video_stem}{ext}")
            if os.path.exists(temp_path):
                image_path = temp_path
                break
        
        if image_path is None:
            print(f"\n[跳过] 未找到视频 {video_filename} 对应的图片")
            continue

        # 3. 获取 Prompt
        current_prompt = captions.get(video_filename, "")

        # 4. 获取视频元数据 (OpenCV)
        cap = cv2.VideoCapture(ref_video_path)
        if not cap.isOpened():
            print(f"\n[错误] 无法打开视频: {ref_video_path}")
            continue
            
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # [关键步骤] 尺寸对齐：确保是16的倍数
        aligned_width = (raw_width // 16) * 16
        aligned_height = (raw_height // 16) * 16

        # 5. 准备推理数据
        image = Image.open(image_path).convert("RGB")
        # 将输入图片也缩放到对齐尺寸
        image = image.resize((aligned_width, aligned_height), Image.LANCZOS)
        
        ref_video = VideoData(ref_video_path)
        
        # 确定实际可读取的最大帧数 (取 VideoData 和 OpenCV 结果的较小值，防止越界)
        safe_frame_count = min(len(ref_video), original_frame_count)
        
        # [逻辑修改] : 确定目标帧数 (取 49 和 实际帧数 的较小值)
        # 如果视频长，就截取前49帧；如果视频短，就按原长度。
        target_num_frames = min(49, safe_frame_count)
        
        ref_frames = []
        # 只循环到 target_num_frames 即可，不用多读，也不用补帧
        for i in range(target_num_frames):
            frame = ref_video[i]
            # 强制 Resize 到对齐尺寸 (解决 Tensor 维度不匹配报错)
            if frame.size != (aligned_width, aligned_height):
                frame = frame.resize((aligned_width, aligned_height), Image.LANCZOS)
            ref_frames.append(frame)

        # 6. 执行推理
        video = pipe(
            prompt=current_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            input_image=image,
            seed=42, 
            tiled=True,
            height=aligned_height, 
            width=aligned_width,
            num_frames=target_num_frames, # 这里的数值必须和 len(ref_frames) 一致
            vap_video=ref_frames,
            vap_prompt=VAP_PROMPT,
            negative_vap_prompt=NEGATIVE_PROMPT,
        )

        # 7. 保存结果
        output_filename = f"{video_stem}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        save_video(video, output_path, fps=input_fps, quality=5)
        
        # 清理显存
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n[错误] 处理 {video_filename} 时发生异常: {e}")
        import traceback
        traceback.print_exc()

print("批量推理完成。")