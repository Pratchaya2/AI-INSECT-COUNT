"""เทรน YOLOv8m บน lightning.ai (GPU L4, 24GB VRAM).

วิธีใช้บน Lightning Studio (เลือก GPU = L4):
    pip install ultralytics
    unzip ai-insect-dataset.zip      # ได้โฟลเดอร์ Dataset/
    python train_lightning.py
ผลลัพธ์: runs/insect_train_v5/weights/best.pt
"""
from pathlib import Path
import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATASET = HERE / "Dataset"  # แก้ตรงนี้ถ้า unzip ไว้ที่อื่น

cfg = {
    "path": str(DATASET),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 2,
    "names": ["fly", "test"],
}
cfg_path = HERE / "data_train.yaml"
cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

model = YOLO("runs_detect_train-4_weights_best.pt")  # warm-start ต่อจากโมเดลเดิม (ต้องอัปไฟล์นี้ขึ้น Studio)
model.train(
    data=str(cfg_path),
    epochs=150,
    patience=30,
    imgsz=1920,  # ตรงกับ inference ในแอป — รันได้บน L4 24GB (T4 16GB ไม่พอ assigner)
    batch=2,  # L4 24GB: batch=2/1920 ปลอดภัยจาก assigner OOM (ถ้านิ่งค่อยลองเพิ่ม 4)
    device=0,  # L4 CUDA
    project="runs",
    name="insect_train_v5",
)
