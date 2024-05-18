from torchinfo import summary
from super_gradients.training import models

model = models.get(
        'yolo_nas_m',
        num_classes=28,
        checkpoint_path='/Users/amade/OneDrive/Desktop/SKRIPSI/YOLO-NAS-Car-Logo-Detection/ckpt_best.pth'
    )

summary(model, (16, 3, 480, 480))