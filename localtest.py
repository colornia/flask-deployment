from torchvision import transforms
import io
from detection import models
from segmentation import models1
from PIL import Image
def transform_image(image_bytes):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.40760392, 0.4595686, 0.48501961],
                std=[0.225, 0.224, 0.229]
    ),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)
model = models1.ResNeXTDAHead()
img_bytes = open('/123.jpg', 'rb').read()
k= transform_image(img_bytes)
rsu = model(k)
img = rsu[0].squeeze(0)
transs = transforms.ToPILImage()
img = transs(img)
img.show()
print(img)


