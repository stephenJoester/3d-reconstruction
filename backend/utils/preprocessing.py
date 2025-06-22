from PIL import Image
import numpy as np  
import torch
from torchvision import transforms
import io
from rembg import remove

class Preprocessor:
    def __init__(self, device=None, model_type="DPT_Hybrid", resize=(128, 128)):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.resize = resize

        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device)
        self.midas.eval()

        self.transform = self._build_transform(['resize', 'totensor'], self.resize)

    def _build_transform(self, preprocess_list, resize):
        preprocess = []
        if 'grayscale' in preprocess_list:
            preprocess.append(transforms.Grayscale(1))
        if 'resize' in preprocess_list:
            preprocess.append(transforms.Resize(resize))
        if 'totensor' in preprocess_list:
            preprocess.append(transforms.ToTensor())
        return transforms.Compose(preprocess)

    def process(self, image_input):
        """
        image_input: bytes (from API) or PIL.Image.Image
        """
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError("image_input must be bytes or PIL.Image.Image")
        
        image = self.background_removal(image)
        print("-----Transforming image-----")
        transformed_image = self.transform(image)
        transformed_image_np = transformed_image.numpy()
        transformed_image = transformed_image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth_prediction = self.midas(transformed_image)
        depth_prediction = depth_prediction.squeeze().cpu().numpy()
        depth_prediction = (depth_prediction - depth_prediction.min()) / (depth_prediction.max() - depth_prediction.min())
        depth_pil = Image.fromarray((depth_prediction * 255).astype(np.uint8))

        depth_map = self.transform(depth_pil).numpy()
        depth_map_np = depth_map.squeeze()

        combined = np.concatenate((transformed_image_np, depth_map_np[None, :, :]), axis=0)
        return combined
    
    def background_removal(self, image_input):
        """Remove background from raw image

        Args:
            image_input (_type_): _description_
        """
        print("-----Removing background...-----")
        no_bg = remove(image_input)
        return no_bg.convert("RGB")