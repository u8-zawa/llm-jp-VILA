import torch

from llava.model.multimodal_encoder.vision_encoder import VisionTower, VisionTowerS2
from llava.model.multimodal_encoder.clyp.vision_modeling_clyp import CLYPVisionModel
from llava.model.multimodal_encoder.clyp.image_processing_clyp import CLYPImageProcessor
from llava.model.multimodal_encoder.clyp.configuration_clyp import CLYPConfig
from transformers import AutoConfig, PretrainedConfig, AutoModel


class CLYPVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.image_processor = CLYPImageProcessor.from_pretrained(model_name_or_path)
        # This cause an error
        # self.vision_tower = CLYPVisionModel.from_pretrained(
        #     model_name_or_path, torch_dtype=eval(config.model_dtype),
        # )
        self.vision_tower = CLYPVisionModel.from_pretrained(
            model_name_or_path
        )
        self.vision_tower.config.hidden_size = 768
        self.vision_tower.config.image_size = 224
        self.is_loaded = True


# class CLIPVisionTowerS2(VisionTowerS2):
#     def __init__(self, model_name_or_path: str, config: PretrainedConfig):
#         super().__init__(model_name_or_path, config)
#         self.image_processor = CLYPImageProcessor.from_pretrained(model_name_or_path)
#         self.vision_tower = CLYPVisionModel.from_pretrained(
#             model_name_or_path, torch_dtype=eval(config.model_dtype)
#         )

#         # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
#         self.image_processor.size['shortest_edge'] = self.scales[-1]
#         self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.scales[-1]

#         self.is_loaded = True

# AutoConfig.register("clyp", CLYPVisionConfig)
# AutoModel.register(CLYPVisionConfig, CLYPVisionModel)