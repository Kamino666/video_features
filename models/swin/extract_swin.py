from typing import Dict, List
from pathlib import Path

import numpy as np
import torch
from models._base.base_extractor import BaseExtractor
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from utils.utils import dp_state_to_normal, show_predictions_on_dataset
from utils.io import VideoLoader
from einops import rearrange


class ExtractSwin(BaseExtractor):

    def __init__(self, args) -> None:
        # init the BaseExtractor
        super().__init__(
            feature_type=args.feature_type,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            device=args.device,
        )
        # (Re-)Define arguments for this class
        self.model_name = args.model_name
        self.extraction_total = int(args.extraction_total)
        self.transforms = Compose([
            lambda np_array: Image.fromarray(np_array),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert('RGB'),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.pooling = args.pooling
        # self.extraction_fps = args.extraction_fps
        self.show_pred = args.show_pred
        self.output_feat_keys = ['swin', 'fps', 'timestamps_ms']
        self.name2module = self.load_model()

    @torch.no_grad()
    def extract(self, video_path: str) -> Dict[str, np.ndarray]:
        """The extraction call. Made to clean the forward call a bit.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: feature name (e.g. 'fps' or feature_type) to the feature tensor
        """

        video = VideoLoader(
            video_path,
            batch_size=self.extraction_total,
            total=self.extraction_total,
            tmp_path=self.tmp_path,
            keep_tmp=self.keep_tmp_files,
            transform=lambda x: self.transforms(x).unsqueeze(0)
        )
        vid_feats = []
        timestamps_ms = []
        for batch, timestamp_ms, idx in video:
            # batch = torch.stack(batch, dim=0)
            batch_feats = self.run_on_a_batch(batch)
            vid_feats.extend(batch_feats.tolist())
            timestamps_ms.extend(timestamp_ms)

        features_with_meta = {
            self.feature_type: np.array(vid_feats),
            'fps': np.array(video.fps),
            'timestamps_ms': np.array(timestamps_ms)
        }

        return features_with_meta

    def run_on_a_batch(self, batch: List[torch.Tensor]) -> torch.Tensor:
        model = self.name2module['model']
        batch = torch.cat(batch).unsqueeze(dim=0).to(self.device)  # [1, 16, 3, 224, 224]
        batch = rearrange(batch, 'b t c h w -> b c t h w')
        batch_feats = model.extract_feat(batch)
        if self.pooling is None:
            batch_feats = rearrange(batch_feats, 'b c t h w -> (b t h w) c')
        elif self.pooling == 'spatial':
            batch_feats = rearrange(batch_feats, 'b c t h w -> t (b h w) c').mean(dim=1)
        elif self.pooling == 'temporal':
            batch_feats = rearrange(batch_feats, 'b c t h w -> t (b h w) c').mean(dim=0)
        else:
            batch_feats = rearrange(batch_feats, 'b c t h w -> (b t h w) c').mean(dim=0)

        return batch_feats

    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints, sends them to the device.
        Since I3D is two-stream, it may load a optical flow extraction model as well.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        name2module = {}
        local_path = Path(__file__).parent
        # swin_base_patch244_window1677_sthv2
        config = local_path / f'./configs/recognition/swin/{self.model_name}.py'
        checkpoint = local_path / f'./checkpoints/{self.model_name}.pth'

        cfg = Config.fromfile(config)
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, str(checkpoint), map_location=self.device)

        name2module['model'] = model

        return name2module
