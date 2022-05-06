import os
import pathlib
from typing import Dict, Union

import numpy as np
import torch
from tqdm import tqdm
import pathlib as plb
import traceback

# from utils.utils import form_list_from_user_input, extract_wav_from_mp4, action_on_extraction
# from models.vggish.vggish_src import (vggish_input, vggish_params,
#                                       vggish_postprocess, vggish_slim)
from .vggish_src.vggish import VGGish
if "submodules" in __name__:
    from ...utils.utils import (action_on_extraction, form_list_from_user_input, extract_wav_from_mp4)
else:
    from utils.utils import (action_on_extraction, form_list_from_user_input, extract_wav_from_mp4)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}

class ExtractVGGish(torch.nn.Module):

    def __init__(self, args):
        super(ExtractVGGish, self).__init__()
        self.feature_type = args.feature_type
        self.path_list = form_list_from_user_input(args)
        self.keep_tmp_files = args.keep_tmp_files
        self.on_extraction = args.on_extraction
        self.tmp_path = os.path.join(args.tmp_path, self.feature_type)
        self.output_path = os.path.join(args.output_path, self.feature_type)
        self.progress = tqdm(total=len(self.path_list))
        self.output_direct = args.output_direct

    def forward(self, indices: torch.LongTensor):
        """
        Arguments:
            indices {torch.LongTensor} -- indices to self.path_list
        """
        device = indices.device

        # Define the model in inference mode, load the model
        # print(os.path.join(plb.Path(__file__).parent, "checkpoints"))
        model = VGGish(urls=model_urls, model_dir=os.path.join(plb.Path(__file__).parent, "checkpoints"),
                       postprocess=False, device=device).to(device)
        model.eval()

        # iterate over the list of videos
        for idx in indices:
            # when error occurs might fail silently when run from torch data parallel
            try:
                feats_dict = self.extract(model, self.path_list[idx])
                action_on_extraction(feats_dict, self.path_list[idx], self.output_path, self.on_extraction,
                                     output_direct=self.output_direct)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                traceback.print_exc()  # for the whole traceback
                print(e)
                print(f'Extraction failed at: {self.path_list[idx]}. Continuing extraction')

            # update tqdm progress bar
            self.progress.update()

    def extract(self, model, video_path: Union[str, None] = None) -> Dict[str, np.ndarray]:
        """The extraction call. Made to clean the forward call a bit.

        Args:
            model (Module): pytorch model
            video_path (Union[str, None], optional): . Defaults to None.

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as
                                             "path -> model"-fashion (default: {None})

        Returns:
            Dict[str, np.ndarray]: extracted VGGish features
        """
        file_ext = pathlib.Path(video_path).suffix

        if file_ext == '.mp4':
            # extract audio files from .mp4
            audio_wav_path, audio_aac_path = extract_wav_from_mp4(video_path, self.tmp_path)
        elif file_ext == '.wav':
            audio_wav_path = video_path
            audio_aac_path = None
        else:
            raise NotImplementedError

        # extract features
        vggish_stack = model.forward(audio_wav_path).cpu().detach().numpy()

        # removes the folder with audio files created during the process
        if not self.keep_tmp_files:
            if video_path.endswith('.mp4'):
                os.remove(audio_wav_path)
                os.remove(audio_aac_path)

        feats_dict = {
            self.feature_type: vggish_stack
        }

        return feats_dict
