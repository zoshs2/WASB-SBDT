'''
import os
import os.path as osp
import logging
from omegaconf import DictConfig
import cv2

from .base import BaseRunner
from utils import mkdir_if_missing

log = logging.getLogger(__name__)

def extract_frame_badminton(cfg):
    root_dir      = cfg['dataset']['root_dir']
    video_dirname = cfg['dataset']['video_dirname']
    frame_dirname = cfg['dataset']['frame_dirname']
    train_matches = cfg['dataset']['train']['matches']
    test_matches  = cfg['dataset']['test']['matches']
    overwrite     = cfg['runner']['overwrite']

    matches = train_matches + test_matches
    for match in matches:
        match_video_dir = osp.join(root_dir, match, video_dirname)
        video_names = os.listdir(match_video_dir)
        video_names.sort()
        for video_name in video_names:
            video_path = osp.join(match_video_dir, video_name)
            frame_dir  = osp.join(root_dir, match, frame_dirname, osp.splitext(video_name)[0])
            if osp.exists(frame_dir) and not overwrite:
                log.info('{} already exists. skip extracting frames'.format(frame_dir))
                continue

            log.info('extract frames in {} to {}'.format(video_path, frame_dir))
            mkdir_if_missing(frame_dir)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                assert 0, '{} cannot opened'.format(video_path)
            cnt = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    frame_path = osp.join(frame_dir, '{:05d}.png'.format(cnt))
                    cv2.imwrite(frame_path, frame)
                    cnt+=1
                else:
                    break

def extract_frame_soccer(cfg: DictConfig):
    root_dir      = cfg['dataset']['root_dir']
    video_dirname = cfg['dataset']['video_dirname']
    frame_dirname = cfg['dataset']['frame_dirname']
    train_videos  = cfg['dataset']['train']['videos']
    test_videos   = cfg['dataset']['test']['videos']
    img_ext       = cfg['dataset']['img_ext']
    video_ext     = cfg['dataset']['video_ext']
    overwrite     = cfg['runner']['overwrite']
    
    videos = train_videos + test_videos
    for video in videos:
        video_path = osp.join(root_dir, video_dirname, '{}{}'.format(video, video_ext) )
        frame_dir  = osp.join(root_dir, frame_dirname, video )
        if osp.exists(frame_dir) and not overwrite:
            log.info('{} already exists. skip extracting frames'.format(frame_dir))
            continue

        log.info('extract frames in {} to {}'.format(video_path, frame_dir))
        mkdir_if_missing(frame_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            assert 0, '{} cannot opened'.format(video_path)
        cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = osp.join(frame_dir, '{:05d}{}'.format(cnt, img_ext))
            cv2.imwrite(frame_path, frame)
            cnt+=1

def extract_frame(cfg: DictConfig):
    dataset_name = cfg['dataset']['name']
    if dataset_name=='badminton':
        extract_frame_badminton(cfg)
    elif dataset_name=='soccer':
        extract_frame_soccer(cfg)
    else:
        raise KeyError('for this dataset extrac_frame is not defined : {}'.format(dataset_name))

class ExtractFrameRunner(BaseRunner):
    def __init__(self,
                 cfg: DictConfig,
    ):
        super().__init__(cfg)
        self._dataset_name = cfg['dataset']['name']
        if not self._dataset_name in ['badminton', 'soccer']:
            raise KeyError('{} does not require frame extraction : {}'.format(dataset_name))
        
    def run(self):
        if self._dataset_name=='badminton':
            extract_frame_badminton(self._cfg)
        elif self._dataset_name=='soccer':
            extract_frame_soccer(self._cfg)
        else:
            raise KeyError('for this dataset extrac_frame is not defined : {}'.format(dataset_name))

            
'''
import os
import os.path as osp
import logging
from omegaconf import DictConfig
import cv2

from .base import BaseRunner
from utils import mkdir_if_missing

log = logging.getLogger(__name__)


def _extract_frames_from_video(video_path: str, out_dir: str, overwrite: bool, img_ext: str = ".png"):
    """
    Extract all frames from a single video into out_dir.
    """
    if osp.exists(out_dir) and not overwrite:
        log.info("%s already exists. skip extracting frames", out_dir)
        return

    mkdir_if_missing(out_dir)
    log.info("extract frames in %s to %s", video_path, out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"{video_path} cannot be opened")

    cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = osp.join(out_dir, f"{cnt:05d}{img_ext}")
        cv2.imwrite(frame_path, frame)
        cnt += 1

    cap.release()


def _resolve_match_video_dir(root_dir: str, match: str, video_dirname: str):
    """
    Support two common layouts:

    A) {root_dir}/{match}/{video_dirname}/<videos>
    B) {root_dir}/{video_dirname}/{match}/<videos>
    """
    cand_a = osp.join(root_dir, match, video_dirname)
    cand_b = osp.join(root_dir, video_dirname, match)

    if osp.isdir(cand_a):
        return cand_a, "A"
    if osp.isdir(cand_b):
        return cand_b, "B"

    raise FileNotFoundError(
        f"Cannot find video directory for match='{match}'. Tried:\n"
        f" - {cand_a}\n"
        f" - {cand_b}\n"
        f"Please place your mp4 under one of the paths above."
    )


def extract_frame_match_based(cfg: DictConfig, img_ext: str = None):
    """
    For datasets that are organized by matches:
      - list matches from cfg.dataset.train.matches + cfg.dataset.test.matches
      - find videos under match/video_dirname
      - save frames under match/frame_dirname/<video_stem> (layout A)
        or frame_dirname/match/<video_stem> (layout B)
    """
    # IMPORTANT: default arg cannot reference cfg; decide here
    if img_ext is None:
        # tennis.yaml has ext: '.jpg' (or similar)
        img_ext = cfg["dataset"].get("ext", ".png")

    root_dir = cfg["dataset"]["root_dir"]
    video_dirname = cfg["dataset"]["video_dirname"]
    frame_dirname = cfg["dataset"]["frame_dirname"]
    train_matches = cfg["dataset"]["train"]["matches"]
    test_matches = cfg["dataset"]["test"]["matches"]
    overwrite = cfg["runner"]["overwrite"]

    matches = list(train_matches) + list(test_matches)
    if len(matches) == 0:
        raise ValueError("No matches are configured. Set dataset.train.matches / dataset.test.matches.")

    for match in matches:
        match_video_dir, layout = _resolve_match_video_dir(root_dir, match, video_dirname)

        video_names = os.listdir(match_video_dir)
        video_names.sort()

        if len(video_names) == 0:
            log.warning("No videos found in %s", match_video_dir)
            continue

        for video_name in video_names:
            video_path = osp.join(match_video_dir, video_name)
            video_stem = osp.splitext(video_name)[0]

            # keep output layout consistent with the input layout
            if layout == "A":
                frame_base = osp.join(root_dir, match, frame_dirname)
            else:
                frame_base = osp.join(root_dir, frame_dirname, match)

            out_dir = osp.join(frame_base, video_stem)
            _extract_frames_from_video(video_path, out_dir, overwrite, img_ext=img_ext)


def extract_frame_soccer(cfg: DictConfig):
    """
    Soccer uses a different config schema: train.videos/test.videos and ext fields.
    """
    root_dir = cfg["dataset"]["root_dir"]
    video_dirname = cfg["dataset"]["video_dirname"]
    frame_dirname = cfg["dataset"]["frame_dirname"]
    train_videos = cfg["dataset"]["train"]["videos"]
    test_videos = cfg["dataset"]["test"]["videos"]
    img_ext = cfg["dataset"]["img_ext"]
    video_ext = cfg["dataset"]["video_ext"]
    overwrite = cfg["runner"]["overwrite"]

    videos = list(train_videos) + list(test_videos)
    for video in videos:
        video_path = osp.join(root_dir, video_dirname, f"{video}{video_ext}")
        out_dir = osp.join(root_dir, frame_dirname, video)
        _extract_frames_from_video(video_path, out_dir, overwrite, img_ext=img_ext)


def extract_frame(cfg: DictConfig):
    dataset_name = cfg["dataset"]["name"]

    # soccer uses its own schema
    if dataset_name == "soccer":
        extract_frame_soccer(cfg)
        return

    # match-based datasets (badminton/tennis/basketball/volleyball...)
    # use dataset.ext if present (e.g., '.jpg')
    extract_frame_match_based(cfg, img_ext=cfg["dataset"].get("ext", ".png"))


class ExtractFrameRunner(BaseRunner):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._dataset_name = cfg["dataset"]["name"]

        # minimal schema validation
        if self._dataset_name == "soccer":
            _ = cfg["dataset"]["train"]["videos"]
            _ = cfg["dataset"]["test"]["videos"]
        else:
            _ = cfg["dataset"]["train"]["matches"]
            _ = cfg["dataset"]["test"]["matches"]

    def run(self):
        extract_frame(self._cfg)