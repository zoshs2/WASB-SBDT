'''
import os
import os.path as osp
from typing import Tuple
from tqdm import tqdm
import cv2

from utils import Center

def draw_frame(img_or_path,
               center: Center,
               color: Tuple,
               radius : int = 5,
               thickness : int = -1,
    ):
        if osp.isfile(img_or_path):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path

        xy   = center.xy
        visi = center.is_visible
        if visi:
            x, y = xy
            x, y = int(x), int(y)
            img  = cv2.circle(img, (x,y), radius, color, thickness=thickness)
        
        return img
        
def gen_video(video_path, 
              vis_dir, 
              resize=1.0, 
              fps=30.0, 
              fourcc='mp4v'
):

    fnames = os.listdir(vis_dir)
    fnames.sort()
    h,w,_   = cv2.imread(osp.join(vis_dir, fnames[0])).shape
    im_size = (int(w*resize), int(h*resize))
    fourcc  = cv2.VideoWriter_fourcc(*fourcc)
    out     = cv2.VideoWriter(video_path, fourcc, fps, im_size)

    for fname in tqdm(fnames):
        im_path = osp.join(vis_dir, fname)
        im      = cv2.imread(im_path)
        im = cv2.resize(im, None, fx=resize, fy=resize)
        out.write(im)

'''

'''
import os
import os.path as osp
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

from utils import Center


def draw_frame(
    img_or_path,
    center: Center,
    color: Tuple,
    radius: int = 5,
    thickness: int = -1,
):
    """
    img_or_path:
      - str : 이미지 파일 경로
      - np.ndarray : 이미 로드된 이미지 배열
    """
    # 문자열인 경우에만 파일 여부 체크
    if isinstance(img_or_path, str) and osp.isfile(img_or_path):
        img = cv2.imread(img_or_path)
    else:
        # 그 외에는 이미지를 그대로 쓴다고 가정
        img = img_or_path

    # 안전장치: ndarray가 아니면 에러
    if not isinstance(img, np.ndarray):
        raise TypeError(
            f"draw_frame expected numpy.ndarray or valid image path, "
            f"got {type(img_or_path)}"
        )

    xy = center.xy
    visi = center.is_visible
    if visi:
        x, y = xy
        x, y = int(x), int(y)
        img = cv2.circle(img, (x, y), radius, color, thickness=thickness)

    return img


def gen_video(
    video_path,
    vis_dir,
    resize: float = 1.0,
    fps: float = 30.0,
    fourcc: str = "mp4v",
):
    fnames = os.listdir(vis_dir)
    fnames.sort()
    if len(fnames) == 0:
        raise RuntimeError(f"No frames found in {vis_dir}")

    first_img = cv2.imread(osp.join(vis_dir, fnames[0]))
    if first_img is None:
        raise RuntimeError(f"Failed to read first frame: {osp.join(vis_dir, fnames[0])}")

    h, w, _ = first_img.shape
    im_size = (int(w * resize), int(h * resize))
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(video_path, fourcc_code, fps, im_size)

    for fname in tqdm(fnames):
        im_path = osp.join(vis_dir, fname)
        im = cv2.imread(im_path)
        if im is None:
            continue
        im = cv2.resize(im, None, fx=resize, fy=resize)
        out.write(im)

    out.release()
    '''

import os
import os.path as osp
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm
import logging

from utils import Center

log = logging.getLogger(__name__)


def draw_frame(
    img_or_path,
    center: Center,
    color: Tuple,
    radius: int = 5,
    thickness: int = -1,
):
    """
    img_or_path:
      - str : 이미지 파일 경로
      - np.ndarray : 이미 로드된 이미지 배열
    """
    # 문자열인 경우에만 파일 여부 체크
    if isinstance(img_or_path, str) and osp.isfile(img_or_path):
        img = cv2.imread(img_or_path)
    else:
        # 그 외에는 이미 로드된 이미지라고 가정
        img = img_or_path

    # 안전장치: ndarray가 아니면 에러
    if not isinstance(img, np.ndarray):
        raise TypeError(
            f"draw_frame expected numpy.ndarray or valid image path, "
            f"got {type(img_or_path)}"
        )

    xy = center.xy
    visi = center.is_visible
    if visi:
        x, y = xy
        x, y = int(x), int(y)
        img = cv2.circle(img, (x, y), radius, color, thickness=thickness)

    return img


def gen_video(
    video_path,
    vis_dir,
    resize: float = 1.0,
    fps: float = 30.0,
    fourcc: str = "mp4v",
):
    # 1) 디렉터리 존재 여부 체크
    if not osp.isdir(vis_dir):
        log.warning(
            "gen_video: vis_dir '%s' does not exist. Skip video generation.", vis_dir
        )
        return

    fnames = os.listdir(vis_dir)
    fnames.sort()

    # 2) 파일이 하나도 없으면 스킵
    if len(fnames) == 0:
        log.warning(
            "gen_video: no frames found in '%s'. Skip video generation.", vis_dir
        )
        return

    first_path = osp.join(vis_dir, fnames[0])
    first_img = cv2.imread(first_path)
    if first_img is None:
        log.warning(
            "gen_video: failed to read first frame '%s'. Skip video generation.",
            first_path,
        )
        return

    h, w, _ = first_img.shape
    im_size = (int(w * resize), int(h * resize))
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(video_path, fourcc_code, fps, im_size)

    for fname in tqdm(fnames, desc=f"[gen_video] {os.path.basename(vis_dir)}"):
        im_path = osp.join(vis_dir, fname)
        im = cv2.imread(im_path)
        if im is None:
            continue
        im = cv2.resize(im, None, fx=resize, fy=resize)
        out.write(im)

    out.release()
    log.info("gen_video: saved video to '%s'", video_path)
