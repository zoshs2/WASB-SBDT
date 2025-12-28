import os
import os.path as osp
import shutil
import time
import logging
from collections import defaultdict
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from torch import nn
import cv2
import matplotlib.pyplot as plt

from dataloaders import build_dataloader
from detectors import build_detector
from trackers import build_tracker
from utils import mkdir_if_missing, draw_frame, gen_video, Center, Evaluator

from .base import BaseRunner

log = logging.getLogger(__name__)

@torch.no_grad()
def inference_video(
    detector,
    tracker,
    dataloader,
    cfg,
    vis_frame_dir=None,
    vis_hm_dir=None,
    vis_traj_path=None,
    evaluator_all=None,
    gt=None,
    dist_thresh=10.0,
):

    # 꼬리 길이 N (yaml에서 runner.tail_len으로 오버라이드 가능)
    try:
        tail_len = int(cfg["runner"].get("tail_len", 15))
    except Exception:
        tail_len = 15

    evaluator = None
    if evaluator_all is not None:
        evaluator = Evaluator(cfg)

    frames_in = detector.frames_in
    frames_out = detector.frames_out

    # +---------------
    t_start = time.time()

    det_results = defaultdict(list)
    hm_results = defaultdict(list)
    rescale = -1.0
    num_frames = 0

    for batch_idx, (imgs, hms, trans, xys_gt, visis_gt, img_paths) in enumerate(
        tqdm(dataloader, desc="[(CLIP-WISE INFERENCE)]")
    ):
        num_frames += imgs.shape[0] * frames_in
        if rescale < 0:
            rescale = trans[0][0, 0, 0].item()

        batch_results, hms_vis = detector.run_tensor(imgs, trans)
        img_paths = [list(in_tuple) for in_tuple in img_paths]

        for ib in batch_results.keys():
            for ie in batch_results[ib].keys():
                img_path = img_paths[ie][ib]
                preds = batch_results[ib][ie]
                det_results[img_path].extend(preds)
                hm_results[img_path].extend(hms_vis[ib][ie])

    tracker.refresh()
    result_dict = {}
    for img_path, preds in det_results.items():
        result_dict[img_path] = tracker.update(preds)

    t_elapsed = time.time() - t_start
    # +---------------
    log.info("Time:{:.1f}(sec)".format(t_elapsed))

    # 결과 프레임 순서를 리스트로 고정
    frame_paths = list(result_dict.keys())
    num_result_frames = len(frame_paths)

    cm_pred = plt.get_cmap("Reds", num_result_frames)
    cm_gt = plt.get_cmap("Greens", num_result_frames)

    fp1_im_list = []

    for idx, img_path in enumerate(frame_paths):
        pred_cur = result_dict[img_path]
        x_pred = pred_cur["x"]
        y_pred = pred_cur["y"]
        visi_pred = pred_cur["visi"]
        score_pred = pred_cur["score"]
        xy_pred = (x_pred, y_pred)

        center_gt = None
        if gt is not None:
            center_gt = gt[img_path]

        # 평가 로직은 그대로 유지
        if (gt is not None) and (evaluator_all is not None) and (center_gt is not None):
            evaluator_all.eval_single_frame(
                xy_pred,
                visi_pred,
                score_pred,
                center_gt.xy,
                center_gt.is_visible,
            )
            result = evaluator.eval_single_frame(
                xy_pred,
                visi_pred,
                score_pred,
                center_gt.xy,
                center_gt.is_visible,
            )

            if result["fp1"] > 0 and result["se"] < rescale * dist_thresh:
                fp1_im_list.append(img_path)

        # -------------------------
        # 시각화: GT는 "현재 프레임"만, 예측은 "최근 tail_len 프레임"만 꼬리처럼
        # -------------------------
        if vis_frame_dir is not None:
            vis_frame_path = osp.join(vis_frame_dir, osp.basename(img_path))
            vis_gt = cv2.imread(img_path)
            vis_pred = cv2.imread(img_path)

            # (1) GT: 현재 프레임의 GT만 (있을 때만)
            if gt is not None and center_gt is not None:
                color_gt = (
                    int(cm_gt(idx)[2] * 255),
                    int(cm_gt(idx)[1] * 255),
                    int(cm_gt(idx)[0] * 255),
                )
                vis_gt = draw_frame(
                    vis_gt,
                    center=center_gt,
                    color=color_gt,
                    radius=8,
                )

            # (2) Pred: 최근 tail_len 프레임만 꼬리처럼
            start = max(0, idx - tail_len + 1)
            end = idx  # inclusive
            tail_range = range(start, end + 1)

            for j in tail_range:
                path_j = frame_paths[j]
                pred_j = result_dict[path_j]

                if not pred_j["visi"]:
                    continue

                xj, yj = pred_j["x"], pred_j["y"]

                # j가 과거일수록 더 옅고 작게, 현재에 가까울수록 진하고 크게
                age = end - j  # 0 = 최신, tail_len-1 = 가장 오래된
                denom = max(1, tail_len - 1)
                w = 1.0 - (age / float(denom))  # 0~1, 1이 최신

                base = cm_pred(j)
                # 간단한 밝기 스케일링 (0.5~1.0 배 정도)
                brightness = 0.5 + 0.5 * w
                color_pred = (
                    int(base[2] * 255 * brightness),
                    int(base[1] * 255 * brightness),
                    int(base[0] * 255 * brightness),
                )
                # 꼬리: 오래된 점은 작고, 최신 점은 크게
                radius = int(3 + 5 * w)  # 대략 3~8 pixel

                vis_pred = draw_frame(
                    vis_pred,
                    center=Center(is_visible=True, x=xj, y=yj),
                    color=color_pred,
                    radius=radius,
                )

            vis = np.hstack((vis_gt, vis_pred))
            cv2.imwrite(vis_frame_path, vis)

        # vis_traj_path 쪽은 원본 코드에서도 visualizer 정의가 안 되어 있어서
        # vis_traj 기능을 쓰지 않는 한(기본 false) 건드리지 않고 패스
        if vis_traj_path is not None:
            # TODO: 필요하면 여기 별도 구현
            pass

    if vis_frame_dir is not None:
        video_path = "{}.mp4".format(vis_frame_dir)
        gen_video(video_path, vis_frame_dir, fps=25.0)

    if evaluator is not None:
        evaluator.print_results(with_ap=False)

    return fp1_im_list, {"t_elapsed": t_elapsed, "num_frames": num_frames}

class VideosInferenceRunner(BaseRunner):
    def __init__(self,
                 cfg: DictConfig,
                 clip_loaders_and_gts = None,
                 vis_result = None,
                 vis_hm = None,
    ):
        super().__init__(cfg)

        self._vis_result = cfg['runner']['vis_result']
        self._vis_hm     = cfg['runner']['vis_hm']
        if vis_result is not None:
            self._vis_result = vis_result
        if vis_hm is not None:
            self._vis_hm = vis_hm
        self._vis_traj = cfg['runner']['vis_traj']

        if clip_loaders_and_gts is None:
            split = cfg['runner']['split']
            if split=='train':
                _, _, self._clip_loaders_and_gts, _ = build_dataloader(cfg)
            elif split=='test':
                _, _, _, self._clip_loaders_and_gts = build_dataloader(cfg)
            else:
                raise ValueError('unknown split: {}'.format(split))
        else:
            self._clip_loaders_and_gts = clip_loaders_and_gts

    def run(self, model=None, model_dir=None):
        return self._run_model()

    def _run_model(self, model=None):
        #evaluator = build_evaluator(self._cfg)
        evaluator = Evaluator(self._cfg)
        detector  = build_detector(self._cfg, model=model)
        tracker   = build_tracker(self._cfg)

        t_elapsed_all = 0.
        num_frames_all   = 0
        fp1_im_list_dict = {}
        for key, dataloader_and_gt in self._clip_loaders_and_gts.items():
            match, clip_name = key
            dataloader = dataloader_and_gt['clip_loader']
            gt_dict    = dataloader_and_gt['clip_gt']

            vis_frame_dir, vis_hm_dir, vis_traj_path = None, None, None
            if self._vis_result:
                vis_frame_dir = osp.join(self._output_dir, '{}_{}'.format(match, clip_name) )
                mkdir_if_missing(vis_frame_dir)
            if self._vis_hm:
                vis_hm_dir = osp.join(self._output_dir, '{}_{}'.format(match, clip_name), 'hm')
                mkdir_if_missing(vis_hm_dir)
            if self._vis_traj:
                vis_traj_dir = osp.join(self._output_dir, 'vis_traj')
                mkdir_if_missing(vis_traj_dir)
                vis_traj_path = osp.join(vis_traj_dir, '{}_{}.png'.format(match, clip_name))

            log.info('eval @ match={}, clip={}'.format(match, clip_name))

            fp1_im_list, tmp = inference_video(detector, 
                            tracker, 
                            dataloader, 
                            self._cfg,
                            vis_frame_dir=vis_frame_dir, 
                            vis_hm_dir=vis_hm_dir, 
                            vis_traj_path=vis_traj_path,
                            evaluator_all=evaluator, 
                            gt=gt_dict)
            fp1_im_list_dict[key] = fp1_im_list
            
            t_elapsed_all += tmp['t_elapsed']
            num_frames_all += tmp['num_frames']

        log.info('-- TOTAL --')
        evaluator.print_results(txt='{} @ dist_threshold={}'.format(self._cfg['model']['name'], evaluator.dist_threshold), 
                                elapsed_time=t_elapsed_all, 
                                num_frames=num_frames_all)

        return {'prec': evaluator.prec, 
                'recall': evaluator.recall, 
                'f1': evaluator.f1, 
                'accuracy': evaluator.accuracy, 
                'rmse': evaluator.rmse, 
                'fp1_im_list_dict': fp1_im_list_dict}

