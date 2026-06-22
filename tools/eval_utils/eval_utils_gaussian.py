
import pickle
import time
import numpy as np
import torch
import tqdm
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    return np.asarray(x)


def eval_one_epoch_gaussian(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    final_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('*************** EPOCH %s EVALUATION (Gaussian Mode) *****************' % epoch_id)
    if dist_test:
        logger.info('Distributed testing not fully supported in this custom script yet (use single GPU for safety)')

    model.eval()
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval')

    # Container for frame-level Gaussian uncertainty exports.
    gaussian_results = []

    # 容器：用于标准 mAP 评估
    det_annos = []
    dataset = dataloader.dataset
    class_names = dataset.class_names

    start_time = time.time()
    partial_save_interval = 500

    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            load_data_to_gpu(batch_dict)
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        # 处理预测结果
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos

        # 提取不确定性并保存到我们的自定义列表
        for batch_index, pred_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][batch_index]
            pred_boxes = to_numpy(pred_dict['pred_boxes'])
            pred_scores = to_numpy(pred_dict['pred_scores'])
            pred_labels = to_numpy(pred_dict['pred_labels'])

            # 获取 XYZ Uncertainty
            if 'pred_uncertainty' in pred_dict and pred_dict['pred_uncertainty'] is not None:
                pred_uncertainty = to_numpy(pred_dict['pred_uncertainty'])
            else:
                pred_uncertainty = np.zeros((pred_boxes.shape[0], 3))

            sample_result = {
                'frame_id': frame_id,
                'boxes_3d': pred_boxes,
                'scores': pred_scores,
                'labels': pred_labels,
                'uncertainty_xyz': pred_uncertainty
            }
            gaussian_results.append(sample_result)

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

            if (i + 1) % partial_save_interval == 0:
                partial_file = final_output_dir / f'result_with_uncertainty_epoch_{epoch_id}.partial.pkl'
                logger.info(
                    'Saving partial Gaussian results (%d/%d batches) to %s'
                    % (i + 1, len(dataloader), partial_file)
                )
                with open(partial_file, 'wb') as f:
                    pickle.dump(gaussian_results, f)

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    # 1. 保存带有 Uncertainty 的详细数据
    if cfg.LOCAL_RANK == 0:
        save_file = final_output_dir / f'result_with_uncertainty_epoch_{epoch_id}.pkl'
        logger.info(f'Saving Gaussian results to {save_file}')
        with open(save_file, 'wb') as f:
            pickle.dump(gaussian_results, f)

    # 2. 及其重要的步骤：运行标准评估 (mAP Calculation)
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / max(len(dataloader.dataset), 1)

    if cfg.LOCAL_RANK == 0:
        ret_dict = {}
        if dist_test:
            # 简化版: 分布式下合并结果略杂，此处假设单卡
            # 实际生产代码需要 all_gather(det_annos)
            pass

        has_gt_annos = all('annos' in info for info in getattr(dataset, 'kitti_infos', []))
        if not has_gt_annos:
            logger.info('No ground-truth annotations found. Skipping Kitti mAP evaluation for test split.')
            logger.info('Average predicted export time: %.4f sec per example' % sec_per_example)
            logger.info('****************Evaluation done.*****************')
            return {}

        logger.info('Generate Kitti-Style Results...')
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.get('EVAL_METRIC', 'kitti'),
            output_path=final_output_dir
        )

        logger.info(result_str)
        ret_dict.update(result_dict)

    logger.info('****************Evaluation done.*****************')
    return {}
