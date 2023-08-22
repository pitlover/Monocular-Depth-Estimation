import numpy as np


def cal_eval_mask(opt, gt_depth, data_type: str):
    # opt = opt["eval"]
    gt_height, gt_width = gt_depth.shape[-2:]
    eval_mask = np.zeros(gt_depth.shape[-2:], dtype=np.bool)

    if opt['garg_crop']:
        eval_mask[
        int(0.40810811 * gt_height):int(0.99189189 * gt_height),
        int(0.03594771 * gt_width):int(0.96405229 * gt_width)
        ] = 1

    elif opt['eigen_crop']:
        if (data_type == "KITTI") or (data_type == "ONLINE"):  # TODO is this right?
            eval_mask[
            int(0.3324324 * gt_height):int(0.91351351 * gt_height),
            int(0.0359477 * gt_width):int(0.96405229 * gt_width)
            ] = 1
        elif data_type == "NYU":
            eval_mask[45:471, 41:601] = 1
        else:
            raise ValueError(f"Unsupported data_type {data_type}.")

    else:  # should not be here
        raise ValueError("Unsupported crop configuration.")

    return eval_mask


def tcompute_errors(gt: np.ndarray, pred: np.ndarray) -> dict:
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3,
                abs_rel=abs_rel, sq_rel=sq_rel,
                rmse=rmse, rmse_log=rmse_log,
                silog=silog, log_10=log_10)
