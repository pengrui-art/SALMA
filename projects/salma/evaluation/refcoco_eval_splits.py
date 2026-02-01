import argparse
import sys
import os

# Ensure repository root is on sys.path so 'projects.salma...' imports work when running this file directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import copy
import math
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np
import socket
import torch.multiprocessing as mp
import json

from transformers import (
    AutoModel,
    AutoTokenizer,
)
from transformers.configuration_utils import PretrainedConfig as _HFPretrainedConfig

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import RESDataset
from dataset.RES import rle_to_mask as _rle_to_mask


DATASETS_ATTRIBUTES = {
    "refcoco": {"splitBy": "unc", "dataset_name": "refcoco"},
    # NOTE: downstream REFER api expects folder name "refcoco+" in dataset; RESDataset handles that mapping
    "refcoco_plus": {"splitBy": "unc", "dataset_name": "refcoco_plus"},
    "refcocog": {"splitBy": "umd", "dataset_name": "refcocog"},
}

# Defaults retained for backward compatibility if not provided via flags/env.
DEFAULT_IMAGE_FOLDER = "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/glamm_data/images/coco2014/train2014/"
DEFAULT_DATA_PATH = "/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/ref_seg/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="RefCOCO family segmentation eval (multi-split)"
    )
    parser.add_argument("model_path", help="hf model path.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS_ATTRIBUTES.keys(),
        default="refcoco",
        help="Choose dataset: refcoco | refcoco_plus | refcocog",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Single split when --splits not set (val|testA|testB|test)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Multiple splits, e.g. --splits val testA testB or --splits all",
    )
    parser.add_argument(
        "--image-folder",
        default=os.environ.get("COCO2014_TRAIN", None),
        help="Path to COCO2014 train images folder. Can also set COCO2014_TRAIN env var.",
    )
    parser.add_argument(
        "--data-path",
        default=os.environ.get("REFCOCO_DATA", None),
        help="Path to RefCOCO dataset (refer-style folder). Can also set REFCOCO_DATA env var.",
    )
    parser.add_argument(
        "--work-dir",
        default=os.environ.get("WORK_DIR", os.path.join(_REPO_ROOT, "work_dirs")),
        help="Directory to save evaluation outputs/metrics.",
    )
    parser.add_argument(
        "--tmpdir",
        default=None,
        help="Temporary directory for distributed result collection.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index to use when launcher=='none' (e.g., --gpu 0). Ignored if using distributed launcher.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use in parallel when launcher=='none'. Will spawn that many processes and aggregate results.",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def _safe_config_repr(self):
    try:
        name = self.__class__.__name__
        model_type = getattr(self, "model_type", None)
        return f"{name}(model_type={model_type})" if model_type else name
    except Exception:
        return self.__class__.__name__


def _resolve_splits(args):
    if args.splits is not None:
        raw = [s.strip() for s in args.splits]
        if any(s.lower() == "all" for s in raw):
            return (
                ["val", "test"]
                if args.dataset == "refcocog"
                else ["val", "testA", "testB"]
            )
        return raw
    return [args.split]


def _normalize_split_names(splits):
    norm_map = {
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "testa": "testA",
        "test_a": "testA",
        "testb": "testB",
        "test_b": "testB",
        "testA": "testA",
        "testB": "testB",
    }
    return [
        norm_map.get(s if s in {"testA", "testB"} else s.lower(), s) for s in splits
    ]


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]["counts"] = rle[-1]["counts"].decode()
    return rle


def _decode_rle_list(rle_list):
    """Decode a list of pycocotools RLE dicts into a stacked np.ndarray.

    Returns an empty array when input is None or empty to simplify callers.
    """

    if rle_list is None:
        return np.zeros((0,), dtype=np.uint8)

    if len(rle_list) == 0:
        return np.zeros((0,), dtype=np.uint8)

    # Reuse dataset helper to keep behavior aligned with RESDataset.
    try:
        masks = _rle_to_mask(rle_list)
    except ValueError:
        # Guard against malformed inputs by falling back to an empty stack.
        return np.zeros((0,), dtype=np.uint8)

    if masks.ndim == 2:
        masks = np.expand_dims(masks, axis=0)
    return masks


def _compute_ciou_giou(results):
    """Compute CIoU/GIoU metrics mirroring refcoco_lira evaluation."""

    total_intersection = 0
    total_union = 0
    giou_sum = 0.0
    mask_count = 0

    for pred_dict in results:
        gt_stack = _decode_rle_list(pred_dict.get("gt_masks", []))
        if gt_stack.size == 0:
            continue

        prediction_list = pred_dict.get("prediction_masks", [])
        num_targets = gt_stack.shape[0]

        for idx in range(num_targets):
            target_mask = gt_stack[idx]
            mask_count += 1

            if idx >= len(prediction_list) or prediction_list[idx] is None:
                # Missing prediction contributes zero IoU but still counts towards gIoU average.
                continue

            pred_stack = _decode_rle_list(prediction_list[idx])
            if pred_stack.size == 0:
                continue

            prediction_mask = pred_stack[0]
            prediction_bin = prediction_mask.astype(np.uint8) > 0
            target_bin = target_mask.astype(np.uint8) > 0

            intersection = int(np.logical_and(prediction_bin, target_bin).sum())
            union = int(np.logical_or(prediction_bin, target_bin).sum())

            total_intersection += intersection
            total_union += union

            if union == 0:
                giou_sum += 1.0
            else:
                giou_sum += intersection / union

    ciou = total_intersection / total_union if total_union > 0 else 0.0
    giou = giou_sum / mask_count if mask_count > 0 else 0.0

    return {
        "pixel_intersection": float(total_intersection),
        "pixel_union": float(total_union),
        "ciou": float(ciou),
        "giou": float(giou),
        "mask_count": int(mask_count),
    }


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _json_default(o):
    try:
        import numpy as _np

        if isinstance(o, _np.generic):
            return o.item()
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    try:
        import torch as _torch

        if isinstance(o, _torch.Tensor):
            # best-effort conversion for any stray tensors
            if o.dim() == 0:
                return o.item()
            return o.detach().cpu().tolist()
    except Exception:
        pass
    # Fallback to string to avoid hard failures
    return str(o)


def run_one_split(
    model, tokenizer, dataset, tmpdir, world_size, rank, split_tag, work_dir
):
    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(
        per_rank_samples * rank, min(n_samples, per_rank_samples * (rank + 1))
    )
    for idx in tqdm.tqdm(per_rank_ids, desc=f"{dataset.METAINFO['name']}:{split_tag}"):
        data_batch = dataset[idx]
        prediction = {
            "img_id": data_batch["img_id"],
            "gt_masks": data_batch["gt_masks"],
        }
        prediction["gt_masks"] = mask_to_rle(prediction["gt_masks"].cpu().numpy())
        del data_batch["img_id"], data_batch["gt_masks"]

        texts = data_batch["text"]
        del data_batch["text"]
        pred_masks = []
        for text in texts:
            _data_batch = copy.deepcopy(data_batch)
            _data_batch["text"] = text
            with torch.no_grad():
                pred_mask = model.predict_forward(**_data_batch, tokenizer=tokenizer)[
                    "prediction_masks"
                ]
            if len(pred_mask) == 0:
                if get_rank() == 0:
                    print("No seg pred !!!")
                pred_masks.append(None)
            else:
                _ret_mask = pred_mask[0]
                _ret_mask = mask_to_rle(_ret_mask)
                pred_masks.append(_ret_mask)

        prediction.update({"prediction_masks": pred_masks})
        results.append(prediction)

    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0 and results is not None:
        metric = dataset.evaluate(results, work_dir)
        if metric is None:
            metric = {}
        # Augment metrics with CIoU/GIoU to align with refcoco_lira evaluation.
        if results:
            ciou_stats = _compute_ciou_giou(results)
            metric.update(ciou_stats)
            print(
                f"[Metrics] CIoU: {metric['ciou']:.4f} | GIoU: {metric['giou']:.4f} (mask_count={metric['mask_count']})"
            )
        # save per-split metric to JSON
        try:
            out_dir = os.path.join(work_dir, "eval_metrics")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir, f"{dataset.dataset_name}_{split_tag}_metric.json"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    metric, f, ensure_ascii=False, indent=2, default=_json_default
                )
        except Exception as e:
            print(f"[WARN] Failed to save metric JSON: {e}")
        return metric
    return None


def _distributed_worker(rank, world_size, port, args, splits):
    # set env for torch.distributed default init in our utils
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # init
    _init_dist_pytorch("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Make HF config repr safe
    try:
        _HFPretrainedConfig.__repr__ = _safe_config_repr  # type: ignore[attr-defined]
    except Exception:
        pass

    image_folder = args.image_folder or DEFAULT_IMAGE_FOLDER
    data_path = args.data_path or DEFAULT_DATA_PATH
    dataset_info = DATASETS_ATTRIBUTES[args.dataset]

    model = (
        AutoModel.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    all_metrics = {}
    for split in splits:
        dataset = RESDataset(
            image_folder=image_folder,
            dataset_name=dataset_info["dataset_name"],
            data_path=data_path,
            split=split,
        )
        if len(dataset) == 0:
            if get_rank() == 0:
                print(
                    f"[WARN] Split '{split}' has 0 samples for dataset '{args.dataset}'. Skipping."
                )
            continue

        tmpdir = args.tmpdir or (
            "./dist_test_temp_res_"
            + args.dataset
            + split
            + args.model_path.replace("/", "").replace(".", "")
        )
        metric = run_one_split(
            model, tokenizer, dataset, tmpdir, world_size, rank, split, args.work_dir
        )
        if metric is not None:
            all_metrics[split] = metric
            print({split: metric})

    # Print and save summary on rank 0
    if get_rank() == 0 and len(all_metrics) > 0:
        try:
            print("================ Summary ================")
            for s, m in all_metrics.items():
                print(f"Split: {s} -> {m}")
            print("=========================================")
            out_dir = os.path.join(args.work_dir, "eval_metrics")
            os.makedirs(out_dir, exist_ok=True)
            summary_path = os.path.join(
                out_dir, f"{dataset_info['dataset_name']}_summary.json"
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    all_metrics, f, ensure_ascii=False, indent=2, default=_json_default
                )
        except Exception as e:
            print(f"[WARN] Failed to save summary JSON: {e}")

    # finalize
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.barrier(device_ids=[rank])
        except TypeError:
            torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()

    # Make HF config repr safe for custom configs
    try:
        _HFPretrainedConfig.__repr__ = _safe_config_repr  # type: ignore[attr-defined]
    except Exception:
        pass

    # If user requests internal multi-GPU spawning and launcher=='none', spawn N processes
    if args.launcher == "none" and args.num_gpus and args.num_gpus > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("--num-gpus > 1 specified but CUDA is not available")
        num = torch.cuda.device_count()
        if args.num_gpus > num:
            raise ValueError(
                f"--num-gpus={args.num_gpus} exceeds available GPUs ({num})"
            )
        port = _find_free_port()
        splits = _normalize_split_names(_resolve_splits(args))
        mp.spawn(
            _distributed_worker,
            args=(args.num_gpus, port, args, splits),
            nprocs=args.num_gpus,
            join=True,
        )
        sys.exit(0)

    # Otherwise, proceed single-process or external-distributed
    if args.launcher != "none":
        _init_dist_pytorch("nccl")
        rank, world_size = get_dist_info()
        local_rank_env = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank_env)
    else:
        rank, world_size = 0, 1
        # Allow manual GPU selection when running single-process
        if args.gpu is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("--gpu specified but CUDA is not available")
            num = torch.cuda.device_count()
            if args.gpu < 0 or args.gpu >= num:
                raise ValueError(f"--gpu index {args.gpu} is out of range (0..{num-1})")
            torch.cuda.set_device(args.gpu)

    # resolve splits
    splits = _normalize_split_names(_resolve_splits(args))

    # paths
    image_folder = args.image_folder or DEFAULT_IMAGE_FOLDER
    data_path = args.data_path or DEFAULT_DATA_PATH
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(
            f"Image folder not found: {image_folder}. Use --image-folder or set COCO2014_TRAIN env var."
        )
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data path not found: {data_path}. Use --data-path or set REFCOCO_DATA env var."
        )

    # model/tokenizer
    model = (
        AutoModel.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # dataset info mapping
    dataset_info = DATASETS_ATTRIBUTES[args.dataset]

    # run per split
    all_metrics = {}
    for split in splits:
        dataset = RESDataset(
            image_folder=image_folder,
            dataset_name=dataset_info["dataset_name"],
            data_path=data_path,
            split=split,
        )
        if len(dataset) == 0:
            if get_rank() == 0:
                print(
                    f"[WARN] Split '{split}' has 0 samples for dataset '{args.dataset}'. Skipping."
                )
            continue

        tmpdir = args.tmpdir or (
            "./dist_test_temp_res_"
            + args.dataset
            + split
            + args.model_path.replace("/", "").replace(".", "")
        )
        metric = run_one_split(
            model, tokenizer, dataset, tmpdir, world_size, rank, split, args.work_dir
        )
        if metric is not None:
            all_metrics[split] = metric
            print({split: metric})

    # barrier and teardown
    if (
        args.launcher != "none"
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        try:
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])
        except TypeError:
            torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    if get_rank() == 0 and len(all_metrics) > 0:
        print("================ Summary ================")
        for s, m in all_metrics.items():
            print(f"Split: {s} -> {m}")
        print("=========================================")
