import os
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as _mask
from projects.salma.evaluation.utils import REFER, Summary, AverageMeter, intersectionAndUnionGPU, master_only

DATASETS_ATTRIBUTES = {
    'refcoco': {'splitBy': "unc", 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': "unc", 'dataset_name': 'refcoco+'},
    'refcocog': {'splitBy': "umd", 'dataset_name': 'refcocog'},
}

class RESDataset:
    METAINFO: dict = dict(name='Referring Expression Segmentation')

    def __init__(self,
                 image_folder,
                 dataset_name,
                 data_path=None,
                 split='val',
                 ):
        self.split = split
        self._set_attribute(dataset_name)
        json_datas = self.json_file_preprocess(data_path)
        self.json_datas = json_datas
        self.image_folder = image_folder

    def _set_attribute(self, dataset_name):
        attr_dict = DATASETS_ATTRIBUTES[dataset_name]
        self.splitBy = attr_dict['splitBy']
        self.dataset_name = attr_dict['dataset_name']

    def __len__(self):
        return len(self.json_datas)

    def real_len(self):
        return len(self.json_datas)

    def json_file_preprocess(self, data_path):
        splitBy = self.splitBy
        dataset_name = self.dataset_name
        refer_api = REFER(data_path, dataset_name, splitBy)
        ref_ids_train = refer_api.getRefIds(split=self.split)
        images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
        refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
        self.img2refs = self.create_img_to_refs_mapping(refs_train)

        image_infos = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        for item in loaded_images:
            item = item.copy()
            image_infos.append(item)

        self.annotations = refer_api.Anns
        refs = [self.img2refs[image_info['id']] for image_info in image_infos]

        ret = []
        for image_info, ref in zip(image_infos, refs):
            if len(ref) == 0:
                continue

            sents = []
            ann_ids = []
            for _ref in ref:
                for sent in _ref["sentences"]:
                    text = sent["sent"]
                    sents.append(text)
                    ann_ids.append(_ref["ann_id"])

            sampled_inds = list(range(len(sents)))
            sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
            sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
            selected_labels = sampled_sents
            ret.append(
                {'image_info': image_info,
                 'sampled_ann_id': sampled_ann_ids,
                 'selected_labels': selected_labels,
                 'image': image_info['file_name']
                 }
            )
        return ret

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def decode_mask(self, annotations_ids, image_info):
        flag = False
        masks = []

        for ann_id in annotations_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = self.annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = _mask.frPyObjects(
                                    ann["segmentation"], image_info["height"], image_info["width"], )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = _mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = self.annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = _mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = _mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)
        masks = np.stack(masks, axis=0)

        # if self.pad_image_to_square:
        masks = torch.from_numpy(masks)
        return masks

    def only_get_text_infos(self, json_data):
        return {'sampled_sents': json_data['selected_labels']}

    def get_questions(self, text_require_infos):
        sampled_sents = text_require_infos['sampled_sents']
        ret = []
        for sent in sampled_sents:
            ret.append("<image>\n Please segment {} in this image.".format(sent))
        return ret

    def filter_data_dict(self, data_dict):
        names = ['image', 'text', 'gt_masks', 'img_id']
        ret = {name: data_dict[name] for name in names}
        return ret

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = self.json_datas[index]
        text_require_infos = self.only_get_text_infos(data_dict)
        questions = self.get_questions(text_require_infos)

        assert data_dict.get('image', None) is not None
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image_file = os.path.join(self.image_folder, image_file)
            image = Image.open(image_file).convert('RGB')

            # process and get masks for evaluation
            masks = self.decode_mask(data_dict['sampled_ann_id'], data_dict['image_info'])
            data_dict['gt_masks'] = masks
            data_dict['image'] = image
            data_dict['text'] = questions
            data_dict['img_id'] = str(index)
        return self.filter_data_dict(data_dict)

    @master_only
    def evaluate(self, result, work_dir):
        # Initialize trackers for each size bucket
        buckets = ["<1%", "<2%", "<5%", "All"]
        trackers = {
            bucket: {
                "intersection": AverageMeter(f"Intersec_{bucket}", ":6.3f", Summary.SUM),
                "union": AverageMeter(f"Union_{bucket}", ":6.3f", Summary.SUM),
                "gIoU": AverageMeter(f"gIoU_{bucket}", ":6.3f", Summary.SUM)
            } for bucket in buckets
        }

        for pred_dict in result:
            masks = pred_dict['prediction_masks']
            _masks = []
            for mask in masks:
                if mask is not None:
                    mask = rle_to_mask(mask)
                _masks.append(mask)
            targets = pred_dict['gt_masks']
            _targets = rle_to_mask(targets)

            # Iterate over each sample in the batch/result
            for i_item, (_mask, _target) in enumerate(zip(_masks, _targets)):
                # Calculate Object Area Ratio for Stratification
                # _target shape is (H, W), values 0 or 1
                img_area = _target.shape[0] * _target.shape[1]
                obj_area = _target.sum()
                ratio = obj_area / max(img_area, 1)

                active_buckets = ["All"]
                if ratio < 0.01:
                    active_buckets.append("<1%")
                if ratio < 0.02:
                    active_buckets.append("<2%")
                if ratio < 0.05:
                    active_buckets.append("<5%")

                # Calculate IoU for this sample
                intersection, union, accuracy_iou = 0.0, 0.0, 0.0
                
                # If prediction is None (missing), treat as 0 IoU
                if _mask is None:
                    # Intersection is 0, Union is area of target
                    _intersect = 0.0
                    _union_ = float(obj_area)
                else:
                    prediction = torch.from_numpy(_mask).int().cuda()
                    target_tensor = torch.from_numpy(_target).int().cuda()
                    intersect, union_, _ = intersectionAndUnionGPU(
                        prediction.contiguous().clone(), target_tensor.contiguous(), 2, ignore_index=255
                    )
                    _intersect = intersect.cpu().numpy()
                    _union_ = union_.cpu().numpy()
                    
                    # intersect/union_ returns array [bg, fg], we want fg (index 1)
                    # intersectionAndUnionGPU returns (intersection, union, target_area) for each class
                    # Assuming class 1 is foreground
                    if len(_intersect) > 1:
                        _intersect = _intersect[1]
                        _union_ = _union_[1]
                    else:
                        # Should not happen with 2 classes, but safety fallback
                        _intersect = _intersect[0]
                        _union_ = _union_[0]

                # Update trackers for all applicable buckets
                for bucket in active_buckets:
                    trackers[bucket]["intersection"].update(_intersect)
                    trackers[bucket]["union"].update(_union_)
                    
                    # Calculate single sample IoU
                    sample_iou = _intersect / (_union_ + 1e-10)
                    trackers[bucket]["gIoU"].update(sample_iou, n=1)

        # Print Stratified Results
        print('\n' + '=' * 60)
        print(f"{'Metric':<15} | {'<1%':<10} | {'<2%':<10} | {'<5%':<10} | {'All':<10}")
        print('-' * 60)
        
        row_iou = "cIoU"
        row_giou = "gIoU"
        row_count = "Count"
        
        vals_ciou = []
        vals_giou = []
        vals_count = []
        
        final_acc = 0.0

        for bucket in ["<1%", "<2%", "<5%", "All"]:
            t = trackers[bucket]
            intersection = t["intersection"].sum
            union = t["union"].sum
            
            # Helper to handle scalar vs array
            if isinstance(intersection, (list, np.ndarray)):
                if len(intersection) > 1:
                    vis_intersection = intersection[1]
                    vis_union = union[1]
                else:
                    vis_intersection = intersection[0]
                    vis_union = union[0]
            else:
                vis_intersection = intersection
                vis_union = union

            class_iou = vis_intersection / (vis_union + 1e-10)
            global_iou = t["gIoU"].avg
            # Scalar checking
            if isinstance(global_iou, (list, np.ndarray)):
                 global_iou = global_iou[1] if len(global_iou) > 1 else global_iou[0]
            
            count = t["gIoU"].count
            
            vals_ciou.append(f"{class_iou:.4f}")
            vals_giou.append(f"{global_iou:.4f}")
            vals_count.append(f"{count}")
            
            if bucket == "All":
                final_acc = class_iou

        print(f"{row_iou:<15} | {vals_ciou[0]:<10} | {vals_ciou[1]:<10} | {vals_ciou[2]:<10} | {vals_ciou[3]:<10}")
        print(f"{row_giou:<15} | {vals_giou[0]:<10} | {vals_giou[1]:<10} | {vals_giou[2]:<10} | {vals_giou[3]:<10}")
        print(f"{row_count:<15} | {vals_count[0]:<10} | {vals_count[1]:<10} | {vals_count[2]:<10} | {vals_count[3]:<10}")
        print('=' * 60 + '\n')
        
        print(f'RES_{self.dataset_name}_{self.split} successfully finished evaluating')
        return {'Acc': final_acc}


def rle_to_mask(rle):
    mask = []
    for r in rle:
        m = _mask.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask
