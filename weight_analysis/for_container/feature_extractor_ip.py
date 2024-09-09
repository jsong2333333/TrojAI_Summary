import torch
import os
import numpy as np
import json
from itertools import product
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from collections import OrderedDict
from itertools import product
import random as rand


ORIGINAL_LEARNED_PARAM_DIR = './learned_parameters'
DATA_PATH = '/scratch/data/TrojAI/object-detection-feb2023-train/models/'
MODEL_ARCH = ['FasterRCNN', 'SSD', 'DetrForObjectDetection']
LOSS_TYPES = ['entr_label', 'cen_label', 'l1_box', 'giou_box']
# LOSS_TYPE_BY_MA = {ma:loss_type for ma, loss_type in 
#                    zip(MODEL_ARCH, [[1], [3], [0]])}
ALL_PERTURBS = []
for perturb_type in ['condensed_triggers_stacked_intersection']: #['random_noise_input', 'train_triggers', 'random_color_filter']:
    ALL_PERTURBS.append(torch.from_numpy(np.load(os.path.join(ORIGINAL_LEARNED_PARAM_DIR, f'{perturb_type}.npy'))))
# with open(os.path.join(ORIGINAL_LEARNED_PARAM_DIR, f'useful_inds_by_ma_loss_type.json'), 'r') as indfile:
#     BEST_FE_INDS = json.load(indfile)
    
l1_smooth_loss = torch.nn.SmoothL1Loss(reduction='mean')
cen_loss = torch.nn.CrossEntropyLoss()


def load_model(model_filepath: str):
    model = torch.load(model_filepath)
    model_class = model.__class__.__name__
    model_repr = OrderedDict({layer: tensor for (layer, tensor) in model.state_dict().items()})
    return model, model_repr, model_class


def _process_img(img_filepath, resize=False, resize_res=16, padding=False, padding_pos='middle', repeat_num=3, canvas_size=256, rotate=False):
    if img_filepath.endswith('png'):
        img = torchvision.io.read_image(img_filepath, torchvision.io.ImageReadMode.RGB_ALPHA)
    else:
        img = torchvision.io.read_image(img_filepath)
    image = torch.as_tensor(img)
    # image = image.permute((2, 0, 1))
    images = None
    if rotate:
        image = T.functional.rotate(image, 180)
    if resize:
        resize_transforms = T.Resize(size=(resize_res, resize_res))
        image = resize_transforms(image)
    if padding:
        img_size = resize_res if resize else image.shape[0]
        if padding_pos == 'middle':
            p_trans = T.Pad(padding=(canvas_size-img_size)//2, padding_mode='constant', fill=0)
            image = p_trans(image)
        else:
            padding_transforms = []
            mid = int(canvas_size/2)
            padding_slot = int(mid/repeat_num - img_size/2)
            if repeat_num % 2 == 0:
                positions = []
                for rep in range(repeat_num // 2):
                    positions.append(int(mid-(2*rep+1)*(img_size+2*padding_slot)/2))
                    positions.append(int(mid+(2*rep+1)*(img_size+2*padding_slot)/2))
            else:
                positions = [mid]
                for rep in range(1, repeat_num // 2 + 1):
                    positions.append(mid-rep*(img_size-2*padding_slot))
                    positions.append(mid+rep*(img_size+2*padding_slot))
            for p1, p2 in list(product(positions, positions)):
                left, top = int(p1 - img_size/2), int(p2 - img_size/2)
                p_trans = T.Pad(padding=(left, top, canvas_size-img_size-left, canvas_size-img_size-top), padding_mode='constant', fill=0)
                padding_transforms.append(p_trans)
            if padding_pos in ['arr', 'rand']:
                images = []
                for p_trans in padding_transforms:
                    images.append(p_trans(image))
                if padding_pos == 'rand':
                    r = rand.randint(0, len(images)-1)
                    images = images[r]
            else:
                images = 0
                for p_trans in padding_transforms:
                    images += p_trans(image)
    augmentation_transforms = T.Compose([T.ConvertImageDtype(torch.float)]) 
    if images is not None:
        if isinstance(images, list):
            return [augmentation_transforms(image) for image in images]
        else:
            return augmentation_transforms(images)
    return augmentation_transforms(image)


def _center_to_corners_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    Provided by NIST.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - (0.5 * w)), (y_c - (0.5 * h)), (x_c + (0.5 * w)), (y_c + (0.5 * h))]
    return torch.stack(b, dim=-1)


def _detr_output_process(img_shape = None, pred_box = None, logit = None, device=torch.device('cpu')):
    ret = {}
    # boxes from DETR emerge in center format (center_x, center_y, width, height) in the range [0,1] relative to the input image size
    # convert to [x0, y0, x1, y1] format
    if pred_box is not None and img_shape is not None:
        boxes = _center_to_corners_format(pred_box)
        # clamp to [0, 1]
        boxes = torch.clamp(boxes, min=0, max=1)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h = img_shape[0] * torch.ones(1)  # 1 because we only have 1 image in the batch
        img_w = img_shape[1] * torch.ones(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(device)
        boxes = boxes * scale_fct[:, None, :]
        boxes = boxes[0,]
        ret['boxes'] = boxes

    # unpack the logits to get scores and labels
    if logit is not None:
        prob = torch.nn.functional.softmax(logit, -1)
        scores, labels = prob[..., :-1].max(-1)
        scores = scores[0,]
        labels = labels[0,]
        ret['scores'] = scores
        ret['labels'] = labels    

    return ret


def get_sample_images(examples_dirpath):
    clean_images, all_clean_label_to_boxes = [], []
    for example_image_fp in os.listdir(examples_dirpath):
        if example_image_fp.endswith('.png'):
            clean_image = _process_img(os.path.join(examples_dirpath, example_image_fp))
            clean_images.append(clean_image)

            with open(os.path.join(examples_dirpath, f'{example_image_fp[:-4]}.json')) as ground_truth_file:
                bbox_labels = json.load(ground_truth_file)[1:]
            clean_label_to_boxes = {}
            for bbox_label in bbox_labels:
                box, label = bbox_label['bbox'], bbox_label['label']
                clean_label_to_boxes.setdefault(label, []).append([box['x1'], box['y1'], box['x2'], box['y2']])
            all_clean_label_to_boxes.append(clean_label_to_boxes)
    return clean_images, all_clean_label_to_boxes


def _get_eigen_vals(filtered_model_repr: OrderedDict, target_layers=[], percentage=.5):
    features, feature_lengths = [], []
    for num_layer, backbone_params in enumerate(filtered_model_repr.values()):
        if num_layer in target_layers:
            reshaped_params = backbone_params.reshape(backbone_params.shape[0], -1)
            _, singular_values, _ = torch.linalg.svd(reshaped_params,False)
            squared_singular_values = singular_values**2
            ssv = squared_singular_values.detach().cpu().tolist()
            features += ssv[:int(percentage*len(ssv))]
            feature_lengths.append(len(features))
    return features, feature_lengths


def _get_weight_features(filtered_model_repr: OrderedDict, target_layers=[], dim=(0, 1, 2), normalize=False):
    weight_features = []
    for num_layer, backbone_params in enumerate(filtered_model_repr.values()):
        if num_layer in target_layers:
            if normalize:
                norm = torch.linalg.norm(backbone_params.reshape(backbone_params.shape[0], -1), ord=2)
                backbone_params  = backbone_params/norm
            weight_features += torch.amax(backbone_params, dim=dim).flatten().detach().cpu().tolist()
            weight_features += torch.mean(backbone_params, dim=dim).flatten().detach().cpu().tolist()
            end_dim = -1*(len(backbone_params.shape) - len(dim)) - 1
            sub = torch.mean(backbone_params, dim=dim) - torch.median(torch.flatten(backbone_params, start_dim=0, end_dim=end_dim), dim=end_dim)[0]
            weight_features += sub.flatten().detach().cpu().tolist()
            weight_features += torch.median(torch.flatten(backbone_params, start_dim=0, end_dim=end_dim), dim=end_dim)[0].flatten().detach().cpu().tolist()
            weight_features += torch.sum(backbone_params, dim=dim).flatten().detach().cpu().tolist()
    return weight_features


def _get_stats_from_weight_features(weight, axis= (0,), normalized=False) -> list:
    params = []
    
    try:
        norm = torch.linalg.norm(weight, ord=2)
    except:
        norm = torch.linalg.norm(weight.reshape(weight.shape[0], -1), ord=2)
    
    if not normalized:
        norm = 1

    weight /= norm
    p_max = torch.amax(weight).detach().cpu().tolist()
    p_min = torch.amin(weight).detach().cpu().tolist()
    p_mean = torch.mean(weight).detach().cpu().tolist()
    p_median = torch.median(weight).detach().cpu().tolist()
    p_sub = p_mean - p_median
    p_sum = torch.sum(torch.abs(weight)).detach().cpu().tolist()

    try:
        p_rank = [(torch.linalg.norm(weight, ord='fro')**2/torch.linalg.norm(weight, ord=2)**2).detach().cpu().tolist()]
        for ord in [2, 'fro', torch.inf, -torch.inf, 'nuc']:
            p_rank.append(torch.linalg.norm(weight, ord=ord).detach().cpu().tolist())
    except:
        reshaped_weight = weight.reshape(weight.shape[0], -1)
        p_rank = [(torch.linalg.norm(reshaped_weight, ord='fro')**2/torch.linalg.norm(reshaped_weight, ord=2)**2).detach().cpu().tolist()]
        for ord in [2, 'fro', torch.inf, -torch.inf, 'nuc']:
            p_rank.append(torch.linalg.norm(reshaped_weight, ord=ord).detach().cpu().tolist())

    params = [p_max, p_min, p_mean, p_sub, p_median, p_sum] + p_rank
    return params


def get_model_features(model, model_class, model_repr, clean_sample_images, all_clean_label_to_boxes, device=torch.device('cpu'), infer=True):    
    features = []

    features.extend(_get_ip_features(model, model_class, clean_sample_images, all_clean_label_to_boxes, device=device))
    # features.extend(_get_wa_features(model_repr, model_class, device))
    if infer:
        return np.asarray([features])
    else:
        return features


def _get_wa_features(model_repr: OrderedDict, model_class: str, device=torch.device('cpu')):
    feature = []
    if model_class == 'SSD':
        return feature
    
    filtered_model_repr = OrderedDict({
        layer: tensor.to(device) for layer, tensor in model_repr.items() 
        if 'backbone' not in layer and len(tensor.shape) > 1})
    
    for weight in filtered_model_repr.values():
        feature += _get_stats_from_weight_features(weight)

    return feature


def _get_ip_features(model, model_class, clean_sample_images, all_clean_label_to_boxes, device=torch.device('cpu')):
    model = model.to(device)
    model.eval();
    
    loss_types_to_compute = range(4)  #LOSS_TYPE_BY_MA[model_class]

    common_inds = ALL_PERTURBS[0].shape[0]  #set()
    
    batch_size = 8
    sample_size = len(clean_sample_images)
    clean_images = torch.stack(clean_sample_images, dim=0)
    
    results_by_loss_type = {lt:[] for lt in loss_types_to_compute}
    
    for p_ind in range(common_inds):
        pt_p = ALL_PERTURBS[0][p_ind]
        perturbs = torch.stack([pt_p]*sample_size, dim=0)
        # perturbed_images = torch.where(perturbs !=0, perturbs, clean_images)
        perturbed_images = torch.where(perturbs == 0, clean_images, perturbs)[:, :3, :]
            
        eval_loss_type_by_ind = range(4)

        results_by_single_image = {lt:0 for lt in eval_loss_type_by_ind}
        ind_size = sample_size//batch_size if sample_size % batch_size == 0 else sample_size//batch_size + 1
                
        for ind in range(ind_size):
            perturbed_image_per_batch = perturbed_images[ind*batch_size:(ind+1)*batch_size]
            output_per_batch = model(perturbed_image_per_batch.to(device))
            
            for b in range(perturbed_image_per_batch.shape[0]):
                if model_class == 'DetrForObjectDetection':
                    logit = output_per_batch.logits[b].unsqueeze(0)
                    pred_box = output_per_batch.pred_boxes[b].unsqueeze(0)
                    output = _detr_output_process(logit=logit, pred_box=pred_box, img_shape=[256, 256], device=device)
                else:
                    output = output_per_batch[b]
                
                box, score, label = output['boxes'].detach().cpu(), output['scores'].detach().cpu(), output['labels'].detach().cpu()
                
                if 0 in eval_loss_type_by_ind:
                    results_by_single_image[0] += -torch.nansum(score*torch.log2(score)).item()
                    if len(eval_loss_type_by_ind) == 1:
                        continue
                
                clean_label_to_boxes = all_clean_label_to_boxes[ind*batch_size+b]
                label_set = set(clean_label_to_boxes.keys())
                label = (label-1).tolist()
                
                label_to_boxes, label_to_scores = {}, {}
                for l_ind in range(len(label)):
                    l = label[l_ind]
                    label_to_boxes.setdefault(l, []).append(box[l_ind].tolist())
                    label_to_scores.setdefault(l, []).append(score[l_ind].item())
                    label_set.add(l)

                for l in list(label_set):
                    c_box = clean_label_to_boxes[l] if l in clean_label_to_boxes else []
                    p_box = label_to_boxes[l] if l in label_to_boxes else []
                    c_box_len, p_box_len = len(c_box), len(p_box)
                    len_max = max(c_box_len, p_box_len)
                    
                    c_box += [[0.]*4]*(len_max - c_box_len)
                    p_box += [[0.]*4]*(len_max - p_box_len)
                    
                    c_score = [1.]*c_box_len + [0.]*(len_max - c_box_len)
                    p_score = label_to_scores[l] if l in label_to_scores else []
                    p_score += [0.]*(len_max - p_box_len)
                    
                    assert len(p_score) == len(c_score) == len(p_box) == len(c_box)
                    
                    if 1 in eval_loss_type_by_ind:  #cen
                        results_by_single_image[1] += cen_loss(torch.as_tensor(c_score), torch.as_tensor(p_score)).item()
                    if 2 in eval_loss_type_by_ind:  #smooth l1
                        results_by_single_image[2] += l1_smooth_loss(torch.as_tensor(c_box), torch.as_tensor(p_box)).item()
                    if 3 in eval_loss_type_by_ind:  #giou
                        results_by_single_image[3] += torchvision.ops.generalized_box_iou_loss(torch.as_tensor(c_box), torch.as_tensor(p_box), reduction='mean').item()        
        
                del box, score, label, output
            
            del output_per_batch
            
        for lt, val in results_by_single_image.items():
            results_by_loss_type[lt].append(val/sample_size)
    
    del model
    
    ip_features = []    
    for loss_type in loss_types_to_compute:
        ip_features.extend(results_by_loss_type[loss_type])
        
    return ip_features


def _load_useful_wa_inds(model_class: str):
    return np.load(os.path.join(ORIGINAL_LEARNED_PARAM_DIR, f'{model_class}_wa_useful_inds.npy'))