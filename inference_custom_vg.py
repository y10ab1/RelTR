# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import time
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
import torchvision
from models import build_model


import pandas as pd
import os
from tqdm import tqdm

from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from torchvision.ops import nms

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='demo/vg1.jpg',
                        help="Path of the test image")
    
    # image dir
    parser.add_argument('--img_dir', type=str, default='images',
                        help="Path of the images dir")
                        
    # results dir
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    parser.add_argument('--results_dir', type=str, default=f'results/results_{current_time}',
                        help="Path of the results dir")
    
    # log path
    parser.add_argument('--log_path', type=str, default=f'results/results_{current_time}/inference_log.csv',
                        help="Path of the log file")
    
    # confidence threshold
    parser.add_argument('-t', '--confidence_threshold', type=float, default=0.3,
                        help="Confidence threshold for filtering predictions")
    
    # topk
    parser.add_argument('--topk', type=int, default=10,
                        help="Top k predictions to show")
    
    # target class list
    parser.add_argument('--target_class_list', type=list, default=['person', 'dog', 'cat'],
                        help="Classes that we want to detect")
    
    # filter class list
    parser.add_argument('--filter_class_list', type=list, default=['leg', 'tail', 'eye', 'ear', 'mouth', 'nose', 'paw',
                                                                    'beak', 'wing', 'hair', 'face', 'head', 'neck',
                                                                    'hand', 'arm', 'finger', 'foot', 'toe', 'branch','pant','room',
                                                                    'shoe', 'jean', 'short', 'shirt', 'coat', 'jacket', 'glove', 'hat',],
                        help="Classes that we want to filter out")
                        
    

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='ckpt/checkpoint0149.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")


    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser


def main(args):
    

    transform = T.Compose([
        T.Resize(600),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    # VG classes
    CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

    REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']



    model, _, _ = build_model(args)
    #ckpt = torch.load(args.resume) # time consuming!!!

    model.load_state_dict(ckpt['model'])
    
    model.eval()

    img_path = args.img_path
    im = Image.open(img_path)
    

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    
    # put model and data to gpu if available
    if torch.cuda.is_available():
        model.to(args.device)
        img = img.to(args.device)
        
    
    # propagate through the model
    start = time.time()
    outputs = model(img)
    inference_time = time.time() - start
    #print("Time taken for inference: ", inference_time)
    
    # transfer outputs to cpu if its value is on cuda
    outputs = {k: v.cpu() for k, v in outputs.items() if torch.is_tensor(v)}
   
    

    # keep only predictions with 0.+ confidence
    confidence_threshold = args.confidence_threshold
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > confidence_threshold, 
                             torch.logical_and(probas_sub.max(-1).values > confidence_threshold,
                             probas_obj.max(-1).values > confidence_threshold))
    
    # keep only relationship triplets with subject or object containing target classes
    keep = retain_target_classes(probas_sub, probas_obj, keep, args.target_class_list, CLASSES)
    
    # filter out relationship triplets with subject or object containing filter classes
    keep = filter_out_classes(probas_sub, probas_obj, keep, args.filter_class_list, CLASSES)



    
    
    # load csv from log path, it may be an empty file or already contains a dataframe
    if os.path.exists(args.log_path) and os.path.getsize(args.log_path) > 0:
        df = pd.read_csv(args.log_path)
    else:
        df = pd.DataFrame(columns=['subject', 
                                    'relation', 
                                    'object', 
                                    'subject_bbox', 
                                    'object_bbox', 
                                    'image_path',
                                    'inference_time'])
    
    
    # alert if no triplets are found
    if keep.sum() == 0:
        print(f'No relation with confidence > {args.confidence_threshold} is found.')
        df = update_dataframe(df,
                            subject='N/A',
                            relation='N/A',
                            object='N/A',
                            subject_confidence='N/A',
                            relation_confidence='N/A',
                            object_confidence='N/A',
                            subject_bbox='N/A',
                            object_bbox='N/A',
                            image_path=img_path,
                            inference_time=round(inference_time, 4))
        df.to_csv(args.log_path, index=False)
        return


    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)
    
    
    
    
    
    
    # keep only top k predictions
    topk = args.topk
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    
    # retain the triplet with highest probas * probas_sub * probas_obj value within each unique triplet
    # e.g. if triplet (person, on, skateboard) has highest probas * probas_sub * probas_obj value,
    # then retain only this triplet and discard other triplets with the same subject, relation and object
    triplets = torch.cat((probas_sub[keep_queries].argmax(-1).unsqueeze(-1),
                            probas[keep_queries].argmax(-1).unsqueeze(-1),
                            probas_obj[keep_queries].argmax(-1).unsqueeze(-1)), dim=-1)
    # find indices for first occurence of each unique triplet
    unique_triplets = set()
    unique_indices = []
    for idx, triplet in enumerate(triplets):
        if tuple(triplet.tolist()) not in unique_triplets:
            unique_triplets.add(tuple(triplet.tolist()))
            unique_indices.append(idx)
    keep_queries = keep_queries[unique_indices]
    
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    keep_queries = keep_queries[indices]
    
    # TODO : Apply NMS to subject and object bounding boxes separately

    
    
    
    
    
    

    # use lists to store the outputs via up-values
    conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
            lambda self, input, output: dec_attn_weights_sub.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
            lambda self, input, output: dec_attn_weights_obj.append(output[1])
        )
    ]
    with torch.no_grad():
        # propagate through the model
        outputs = model(img)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        dec_attn_weights_sub = dec_attn_weights_sub[0]
        dec_attn_weights_obj = dec_attn_weights_obj[0]

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]
        im_w, im_h = im.size
        
        
        
        

        fig, axs = plt.subplots(ncols=len(indices), nrows=3, figsize=(22, 7))
        if len(indices) == 1:
            axs = axs.reshape(-1, 1)
        for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            
            ax = ax_i[0] 
            ax.imshow(dec_attn_weights_sub[0, idx].view(h, w).cpu())
            ax.axis('off')
            ax.set_title(f'query id: {idx.item()}')
            ax = ax_i[1] 
            ax.imshow(dec_attn_weights_obj[0, idx].view(h, w).cpu())
            ax.axis('off')
            ax = ax_i[2] 
            ax.imshow(im)
            ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                       fill=False, color='blue', linewidth=2.5))
            ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                       fill=False, color='orange', linewidth=2.5))

            ax.axis('off')
            
            relationship = CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()]
            confidence = probas[idx].max(-1)[0] * probas_sub[idx].max(-1)[0] * probas_obj[idx].max(-1)[0]
            
            ax.set_title(f'{relationship}\nconfidence: {confidence.item():.4f}', fontsize=10)
            
            # update the dataframe
            df = update_dataframe(df, 
                                  subject=CLASSES[probas_sub[idx].argmax()], 
                                  relation=REL_CLASSES[probas[idx].argmax()], 
                                  object=CLASSES[probas_obj[idx].argmax()],
                                  subject_confidence=probas_sub[idx].max(-1)[0].item(),
                                  relation_confidence=probas[idx].max(-1)[0].item(),
                                  object_confidence=probas_obj[idx].max(-1)[0].item(),
                                  subject_bbox=(sxmin.round().item(), symin.round().item(), sxmax.round().item(), symax.round().item()),
                                  object_bbox=(oxmin.round().item(), oymin.round().item(), oxmax.round().item(), oymax.round().item()),
                                  image_path=img_path,
                                  inference_time=round(inference_time, 4))
            
            
        fig.tight_layout()
        plt.savefig(f'{args.results_dir}/result_{os.path.basename(img_path)}')
        
        ### Create a merged figure with the list of top k predictions and original image
        
        df['total_confidence'] = df['subject_confidence'] * df['relation_confidence'] * df['object_confidence']
        df_to_show = df.tail(len(keep_queries)).sort_values(by='total_confidence', ascending=False)

        
        # Round the confidence values
        df_to_show[['subject_confidence', 'relation_confidence', 'object_confidence', 'total_confidence']] = \
            df_to_show[['subject_confidence', 'relation_confidence', 'object_confidence', 'total_confidence']].round(4)

        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))
        ax = axs[0]
        ax.imshow(im)
        for idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                    fill=False, color='blue', linewidth=2.5))
            ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                    fill=False, color='orange', linewidth=2.5))
            ax.text(sxmin, symin, CLASSES[probas_sub[idx].argmax()], color='white', fontsize=10, bbox=dict(facecolor='blue', alpha=0.5))
            ax.text(oxmin, oymin, CLASSES[probas_obj[idx].argmax()], color='white', fontsize=10, bbox=dict(facecolor='orange', alpha=0.5))
        ax.axis('off')
        ax.set_title('Original image with bounding boxes', fontsize=15, pad=15)  # Added padding here

        ax = axs[1]
        ax.axis('off')
        ax.set_title('Predicted relationship triplets', fontsize=15)
        headers = ['subject', 'relation', 'object', 'sub. conf.', 'rel. conf.', 'obj. conf.', 'total conf.']

        table = ax.table(cellText=df_to_show[['subject', 'relation', 'object', 'subject_confidence', 'relation_confidence', 'object_confidence', 'total_confidence']].values,
                        colLabels=headers,
                        loc='upper center',
                        #colWidths=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # Set the column widths
                        )
        table.scale(1, 2)  # Increased the row value
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.auto_set_column_width(col=list(range(len(headers))))
        fig.tight_layout()
        plt.savefig(f'{args.results_dir}/result_merge_{os.path.basename(img_path)}')
        # plt.close(fig)

        # save the dataframe to a csv file
        df.to_csv(args.log_path, index=False)


def update_dataframe(df:pd.DataFrame, **kwargs):
    # columns of dataframe :['subject', 'relation', 'object', 'subject_bbox', 'object_bbox', 'filepath']
    
    # save the subject, relation and object to the dataframe
    df = df.append(kwargs, ignore_index=True)

    return df
    
    
def retain_target_classes(probas_sub, probas_obj, keep, target_class_list, CLASSES):
    # keep only relationship triplets with subject or object containing 'person', 'dog' or 'cat'
    # i.e. only keep triplets with subject or object containing 'person', 'dog' or 'cat'
    keep_class_idx = [idx for idx, class_name in enumerate(CLASSES) if class_name in target_class_list]
    keep_class_idx_tensor = torch.tensor(keep_class_idx)
    
    # Get predicted classes
    pred_sub = probas_sub.argmax(dim=-1)
    pred_obj = probas_obj.argmax(dim=-1)

    # Expand dimensions to match for broadcasting
    pred_sub = pred_sub.unsqueeze(1)
    pred_obj = pred_obj.unsqueeze(1)

    # Check if these classes are in your target classes
    mask_sub = torch.any(pred_sub == keep_class_idx_tensor, dim=1)
    mask_obj = torch.any(pred_obj == keep_class_idx_tensor, dim=1)

    # Apply this mask to your `keep` tensor
    keep = torch.logical_and(keep, torch.logical_or(mask_sub, mask_obj))

    return keep


def filter_out_classes(probas_sub, probas_obj, keep, filter_class_list, CLASSES):
    # Get the indices of the classes to be filtered out
    filter_class_idx = [idx for idx, class_name in enumerate(CLASSES) if class_name in filter_class_list]
    filter_class_idx_tensor = torch.tensor(filter_class_idx)
    
    # Get predicted classes
    pred_sub = probas_sub.argmax(dim=-1)
    pred_obj = probas_obj.argmax(dim=-1)

    # Expand dimensions to match for broadcasting
    pred_sub = pred_sub.unsqueeze(1)
    pred_obj = pred_obj.unsqueeze(1)

    # Check if these classes are in your filter classes
    mask_sub = torch.any(pred_sub == filter_class_idx_tensor, dim=1)
    mask_obj = torch.any(pred_obj == filter_class_idx_tensor, dim=1)

    # Apply this mask to your `keep` tensor, but use logical_and and logical_not 
    # because we want to remove the filter classes, not keep them
    keep = torch.logical_and(keep, torch.logical_not(torch.logical_or(mask_sub, mask_obj)))
    
    return keep



if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # create results dir and log file
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    if not os.path.exists(os.path.dirname(args.log_path)):
        os.makedirs(os.path.dirname(args.log_path))
    
    ckpt = torch.load(args.resume) # time consuming!!!
    
    
    IMAGE_DIR = args.img_dir
    # IMAGE_DIR = 'summary_dataset/video/2023-03-26/002DB303B2DA_video/'
    
    
    
    # interate images from IMAGE_DIR and change arg.img_path with image file path
    tqdm_logger = tqdm(os.listdir(IMAGE_DIR))
    for img_path in tqdm_logger:
        tqdm_logger.set_description(f'Processing {img_path}')
        args.img_path = os.path.join(IMAGE_DIR, img_path)
        main(args)
    
    # transfer args to json file and save it for reference
    import json
    with open(f'{args.results_dir}/config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    
    print('Done.\n')