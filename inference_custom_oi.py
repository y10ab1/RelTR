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

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='oi')

    # image path
    parser.add_argument('--img_path', type=str, default='demo/vg1.jpg',
                        help="Path of the test image")
    
    # log path
    parser.add_argument('--log_path', type=str, default='inference_log.csv',
                        help="Path of the log file")
                        
    # results dir
    parser.add_argument('--results_dir', type=str, default='results',
                        help="Path of the results dir")
                        

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
    parser.add_argument('--resume', default='ckpt/checkpoint0149_oi.pth', help='resume from checkpoint')
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
    CLASSES = ['Tennis ball', 'Shotgun', 'Bicycle', 'Potato', 'Table', 'Cantaloupe', 'Crab', 'Drawer', 'Beer', 
               'Dinosaur', 'Popcorn', 'Adhesive tape', 'Parachute', 'Seahorse', 'Pen', 'Bowling equipment', 'Camera', 
               'Saxophone', 'Train', 'Canoe', 'Trumpet', 'Pear', 'Unicycle', 'Flowerpot', 'Serving tray', 'Axe', 'Billiard table', 
               'Burrito', 'Guitar', '__background__', 'Candy', 'Hammer', 'Punching bag', 'Ski', 'Skateboard', 'Croissant', 
               'Box', 'Kite', 'Snowmobile', 'Mixing bowl', 'Pitcher (Container)', 'Spoon', 'Torch', 'Binoculars', 'Belt', 
               'Lobster', 'Cowboy hat', 'Mug', 'Mango', 'Grapefruit', 'Candle', 'Alpaca', 'Dog', 'Grape', 'Frying pan', 
               'Sewing machine', 'Sword', 'Football helmet', 'Broccoli', 'Egg (Food)', 'Handbag', 'Common fig', 'Microphone', 
               'Snowplow', 'Truck', 'Bottle', 'Indoor rower', 'Musical keyboard', 'Strawberry', 'Bow and arrow', 'Pasta', 
               'Pancake', 'Cat', 'Scissors', 'Tablet computer', 'Chair', 'Cart', 'Boat', 'Boy', 'Fedora', 'Oyster', 'Pizza', 
               'Man', 'Sombrero', 'Wine glass', 'Golf ball', 'Cannon', 'Corded phone', 'High heels', 'Watch', 'Balance beam', 
               'Tea', 'Lantern', 'Desk', 'Rifle', 'Salad', 'Crocodile', 'Coconut', 'Scarf', 'Boot', 'Limousine', 'Microwave oven', 
               'Lemon', 'Necklace', 'Pineapple', 'Football', 'Tree', 'Jaguar (Animal)', 'Bench', 'Wok', 'Tank', 'Doll', 'Bread', 'Glasses', 
               'Toilet paper', 'Sunglasses', 'Backpack', 'Teddy bear', 'Hamster', 'Tin can', 'Dumbbell', 'Gondola', 'Baseball glove', 
               'Shrimp', 'Balloon', 'Book', 'Accordion', 'Flute', 'Ice cream', 'Tomato', 'Closet', 'Waste container', 'Wheelchair', 
               'Bicycle helmet', 'Peach', 'Harbor seal', 'Sofa bed', 'Girl', 'Cupboard', 'Horizontal bar', 'Chopsticks', 'Harp', 
               'Studio couch', 'Houseplant', 'Trombone', 'Bagel', 'Radish', 'Cucumber', 'Crown', 'Cocktail', 'Cutting board', 
               'Drum', 'Stretcher', 'Watermelon', 'Common sunflower', 'Personal flotation device', 'Stool', 'Cheese', 'Coffee cup', 
               'Cricket ball', 'Elephant', 'Harpsichord', 'Lavender (Plant)', 'Infant bed', 'Pretzel', 'Suitcase', 'Sun hat', 'Violin', 'Whale', 
               'Ladder', 'Zucchini', 'Cake stand', 'Countertop', 'Hamburger', 'Briefcase', 'Wine', 'Snowboard', 'Plastic bag', 'Baseball bat', 
               'Flying disc', 'Juice', 'Lily', 'Earrings', 'Panda', 'Monkey', 'Orange', 'Snake', 'Cabbage', 'French fries', 'Coffee table', 
               'Muffin', 'Cello', 'Piano', 'Swim cap', 'Fox', 'Plate', 'Tart', 'Kitchen knife', 'Stethoscope', 'Submarine sandwich', 'Banana', 
               'Banjo', 'Sushi', 'Motorcycle', 'Horse', 'Rugby ball', 'French horn', 'Picnic basket', 'Segway', 'Shark', 'Skunk', 'Palm tree', 
               'Whiteboard', 'Treadmill', 'Brown bear', 'Teapot', 'Apple', 'Pomegranate', 'Lynx', 'Van', 'Handgun', 'Dolphin', 'Paddle', 
               'Volleyball (Ball)', 'Hiking equipment', 'Tent', 'Tripod', 'Bed', 'Washing machine', 'Ambulance', 'Tortoise', 'Saucer', 
               'Rays and skates', 'Lizard', 'Roller skates', 'Stationary bicycle', 'Oboe', 'Artichoke', 'Barrel', 'Loveseat', 'Honeycomb', 
               'Doughnut', 'Jug', 'Maple', 'Sea lion', 'Knife', 'Goggles', 'Surfboard', 'Kitchen & dining room table', 'Carrot', 'Dog bed', 
               'Golf cart', 'Bicycle wheel', 'Airplane', 'Cookie', 'Drinking straw', 'Garden Asparagus', 'Mushroom', 'Oven', 'Sandal', 
               'Sea turtle', 'Bell pepper', 'Bus', 'Rose', 'Whisk', 'Platter', 'Tennis racket', 'Fork', 'Pumpkin', 'Taxi', 'Tiara', 
               'Woman', 'Christmas tree', 'Coffee', 'Organ (Musical Instrument)', 'Tree house', 'Jet ski', 'Milk', 'Racket', 
               'Helicopter', 'Bowl', 'Mobile phone', 'Table tennis racket', 'Cat furniture', 'Polar bear', 'Car', 'Waffle', 'Taco', 'Cake']

    REL_CLASSES = ["at", "catch", "contain", "cut", "dance", "drink", "eat", "handshake", 
                   "hang", "highfive", "hits", "holding_hands", "holds", "hug", "inside_of", 
                   "interacts_with", "is", "kick", "kiss", "on", "plays", "read", "ride", 
                   "skateboard", "ski", "snowboard", "surf", "talk_on_phone", "throw", "under", "wears"]



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
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))
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
        print('No relation with confidence > 0.3 found')
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

    topk = 10
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    #print("Length of indices:", len(indices))
    keep_queries = keep_queries[indices] # descending order by probas * probas_sub * probas_obj value

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
        plt.show()
        plt.savefig(f'{args.results_dir}/result_{os.path.basename(img_path)}')
        
        # save the dataframe to a csv file
        df.to_csv(args.log_path, index=False)


def update_dataframe(df:pd.DataFrame, **kwargs):
    # columns of dataframe :['subject', 'relation', 'object', 'subject_bbox', 'object_bbox', 'filepath']
    
    # save the subject, relation and object to the dataframe
    df = df.append(kwargs, ignore_index=True)

    return df
    
    
    
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    
    ckpt = torch.load(args.resume) # time consuming!!!
    
    IMAGE_DIR = 'summary_dataset/video/2023-03-26/002DB303B2DA_video/'
    
    
    
    # interate images from IMAGE_DIR and change arg.img_path with image file path
    tqdm_logger = tqdm(os.listdir(IMAGE_DIR))
    for img_path in tqdm_logger:
        tqdm_logger.set_description(f'Processing {img_path}')
        args.img_path = os.path.join(IMAGE_DIR, img_path)
        main(args)
    print('Done.\n')