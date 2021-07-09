'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import numpy as np
import argparse
import tensorflow as tf
import tensorflow.keras.backend as K
import keract
from PIL import Image
from keras import models
from Visualize_Model import ModelVisualizationClass

from i3d_inception import Inception_Inflated3d

NUM_FRAMES = 79
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400

SAMPLE_DATA_PATH = {
    'rgb' : 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow' : 'data/v_CricketShot_g04_c01_flow.npy'
}

LABEL_MAP_PATH = 'data/label_map.txt'

def main(args):
    # load the kinetics classes
    kinetics_classes = [x.strip() for x in open(LABEL_MAP_PATH, 'r')]

    rgb_model = None
    flow_model = None

    if args.eval_type in ['rgb', 'joint']:
        if args.no_imagenet_pretrained:
            # build model for RGB data
            # and load pretrained weights (trained on kinetics dataset only) 
            rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_kinetics_only',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for RGB data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)

        # load RGB sample (just one example)
        rgb_sample = np.load(SAMPLE_DATA_PATH['rgb'])
        
        # make prediction
        rgb_logits = rgb_model.predict(rgb_sample)

        visualizer = ModelVisualizationClass(model=rgb_model, save_images=True,
                                             out_path=r'./activations/rgb')
        visualizer.predict_on_tensor(rgb_sample)
        visualizer.plot_activation('Conv3d_1a_7x7_conv')
        visualizer.plot_activation('Conv3d_3b_0a_1x1_conv')
        visualizer.plot_activation('Conv3d_4e_1b_3x3')
        visualizer.plot_activation('Conv3d_5c_0a_1x1_conv')

    if args.eval_type in ['flow', 'joint']:
        if args.no_imagenet_pretrained:
            # build model for optical flow data
            # and load pretrained weights (trained on kinetics dataset only)
            flow_model = Inception_Inflated3d(
                include_top=True,
                weights='flow_kinetics_only',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for optical flow data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            flow_model = Inception_Inflated3d(
                include_top=True,
                weights='flow_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)

        # load flow sample (just one example)
        flow_sample = np.load(SAMPLE_DATA_PATH['flow'])
        
        # make prediction
        flow_logits = flow_model.predict(flow_sample)

        #activations_flow = keract.get_activations(flow_model, flow_sample, layer_names=None, nodes_to_evaluate=None, output_format='simple',
        #                       nested=False,
        #                       auto_compile=True)

        visualizer = ModelVisualizationClass(model=flow_model, save_images=True,
                                             out_path=r'./activations/flow')
        visualizer.predict_on_tensor(flow_sample)
        visualizer.plot_activation('Conv3d_1a_7x7_conv')
        visualizer.plot_activation('Conv3d_3b_0a_1x1_conv')
        visualizer.plot_activation('Conv3d_4e_1b_3x3')
        visualizer.plot_activation('Conv3d_5c_0a_1x1_conv')

    # produce final model logits
    if args.eval_type == 'rgb':
        sample_logits = rgb_logits
    elif args.eval_type == 'flow':
        sample_logits = flow_logits
    else:  # joint
        sample_logits = rgb_logits + flow_logits


    # produce softmax output from model logit for class probabilities
    sample_logits = sample_logits[0] # we are dealing with just one example
    sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))

    sorted_indices = np.argsort(sample_predictions)[::-1]

    print('\nNorm of logits: %f' % np.linalg.norm(sample_logits))
    print('\nTop classes and probabilities')
    for index in sorted_indices[:20]:
        print(sample_predictions[index], sample_logits[index], kinetics_classes[index])

    return


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-type', 
        help='specify model type. 1 stream (rgb or flow) or 2 stream (joint = rgb and flow).', 
        type=str, choices=['rgb', 'flow', 'joint'], default='joint')

    parser.add_argument('--no-imagenet-pretrained',
        help='If set, load model weights trained only on kinetics dataset. Otherwise, load model weights trained on imagenet and kinetics dataset.',
        action='store_true')


    args = parser.parse_args()
    main(args)
