# import the required libraries
import numpy as np
import time
import random
import cPickle
import codecs
import collections
import os
import math
import json
import csv
import tensorflow as tf
from six.moves import xrange

# libraries required for visualisation:
from IPython.display import SVG, display
import svgwrite # conda install -c omnia svgwrite=1.1.6
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
tf.logging.info("TensorFlow Version: %s", tf.__version__)

# import our command line tools
from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *

# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.2, svg_filename = '/tmp/sketch_rnn/svg/sample.svg'):
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in xrange(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  display(SVG(dwg.tostring()))

# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
  def get_start_and_end(x):
    x = np.array(x)
    x = x[:, 0:2]
    x_start = x[0]
    x_end = x.sum(axis=0)
    x = x.cumsum(axis=0)
    x_max = x.max(axis=0)
    x_min = x.min(axis=0)
    center_loc = (x_max+x_min)*0.5
    return x_start-center_loc, x_end
  x_pos = 0.0
  y_pos = 0.0
  result = [[x_pos, y_pos, 1]]
  for sample in s_list:
    s = sample[0]
    grid_loc = sample[1]
    grid_y = grid_loc[0]*grid_space+grid_space*0.5
    grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
    start_loc, delta_pos = get_start_and_end(s)

    loc_x = start_loc[0]
    loc_y = start_loc[1]
    new_x_pos = grid_x+loc_x
    new_y_pos = grid_y+loc_y
    result.append([new_x_pos-x_pos, new_y_pos-y_pos, 0])

    result += s.tolist()
    result[-1][2] = 1
    x_pos = new_x_pos+delta_pos[0]
    y_pos = new_y_pos+delta_pos[1]
  return np.array(result)

def encode(sess, sample_model, eval_model, model,
           input_strokes):
  strokes = to_big_strokes(input_strokes).tolist()
  strokes.insert(0, [0, 0, 1, 0, 0])
  
  del strokes[126:]
  #if strokes[6][4] != 1.0:
  #    print(strokes)
  
  seq_len = [len(input_strokes)]
  #draw_strokes(to_normal_strokes(np.array(strokes)))
  return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]

def decode(sess, sample_model, eval_model, model,
           z_input=None, draw_mode=True, temperature=0.1, factor=0.2):
  z = None
  if z_input is not None:
    z = [z_input]
  sample_strokes, m = sample(sess, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, z=z)
  strokes = to_normal_strokes(sample_strokes)
  if draw_mode:
    draw_strokes(strokes, factor)
  return strokes

def main():
    
    data_dir = './image_full_npz'
    model_dir = './rnn_model_2'

    '''
    [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env(data_dir, model_dir)
    
    # construct the sketch-rnn model here:
    reset_graph()
    model = Model(hps_model)
    eval_model = Model(eval_hps_model, reuse=True)
    sample_model = Model(sample_hps_model, reuse=True)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    # loads the weights from checkpoint into our model
    load_checkpoint(sess, model_dir)
    #load stroke
    while(1):
        stroke = test_set.random_sample()
        print(stroke)
        draw_strokes(stroke)
        z = encode(sess, sample_model, eval_model, model, stroke)

    data_dir = './image_full_npz'
    model_dir = './rnn_model_2'
    [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env(data_dir, model_dir)
    '''
    test = [[[  14,  -12,    0],
            [  58,  -24,    0],
            [  58,  -17,    0],
            [  53,   -5,    0],
            [  49,    4,    1],
            [-229,   55,    0],
            [   0,   10,    0],
            [  11,   21,    0],
            [  11,    9,    0],
            [  20,    8,    0],
            [  26,    5,    0],
            [  77,    0,    0],
            [  19,   -4,    0],
            [  16,   -8,    0],
            [  21,  -34,    0],
            [  24,  -63,    1],
            [-108,   14,    0],
            [ -16,    4,    0],
            [ -14,   15,    0],
            [  -6,   26,    0],
            [   2,   12,    0],
            [   7,   10,    0],
            [  31,   10,    0],
            [  16,   -1,    0],
            [  13,   -6,    0],
            [   9,  -10,    0],
            [   4,  -12,    0],
            [  -9,  -24,    0],
            [ -25,  -14,    1],
            [ -87,   -4,    0],
            [ -41,  -49,    1],
            [  85,   31,    0],
            [  -9,  -47,    1],
            [  74,   37,    0],
            [  11,  -30,    0],
            [  15,  -25,    1]]]
    [hps_model, eval_hps_model, sample_hps_model] = load_model(model_dir)
    # construct the sketch-rnn model here:
    reset_graph()
    model = Model(hps_model)
    eval_model = Model(eval_hps_model, reuse=True)
    sample_model = Model(sample_hps_model, reuse=True)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    # loads the weights from checkpoint into our model
    load_checkpoint(sess, model_dir)
    #load stroke
    test_set = DataLoader(
      test,
      batch_size=1,
      max_seq_length=125,
      random_scale_factor=0.0,
      augment_stroke_prob=0.0)
    normalizing_scale_factor = test_set.calculate_normalizing_scale_factor()
    test_set.normalize(normalizing_scale_factor)
    stroke = test_set.strokes[0]
    print(stroke)
    draw_strokes(stroke)
    z = encode(sess, sample_model, eval_model, model, stroke)
    print(z)
    z_means = np.load('z_means.npy')
    diff = np.zeros(5)
    for j in range(5):
        diff[j] = np.linalg.norm(z - z_means[j])
    name = ['eye', 'finger', 'foot', 'hand', 'leg']
    print(name[np.argmin(diff)])
    
    
    '''
    
    data_dir = './image_full_npz'
    model_dir = './rnn_model_2'
    [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env(data_dir, model_dir)
    
    # construct the sketch-rnn model here:
    reset_graph()
    model = Model(hps_model)
    eval_model = Model(eval_hps_model, reuse=True)
    sample_model = Model(sample_hps_model, reuse=True)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    # loads the weights from checkpoint into our model
    load_checkpoint(sess, model_dir)
    
    csvfile = open('leg.csv', 'wb')
    csvwriter = csv.writer(csvfile, delimiter=',')
    for i in range(len(train_set.strokes)):
        if i%500 == 0:
            print(i)
        stroke = train_set.strokes[i]
        z = encode(sess, sample_model, eval_model, model, stroke)
        csvwriter.writerow(z)
    '''
    
if __name__ == "__main__":
    main()










