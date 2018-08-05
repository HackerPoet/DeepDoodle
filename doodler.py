import pygame
import random
import numpy as np
import cv2
from dutil import *

#User constants
device = "gpu"
model_fname = 'Model.h5'
background_color = (210, 210, 210)
input_w = 144
input_h = 192
image_scale = 3
image_padding = 10
mouse_interps = 10

#Derived constants
drawing_w = input_w * image_scale
drawing_h = input_h * image_scale
window_width = drawing_w*2 + image_padding*3
window_height = drawing_h + image_padding*2
doodle_x = image_padding
doodle_y = image_padding
generated_x = doodle_x + drawing_w + image_padding
generated_y = image_padding

def clear_drawing():
	global cur_drawing
	cur_drawing = np.zeros((1, input_h, input_w), dtype=np.uint8)

#Global variables
prev_mouse_pos = None
mouse_pressed = False
needs_update = True
cur_color_ix = 1
cur_drawing = None
clear_drawing()
cur_gen = np.zeros((3, input_h, input_w), dtype=np.uint8)
rgb_array = np.zeros((input_h, input_w, 3), dtype=np.uint8)
image_result = np.zeros((input_h, input_w, 3), dtype=np.uint8)

#Keras
print "Loading Keras..."
import os
os.environ['THEANORC'] = "./" + device + ".theanorc"
os.environ['KERAS_BACKEND'] = "theano"
import theano
print "Theano Version: " + theano.__version__
from keras.models import Sequential, load_model
from keras import backend as K
K.set_image_data_format('channels_first')

#Load the model
print "Loading Model..."
model = load_model(model_fname)

#Open a window
pygame.init()
screen = pygame.display.set_mode((window_width, window_height))
doodle_surface_mini = pygame.Surface((input_w, input_h))
doodle_surface = screen.subsurface((doodle_x, doodle_y, drawing_w, drawing_h))
gen_surface_mini = pygame.Surface((input_w, input_h))
gen_surface = screen.subsurface((generated_x, generated_y, drawing_w, drawing_h))
pygame.display.set_caption('Deep Doodle - By <CodeParade>')

def update_mouse(mouse_pos):
	global cur_color_ix
	global needs_update
	x = (mouse_pos[0] - generated_x) / image_scale
	y = (mouse_pos[1] - generated_y) / image_scale
	if not (x >= 0 and y >= 0 and x < input_w and y < input_h):
		x = (mouse_pos[0] - doodle_x) / image_scale
		y = (mouse_pos[1] - doodle_y) / image_scale
	if x >= 0 and y >= 0 and x < input_w and y < input_h:
		needs_update = True
		cur_drawing[0, y, x] = 255

def update_mouse_line(mouse_pos):
	global prev_mouse_pos
	if prev_mouse_pos is None:
		prev_mouse_pos = mouse_pos
	if cur_color_ix == 1:
		for i in xrange(mouse_interps):
			a = float(i) / mouse_interps
			ix = int((1.0 - a)*mouse_pos[0] + a*prev_mouse_pos[0])
			iy = int((1.0 - a)*mouse_pos[1] + a*prev_mouse_pos[1])
			update_mouse((ix, iy))
	else:
		update_mouse(mouse_pos)
	prev_mouse_pos = mouse_pos
			
def sparse_to_rgb(sparse_arr):
	t = np.repeat(sparse_arr, 3, axis=0)
	return np.transpose(t, (2, 1, 0))

def draw_doodle():
	pygame.surfarray.blit_array(doodle_surface_mini, rgb_array)
	pygame.transform.scale(doodle_surface_mini, (drawing_w, drawing_h), doodle_surface)
	pygame.draw.rect(screen, (0,0,0), (doodle_x, doodle_y, drawing_w, drawing_h), 1)

def draw_generated():
	pygame.surfarray.blit_array(gen_surface_mini, np.transpose(cur_gen, (2, 1, 0)))
	pygame.transform.scale(gen_surface_mini, (drawing_w, drawing_h), gen_surface)
	pygame.draw.rect(screen, (0,0,0), (generated_x, generated_y, drawing_w, drawing_h), 1)
	
#Main loop
running = True
while running:
	#Process events
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
			break
		elif event.type == pygame.MOUSEBUTTONDOWN:
			if pygame.mouse.get_pressed()[0]:
				prev_mouse_pos = pygame.mouse.get_pos()
				update_mouse(prev_mouse_pos)
				mouse_pressed = True
			elif pygame.mouse.get_pressed()[2]:
				clear_drawing()
				needs_update = True
		elif event.type == pygame.MOUSEBUTTONUP:
			mouse_pressed = False
			prev_mouse_pos = None
		elif event.type == pygame.MOUSEMOTION and mouse_pressed:
			update_mouse_line(pygame.mouse.get_pos())

	#Check if we need an update
	if needs_update:
		fdrawing = np.expand_dims(cur_drawing.astype(np.float32) / 255.0, axis=0)
		pred = model.predict(add_pos(fdrawing), batch_size=1)[0]
		cur_gen = (pred * 255.0).astype(np.uint8)
		rgb_array = sparse_to_rgb(cur_drawing)
		needs_update = False
	
	#Draw to the screen
	screen.fill(background_color)
	draw_doodle()
	draw_generated()
	
	#Flip the screen buffer
	pygame.display.flip()
	pygame.time.wait(10)
