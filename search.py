from os import listdir, path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2
import math

## CONSTANTS - CHANGE THEM FOR TESTING
# DATASET_DIRNAME = '/content/drive/My Drive/UNIBH - Ciência da Computação/4 ano/2 Semestre/Computação Gráfica/Dataset/Video'
DATASET_DIRNAME = '/content/drive/My Drive/Dataset/'

def histogram_to_vector(histogram):
  vector = np.concatenate(histogram)
  return vector.reshape(1,-1)

def color_histogram(image):
  histogram = cv2.calcHist([image], [0, 1], None, [180, 256], [0,180,0,256])
  return cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

def resize_to_same_dimensions(a, b):
  height = a.shape[0]
  width = a.shape[1]

  if (a.shape[0] > b.shape[0]):
    height = b.shape[0]
  
  if (a.shape[1] > b.shape[1]):
    width = b.shape[1]

  dimensions = (width, height)

  a = cv2.resize(a, dimensions)
  b = cv2.resize(b, dimensions)

  return (a, b)

def sort_by_similarity(items):
  get_similarity = lambda item: item['similarity']
  items.sort(reverse=True, key=get_similarity)
  return items

def video_x_video_similarity(video_a, video_b):
  similarities = []

  # Sort by frames count - shortest video comes first,
  # longest video comes later
  videos = [
    { 'video': video_a, 'frames': int(video_a.get(cv2.CAP_PROP_FRAME_COUNT)) },
    { 'video': video_b, 'frames': int(video_b.get(cv2.CAP_PROP_FRAME_COUNT)) },
  ]
  videos.sort(key=lambda x: x['frames'])

  frame_step = videos[1]['frames'] / videos[0]['frames']

  if (videos[0]['video'].isOpened() and videos[1]['video'].isOpened()):
    for frame_index in range(0, videos[0]['frames']):
      success_shortest, frame_shortest = videos[0]['video'].read()

      pos_next_frame_longest_video = math.floor(frame_step * frame_index)
      videos[1]['video'].set(cv2.CAP_PROP_POS_FRAMES, pos_next_frame_longest_video)
      success_longest, frame_longest = videos[1]['video'].read()

      similarity = image_x_image_similarity(frame_shortest, frame_longest)
      similarities.append(similarity)

  videos[0]['video'].release()
  videos[1]['video'].release()
  return np.mean(similarities) if len(similarities) > 0 else 0

def image_x_video_similarity(image, video):
  similarities_each_frame = []

  while (video.isOpened()):
    ret, frame = video.read()
    if ret == False:
      break

    similarity = image_x_image_similarity(image, frame)
    similarities_each_frame.append(similarity)
  
  video.release()
  return np.mean(similarities_each_frame) if len(similarities_each_frame) > 0 else 0

def image_x_image_similarity(file_a, file_b):
  file_a, file_b = resize_to_same_dimensions(file_a, file_b)
  vector_a = histogram_to_vector(color_histogram(file_a))
  vector_b = histogram_to_vector(color_histogram(file_b))
  return cosine_similarity(vector_a, vector_b)[0][0]

def is_video(filename):
  name, ext = path.splitext(filename)
  return ext == '.mp4'

def is_image(filename):
  name, ext = path.splitext(filename)
  return (ext == '.jpg' or ext == '.jpeg' or ext == '.png')

def read_file(filename):
  if (is_image(filename)):
    return cv2.imread(path.join(DATASET_DIRNAME, filename))

  elif (is_video(filename)):
    return cv2.VideoCapture(path.join(DATASET_DIRNAME, filename))

  else:
    raise Exception("Couldn't read file: invalid format. It must be either a video or an image.")

def read_dataset_files(dirname):
  return listdir(dirname)

# main function
def search(query_filename, threshold, max_items):
  similars = []
  dataset_filenames = read_dataset_files(DATASET_DIRNAME)
  
  for dataset_filename in dataset_filenames:
    similarity = 0
    input = read_file(query_filename)
    file = read_file(dataset_filename)

    if (is_image(query_filename) and is_image(dataset_filename)):
      similarity = image_x_image_similarity(input, file)

    elif (is_image(query_filename) and is_video(dataset_filename)):
      similarity = image_x_video_similarity(input, file)

    elif (is_video(query_filename) and is_video(dataset_filename)):
      similarity = video_x_video_similarity(input, file)

    elif (is_video(query_filename) and is_image(dataset_filename)):
      similarity = image_x_video_similarity(file, input)

    if (similarity >= threshold):
      similars.append({ 'filename': dataset_filename, 'similarity': similarity })
    
    if (len(similars) == max_items):
      break

  return sort_by_similarity(similars)

search('video.mp4', 0.5, 5)
