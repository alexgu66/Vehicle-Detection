## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell 1 of the IPython notebook vehicle_detection v2.ipynb

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![data_sample](.\output_images\data_sample.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![Hog sample](.\output_images\Hog sample.png)

![Hog sample 2](.\output_images\Hog sample 2.png)



#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. For orient, I tried some other different value, following is a sample of orient = 18, which shows similar result as orient = 9, so I keep orient as 9.

![Hog_orient_18](.\output_images\Hog_orient_18.png)

For pix_per_cell, following is a sample of 16, which seems too large since the feature of car is not detected very well, so I keep it as 8.

![Hog_cell_16](.\output_images\Hog_cell_16.png)

For cell_per_block, following is a sample of value 2, which is similar with value 4. My final choice is 2, but 4 may also work well, both have 0.98/0.99 accuracy in trained model.

![Hog_block_2](.\output_images\Hog_block_2.png)

Especially for channels. using ALL 3 channels get better accuracy result than only use one of them, so for final model, all 3 channels are used.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using features extracted from both car and not car images. The data was normalized and shuffled, then was split to training and testing data set. The HOG, bin spatial and color HIST are all used. The accuracy I got is 99.04%. The training code is in cell 3. The library functions are in cell 2.

The model and other parameters were written into a pickle file for further usage, which could save training time in the future.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started with one scale 1.5 from 350 to 700, and there're many vehicles are not detected, especially for those faraway cars.  It's normal since the vehicle is actually very large close to the camera, and the smaller scale fits those faraway vehicles better. Following chart shows a pretty small scale is needed for a faraway car. 

![scale_1](.\output_images\scale_1.png)

 After tuning with test images and additional images from video. Finally 3 scales 0.9, 1.5, and 2 are used.

The implementation of sliding window is in code cell 5 (line 131 to 209), `find_cars_multiple_scales()` , which is from the find_cars() function in the lesson, I changed the calling of cv2.rectangle() to draw each slide window with blue, and the caller will draw the boundary with green. The original find_cars() defines 64 as the sampling rate, with 8 cells and 8 pix per cell, and cells_per_step = 2, which means we step 2 cells instead of overlap, and it works well for the project. Following image shows the inner slide windows and outer boundary, note the 2nd image is a negative test and blank is correct.

![scale_window](.\output_images\scale_window.png)

In the 4th image in above diagram, there's a big false positive at the left. I use the heat map and `heat_threshold`1.5 to filter it out. The code is in cell 5 `find_vehicles_boundary_multiple_scales()` line 211 to 236. 

I tried several thresh values from 1 to 2.5, the 1.5 works best. Following images show how the heap map and thresh works, the left false positive is gone.

![scale_heat](.\output_images\apply_thresh.png)



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![multiple_scale_find](.\output_images\multiple_scale_find.png)
---

In order to improve the performance, I tuned the scan scope of each scale to reduce the iteration time. Since the small scale fits top region well, and large scale fits bottom better, restrict their scan scope won't impact the accuracy, and even more, it's also helpful to reduce false positives. The final y scope for each scale is following. 

| Scale value | Y start | Y end |
| ----------- | ------- | ----- |
| 0.9         | 400     | 500   |
| 1.5         | 375     | 600   |
| 2           | 500     | 700   |



### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
It's project_video_vd.mp4.

Following is a screenshot of video.

![project_video_vd_t_39.60](.\output_images\project_video_vd_t_39.60.png)

The .\extra\project_video_vd_all.mp4 is added with lane line detection, just for fun.

Following is a screenshot of video.

![project_video_all_t_17.16](.\output_images\project_video_all_t_17.16.png)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heat map.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The code is in cell 5 `find_vehicles_boundary_multiple_scales()`.

I implemented a *dilution list*  algorithm to track the vehicle. Each vehicle has `live_frames` to record how many frames which it's detected continuously. If the vehicle is still detected after `detection_threashold` frames, it'll be really displayed in the frame. Before that, it virtually resides in a *dilution list*. The `centeroid_margin` is used to decide whether a new boundary can match to a detected vehicle from previous frames, if the distance of 2 centroids is less than `centeroid_margin`, means they're in the near location, I'll treat them as same vehicle. The dilution here means the false positive will be diluted by a series of frames(specified  `detection_threashold`). The code is in cell 8 `process_image_multiple_scales()`.

This is very effective to remove false positives, for example, imaging a Ford billboard along the road may be a false positive, and may be hard to be distinguished by the liner SVC. But it's fixed, rather than moving vehicle, so it'll be gone out of sight after several frames and won't be treated as a vehicle before its `live_frames` pass the threshold.

There're 2 methods to estimate the overlapped blob's boundary. Use a min(x, y) and max(x, y) to define the boundary, or use a weighted average of last several frames. Either works. The max method generates a little larger box and more stable, the average produces smaller box but jitters a little more. The project_video_vd.mp4 is with max method, and .\extra\project_video_vd.smooth.average.mp4 is from average method. The code is also in cell 8 `process_image_multiple_scales()`, line 72 - 85.

### 

## Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One issue I met is that the box is not stable, it changes size very quickly.  I added a `wiggling_threashold` to specify a force update boundary position for several frame. The code is in cell 8 `process_image_multiple_scales()`, line 91 - 101. The result is much better than the vanilla version, which just displays the detected blob boundary directly and jitters a lot.

The writeup template provides an idea to combine the heat of recent several frames, this shall be also very effective to remove the false positive if it only shows in some of frames. I think its essential point is as same as mine, using other frames to dilute the false positives so that it can be filtered out by threshold.

However, if there's a series of false positives at same position in frames (imaging a queue of car billboards), both above *heat merge* method and *dilution list* may fail, since the false positive appear in each frame, it'll have same heat with true positives, or pass `detection_threashold` easily. Perhaps we can try non liner SVC to get a better accuracy.

For the performance, it depends on the sliding window region heavily. So an enhancement can be search the nearby region of detected vehicles firstly, given the lane region percentage in the frame, my guess is it can improve the performance 200-300%.

There're also other misc tricks worthy trying. For example, if knowing the angle of camera lens, we can also restrict the x scope of searching, means search larger scope at bottom, and less at top. This shall be helpful to improve performance too.

