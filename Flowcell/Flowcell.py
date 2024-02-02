import glob
import logging
import os
import time

import cv2
import numpy as np
import pandas as pd
import skimage
from matplotlib import pyplot as plt
from scipy import ndimage as nd
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.filters import threshold_sauvola
from skimage.segmentation import clear_border, watershed

if skimage.__version__ != "0.19.3":
    print("skimage version is not 0.19.3, please change to 0.19.3") 

class Camera_flowcell():

    def __init__(self, main_index=False):
        
        self.Logger = Event_Logger()
        self.cap = None
        
        self.pic_width = 2160
        self.pic_lenght = 3840
        self.main_index = main_index
        
        # Setup Directory for frames
        try:
            if not os.path.exists("Flowcell\\videoframes"):
                os.makedirs(
                    "Flowcell\\videoframes")
            if not os.path.exists("Flowcell\\analysis_feedback"):
                os.makedirs(
                    "Flowcell\\analysis_feedback")
            if not os.path.exists("Flowcell\\picture_analysis"):
                os.makedirs(
                    "Flowcell\\picture_analysis")
        except OSError:
            self.Logger.log("Error: Creating Directory of Data")


    def camera_connect(self):
        """
        Connects to the camera via usb
        and sets the resolution, fps, exposure time

        Returns:
            cv2.VideoCapture: camera object
        """

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.pic_lengt)
        self.cap.set(4, self.pic_width)
        self.cap.set(cv2.CAP_PROP_FPS, 30.0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        
        return self.cap

    def take_video(self):
        """
        Connects to camera and takes a video,
        then saves frames as .tiff files
        """

        self.cap = self.camera_connect()

        currentframe = 0
        lastframe = 20
        
        while True:
            
            success, image = self.cap.read()
            name = "Flowcell\\picture_analysis\\frame" + \
                str(currentframe) + ".tiff"
            cv2.imwrite(name, image)
            currentframe += 1
            
            if currentframe == lastframe:
                break

        self.cap.release()

    def take_calibration_photo(self):
        """
        Connects to camera and takes a photo,
        then saves it as .tiff file
        """

        self.cap = self.camera_connect()
        ret, frame = self.cap.read()

        cv2.imwrite(
            "Flowcell\\picture_analysis\\calibration.tiff", frame)

        self.cap.release()

    def readpicture(self, path, lower_x, upper_x):
        """
        Reads the picture which should be analyzed,
        crops it and converts it to grayscale

        Args:
            path (str): path to the picture which should be analyzed
            lower_x (int): size for the cropping of the picture
            upper_x (int): size for the cropping of the picture

        Returns:
            cv2.cvtColor: cropped and grayscaled image
        """

        img = cv2.imread(path)
        img = cv2.resize(img, (self.pic_lenght, self.pic_width))
        
        # crop the image/delete boundary areas which are not needed
        cropped = img[0:2160, lower_x:upper_x]
        
        # turns colored image into grayscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        return gray

    def thresholding(self, gray_scale, window_size, additional_thresh, i):
        """
        Applies the sauvola thresholding to the image, then converts it to binary image,
        removes small objects and holes, labels the particles,
        and returns the labeled image and the binary image

        Args:
            gray_scale (cv2.cvtColor): cropped and grayscaled image
            window_size (int): size of the local thresholding area
            additional_thresh (int): additional thresholding value
            i (int): defines which thresholding should be applied (1 for calibration or 2 for normal analysis)

        Returns:
            ndarray, ndarray: labeled image and cleared image
        """

        # window_size = 155  # defines size of the local thresholding area. define not too small! - no differences in gray level can be detected anymore
        # local thresholding according to Sauvola
        thresh = threshold_sauvola(gray_scale, window_size=window_size)
        # converts gray scale image into binary image
        binary = gray_scale > thresh + additional_thresh
        if i == 1:
            binary = morphology.remove_small_objects(binary, 2000)
            binary = morphology.remove_small_holes(binary, 5000)

            # Convert binary image to uint8 for contour detection
            cleared = binary.astype(np.uint8) * 255

        if i == 2:
            binary = morphology.remove_small_objects(binary, 3)
            cleared = clear_border(binary)
        # labeling the detected particles
        s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        labeled, num_labels = nd.label(cleared, structure=s)

        return labeled, cleared

    def particle_analysis(self):
        """
        analyses particle pictures using calibration, thresholding,
        watershed segmentation, and finally particle measurements
        
        returns the PSD data, the stationary data, and the total number of particles
        """
        
        path = "Flowcell\\picture_analysis\\*.*"
        
        # `lower_x` and `upper_x` are used to crop the image and remove the boundary areas which are
        # not representative for the analysis. `lower_x` defines the starting pixel for the cropping
        # and `upper_x` defines the ending pixel.
        lower_x = 1200
        upper_x = 2640   

        path_tri = "Flowcell\\picture_analysis\\calibration.tiff"

        # convert pictures into gray scale
        gray_tri = self.readpicture(path_tri, lower_x, upper_x)
        plt.imsave(
            "Flowcell\\analysis_feedback\\gray_tri.png", gray_tri, cmap='gray')
        
        # define windowsize for the local thresholding and the additional value for fine tuning of the threshold
        window_size = 155
        additional_thresh_tri = 40  # Default:40

        # thresholding of the picture to be analyzed
        additional_thresh = 30  # Default:30

        # actual thresholding of the two pictures
        j = 1
        labeled_tri, binary_try = self.thresholding(
            gray_tri, window_size, additional_thresh_tri, j)

        contours, hierarchy = cv2.findContours(
            binary_try, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # fint the bounting rectanle for each particle
        rects = [cv2.boundingRect(ctr) for ctr in contours]
        # find the long side of each rectangle
        long_side = [max(rect[2], rect[3]) for rect in rects]
        # sortlong_side in size
        long_side.sort(reverse=True)

        # __________________Rectangle check function_____________________
        for rect in rects:
            # draw the rectangles
            cv2.rectangle(gray_tri, (rect[0], rect[1]),
                          (rect[0], rect[1]+rect[3]), (0, 255, 0), 2)

        plt.imsave("Flowcell\\analysis_feedback\\binary_tri_rect.png",
                   gray_tri, cmap='gray')

        plt.imsave("Flowcell\\analysis_feedback\\binary_tri.png",
                   labeled_tri, cmap='gray')

        mean_pixel_tri_front = float(long_side[0])
        mean_pixel_tri_back = float(long_side[1])

        # the lengths of the calibration markers on the 
        # front and back side of the flowcell in microns
        length_tri_um_back = 1015
        length_tri_um_front = 1220 

        pixel_length_back = length_tri_um_back/mean_pixel_tri_back
        pixel_length_front = length_tri_um_front/mean_pixel_tri_front

        pixels_to_um = (pixel_length_back+pixel_length_front)/2

        props_table = pd.DataFrame()
        i = 1
        for file in glob.glob(path):
            gray = self.readpicture(file, lower_x, upper_x)
            plt.imsave("Flowcell\\analysis_feedback\\gray_" +
                       str(i)+".png", gray, cmap='gray')

            # subtracting the two pictures/to exclude the flow breaker
            subt = cv2.subtract(gray, gray_tri)
            j = 2
            labeled, binary = self.thresholding(
                subt, window_size, additional_thresh, j)
            plt.imsave("Flowcell\\analysis_feedback\\binary_" +
                       str(i)+".png", binary, cmap='gray')

            # segmentation verbessern!
            D = nd.distance_transform_edt(labeled)
            localMax = peak_local_max(
                D,  indices=False, min_distance=50, labels=labeled)    # Default:50
            markers = nd.label(localMax, structure=np.ones((3, 3)))[
                0]
            labels = watershed(-D, markers, mask=labeled)

            # optional: colorize the thresholded picture to see the crystalls in different colors
            # img_col = color.label2rgb(labeled, bg_label=0)

            # overlay the thresholded picture over the original gray scale image to see if
            # most of the particles can be detected and no areas are fals detected
            image_label_overlay = label2rgb(labels, image=gray)
            plt.imsave("Flowcell\\analysis_feedback\\label_overlay_" +
                       str(i)+".png", image_label_overlay)

        # measurement/measure all detected crystalls and delete some of the noise
            props = pd.DataFrame(measure.regionprops_table(labels, gray,
                                                           properties=['label', 'area',
                                                                       'feret_diameter_max',
                                                                       'axis_minor_length', 'equivalent_diameter_area']))
            props_table = props_table.append(props)
            i += 1

        dataframe = pd.DataFrame(props_table)
        # delete crystals with area smaller than 3 pixels
        dataframe = dataframe[dataframe['area'] > 3]  # Default: 3
        dataframe = dataframe[dataframe['axis_minor_length'] > 0]

        # convert pixels to microns
        dataframe['area_sq_microns'] = dataframe['area'] * (pixels_to_um**2)
        dataframe['feret_diameter_max_microns'] = dataframe['feret_diameter_max'] * \
            (pixels_to_um)
        dataframe['axis_minor_length_microns'] = dataframe['axis_minor_length'] * \
            (pixels_to_um)
        dataframe['equivalent_diameter_area_microns'] = dataframe['equivalent_diameter_area'] * \
            (pixels_to_um)
        dataframe = dataframe[dataframe['feret_diameter_max_microns'] < 2000]
        dataframe = dataframe[dataframe['equivalent_diameter_area_microns'] > 3]

        # self.Logger.log(dataframe.head())
        mean_feret = dataframe['feret_diameter_max_microns'].mean()
        std_feret = dataframe['feret_diameter_max_microns'].std()

        # ___Analysis with feret maximum diameter_____________________
        # Stationary_Data = dataframe.sort_values(
        #     by=['feret_diameter_max_microns'])
        # self.xslx_save(df=Stationary_Data)
        # hist = dataframe.hist('feret_diameter_max_microns', bins = 30, rwidth=0.9)
        # count, division = np.histogram(
        #     dataframe['feret_diameter_max_microns'], bins=200, range=(0, 2000))  # Default 15

        # ___Analysis with circle equivalent diameter___________________________
        Stationary_Data = dataframe.sort_values(
            by=['equivalent_diameter_area'])
        count, division = np.histogram(
            dataframe['equivalent_diameter_area_microns'], bins=200, range=(0, 2000))  # Default 15


        delta_x = np.diff(division)
        division = np.delete(division, 0)
        n_total = len(dataframe.index)
        
        dQ0 = np.divide(count, n_total)
        Q0 = np.cumsum(dQ0)
        q0 = np.divide(dQ0, delta_x)
        dx = np.divide(delta_x, 2)
        x_mean = np.subtract(division, dx)
        
        x_m_3_dQ0 = np.multiply(np.power(x_mean, 3), dQ0)
        xm3_Q0 = np.sum(x_m_3_dQ0)
        dQ3 = np.divide(x_m_3_dQ0, xm3_Q0)
        Q3 = np.cumsum(dQ3)
        q3 = np.divide(dQ3, delta_x)

        PSD_Data = pd.DataFrame({'x_mean': x_mean, 'Q0': Q0, 'q0': q0,
                                'Q3': Q3, 'q3': q3, 'deltax': delta_x,
                                 'count': count, 'division': division})

        if self.main_index == True:
            self.xslx_save(df=PSD_Data)
            x_10_df = Stationary_Data.iloc[int(n_total*0.1)]
            x_50_df = Stationary_Data.iloc[int(n_total*0.5)]
            x_90_df = Stationary_Data.iloc[int(n_total*0.9)]
            self.PSD_Plot(PSD_Data, x_10_df, x_50_df, x_90_df)
            plt.show()

        return PSD_Data, Stationary_Data, n_total

    def PSD_Plot(self, PSD_Data, x_10_df, x_50_df, x_90_df):
        """
        Generates a plot of the PSD data,
        specifically the x10, x50, and x90 values of the Q0 distribution.

        Args:
            PSD_Data (pd.DataFrame): Full set of PSD data.
            x_10_df (float): x10 diameter of the particle distribution.
            x_50_df (float): x50 diameter of the particle distribution.
            x_90_df (float): x90 diameter of the particle distribution.
        """
        x = np.array(PSD_Data['x_mean'].values)
        y = np.array(PSD_Data['Q0'].values)

        plt.title("Q0 plot")
        plt.plot(x, y, color="red")
        plt.scatter(x_10_df['equivalent_diameter_area_microns'],
                    0.1, color='red', label='x_10')
        plt.scatter(x_50_df['equivalent_diameter_area_microns'],
                    0.5, color='blue', label='x_50')
        plt.scatter(x_90_df['equivalent_diameter_area_microns'],
                    0.9, color='green', label='x_90')
        plt.savefig(
            "Flowcell\\analysis_feedback\\Q0_plot.png")

    def xslx_save(self, df):
        """
        saves the particle data DataFrame to an Excel file.

        Args:
            df (pd.DataFrame): Full set of PSD data.
        """

        with pd.ExcelWriter('PSD.xlsx', mode='a', if_sheet_exists='new') as writer:
            df.to_excel(writer, index=False)
            
    def determine_steady_state(self):
        """
        Compares the x10, x25, x50, x75 and x90 values of the latest 2
        analysis loops to determine if a steady state has been reached.
        
        Returns the PSD data of the final analysis loop
        """
        
        Steadystate_df = pd.DataFrame({'x10_mean': [0], 'x_10': [0],
                                       'x25_mean': [0], 'x_25': [0],
                                       'x50_mean': [0], 'x_50': [0],
                                       'x75_mean': [0], 'x_75': [0],
                                       'x90_mean': [0], 'x_90': [0]})
        
        # tolerances of the x10, x25, x50, x75 and x90 values
        # can be adjusted to the specific requirements of the process
        size_tolerance_x10 = 0.2
        size_tolerance_x25 = 0.15
        size_tolerance_x50 = 0.1
        size_tolerance_x75 = 0.05
        size_tolerance_x90 = 0.05
        
        i = 1
        index = self.Message.start_continue.isChecked()

        while index == False:

            stat_t_start = time.time()

            while True:
                try:
                    self.Flowcell_analysis.take_video()
                    break
                except:
                    self.Logger.log("Camera Error, please check camera connection")
                    time.sleep(1)
                    continue
                
            time.sleep(1)
            
            # Q0 PSD für x10, 25, 50 ,75, 90 used
            PSD_Data, Stationary_Data, n_total = self.Flowcell_analysis.particle_analysis()

            x_10_df = Stationary_Data.iloc[int(n_total*0.1)-1]
            x_25_df = Stationary_Data.iloc[int(n_total*0.25)-1]
            x_50_df = Stationary_Data.iloc[int(n_total*0.5)-1]
            x_75_df = Stationary_Data.iloc[int(n_total*0.75)-1]
            x_90_df = Stationary_Data.iloc[int(n_total*0.9)-1]
    
            
            Steadystate_df_old = Steadystate_df.iloc[-1]
            Steadystate_df = Steadystate_df.append({'x10_mean': x_10_df['equivalent_diameter_area_microns'], 'x_10': 0.1,
                                                    'x25_mean': x_25_df['equivalent_diameter_area_microns'], 'x_25': 0.25,
                                                   'x50_mean': x_50_df['equivalent_diameter_area_microns'], 'x_50': 0.5,
                                                    'x75_mean': x_75_df['equivalent_diameter_area_microns'], 'x_75': 0.75,
                                                    'x90_mean': x_90_df['equivalent_diameter_area_microns'], 'x_90': 0.9}, ignore_index=True)

            
            # _________________________________Criteria for steady State _______________________________________    
            if (round(Steadystate_df['x10_mean'].iloc[-1]) <= (round(Steadystate_df_old['x10_mean'])+(round(Steadystate_df_old['x10_mean'])*size_tolerance_x10)) and
                    round(Steadystate_df['x10_mean'].iloc[-1]) >= (round(Steadystate_df_old['x10_mean'])-(round(Steadystate_df_old['x10_mean'])*size_tolerance_x10)) and
                    round(Steadystate_df['x25_mean'].iloc[-1]) <= (round(Steadystate_df_old['x25_mean'])+(round(Steadystate_df_old['x25_mean'])*size_tolerance_x25)) and
                    round(Steadystate_df['x25_mean'].iloc[-1]) >= (round(Steadystate_df_old['x25_mean'])-(round(Steadystate_df_old['x25_mean'])*size_tolerance_x25)) and
                    round(Steadystate_df['x50_mean'].iloc[-1]) <= (round(Steadystate_df_old['x50_mean'])+(round(Steadystate_df_old['x50_mean'])*size_tolerance_x50)) and
                    round(Steadystate_df['x50_mean'].iloc[-1]) >= (round(Steadystate_df_old['x50_mean'])-(round(Steadystate_df_old['x50_mean'])*size_tolerance_x50)) and
                    round(Steadystate_df['x75_mean'].iloc[-1]) <= (round(Steadystate_df_old['x75_mean'])+(round(Steadystate_df_old['x75_mean'])*size_tolerance_x75)) and
                    round(Steadystate_df['x75_mean'].iloc[-1]) >= (round(Steadystate_df_old['x75_mean'])-(round(Steadystate_df_old['x75_mean'])*size_tolerance_x75)) and
                    round(Steadystate_df['x90_mean'].iloc[-1]) <= (round(Steadystate_df_old['x90_mean'])+(round(Steadystate_df_old['x90_mean'])*size_tolerance_x90)) and
                    round(Steadystate_df['x90_mean'].iloc[-1]) >= (round(Steadystate_df_old['x90_mean'])-(round(Steadystate_df_old['x90_mean'])*size_tolerance_x90))):

                self.Logger.log('Steady State reached')
                return PSD_Data, Stationary_Data, n_total,i
            # _____________________________________________________________________________________________________

            stat_t_end = time.time()
            stat_t_diff = stat_t_end-stat_t_start
            
            # makes sure that the analysis loop takes at least 10 minutes
            if stat_t_diff < 600:
                time.sleep(600-stat_t_diff)
            i += 1

        return PSD_Data, Stationary_Data, n_total, i
    
    
class Event_Logger():

    def __init__(self, severity: int = 20):

        self.Logger = logging.getLogger(__name__)
        self.Logger.setLevel(severity)
        self.custom_handler = logging.StreamHandler()
        self.file_handler = logging.FileHandler(
            'Flowcell\\app.log')

        self.custom_handler.setLevel(logging.INFO)
        self.file_handler.setLevel(logging.INFO)

        self.custom_handler_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%m-%y %H:%M:%S')
        self.custom_handler.setFormatter(self.custom_handler_format)

        self.file_handler_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%m-%y %H:%M:%S')
        self.file_handler.setFormatter(self.file_handler_format)

        self.Logger.addHandler(self.custom_handler)
        self.Logger.addHandler(self.file_handler)

    def log(self, message=None, message_type=20,):
        """
        passes a message to the logger,
        which prints it to the console and saves it to the log file.

        Args:
            message (str, optional): the message to be passed to the logger. Defaults to None.
            message_type (int, optional): severity of the message. Defaults to 20.
        """
        
        # in case you want to log where exactly in the scripts a message comes from:
        # self.Logger.log(message_type, message, stack_info=True)
        self.Logger.log(message_type, message)



# test functions
if __name__ == "__main__":
    
    main_index = True
    camera = Camera_flowcell(main_index)
    # camera.take_calibration_photo(25)
    
    # total_start = time.time()
    # camera.take_video(25)
    PSD_Data, Stationary_Data, n_total = camera.particle_analysis()
    # total_end = time.time()
    
    # self.Logger.log("TOTAL:" + str(total_end-total_start))
