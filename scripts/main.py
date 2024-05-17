import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from Human_Detection import *
from FeatureExtraction import *
from utils.datasets import letterbox
import math
from controller import PID
import pickle
import random
import sys
class Evaluation:
    def __init__(self, n_exp) -> None:
        self.n_exp = n_exp
        self.n_frames = 0
        self.time = []
        self.n_success = 0
        self.n_fail = 0
        self.distance_vec = []
        self.angle_error_vec = []
        self.total_distance_travelled = 0
        self.velocity = []
        self.angular_velocity = []

        
        self.rand = random.randint(0, 10000)
    def save_params(self):
        '''
        Save the evaluation parameters to a pickle file
        '''
        self.params = {"n_frames":self.n_frames,
                        "time":self.time,
                        "n_success":self.n_success,
                        "n_fail":self.n_fail,
                        "distance_vec":self.distance_vec,
                        "angle_error_vec":self.angle_error_vec,
                        "total_distance_travelled":self.total_distance_travelled,
                        "velocity":self.velocity,
                        "angular_velocity":self.angular_velocity}

        path = f'/home/amir/Documents/Robotics Project/Experiments/Exp{self.n_exp}/'

        with open(path + f'eval_{self.rand}.pkl', 'wb') as f:
            pickle.dump(self.params, f)

save_params = True

def image_to_histogram(image, bins=32):
    # Convert image to histogram with specified bins for each color channel
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = np.histogramdd(image.reshape(-1, 3), bins=bins, range=[(0, 256), (0, 256), (0, 256)])[0]
    # Normalize the histogram
    histogram = histogram.ravel()
    histogram /= histogram.sum()
    return histogram

class HumanTracker:
    def __init__(self):
        self.detector = Human_Detection()
        self.fe = SuperPointFrontend(nms_dist=4, conf_thresh=0.015, nn_thresh=0.7)
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.rate = rospy.Rate(10)
        self.angular_vel = 0
        self.linear_PID = PID(.2, 0.0001, 0)
        # self.angular_PID = PID(.015, 0, 0)
        # self.linear_PID = PID(.5, 0.0001, 0)
        self.angular_PID = PID(1, 0.0001, 0)
        self.x_thresh = 5
        self.y_thresh = 2
        self.desired_distance = 60
        self.FOV_x = 87 #deg
        self.FOV_y = 58 #deg
        self.lost_target_count = 0
        self.rgb_histo = []
        self.eval = Evaluation(5)

    def depth_callback(self, image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg, "16UC1")
        self.depth_image = image
        self.depth_time = image_msg.header.stamp.secs

    def callback(self, image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8") 

        if not hasattr(self, 'frame_count'):
            self.initialize(image)

        if self.frame_count <= 100:
            self.configure(image)
        else:

            self.track(image)

        self.frame_count += 1

    def initialize(self, image):
        self.frame_count = 0
        self.detector.conf_box = [int(image.shape[1]/2)-120, 0, int(image.shape[1]/2)+120, image.shape[0]]
    
    def configure(self, image):
        frame = letterbox(image, 640, stride=64, auto=True)[0]
        self.detector.detect(frame)
        best_box, best_mask = self.detector.configuration()
        if best_box is None or best_mask is None:
            return
        masked_img = self.detector.mask_bg(frame, best_box, best_mask)
        gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        pts, desc, _ = self.fe.run(gray_masked) 
        indices = np.where(pts[2, :] >= 0.2)[0]
        desc = desc[:, indices]
        self.detector.features = np.concatenate((self.detector.features, desc), axis=1)
        for i in range(pts.shape[1]):
            cv2.circle(masked_img, (int(pts[0][i]), int(pts[1][i])), radius=2, color=(0, 0, 255), thickness=-1)

    def track(self, image):
        global save_params
        self.eval.n_frames += 1
        

        frame = letterbox(image, 640, stride=64, auto=True)[0]
        self.detector.detect(frame)
        best_matches = 0
        best_box, best_mask = None, None
        best_box_histo, best_mask_histo = None, None
        best_histo = 10

        cx, cy = None, None
        for one_mask, bbox, cls, conf in self.detector.zipfile:
            if conf < self.detector.pred_conf or cls != self.detector.person_class_idx:
                continue
            masked_img = self.detector.mask_bg(frame, bbox, one_mask)
            gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            pts, desc, _ = self.fe.run(gray_masked)
            if desc is None or not desc.any():
                continue
            indices = np.where(pts[2, :] >= 0.2)[0]
            desc = desc[:, indices]
            matches = self.fe.nn_match_two_way(self.detector.features, desc, 0.7)
            self.detector.draw(frame, bbox, one_mask, color=(255, 0, 0))
            histo = image_to_histogram(masked_img)
            histo /= np.linalg.norm(histo)
            minn = 10.0
            for i in range(len(self.rgb_histo)):
                minn = min(minn, np.linalg.norm(histo - self.rgb_histo[i]))
            if (minn < best_histo):
                best_box_histo, best_mask_histo = bbox, one_mask
                best_histo = minn
            if matches.shape[1] > best_matches:
                best_matches = matches.shape[1]
                best_box, best_mask = bbox, one_mask
                best_histo = minn

        if best_matches > 5:
            self.detector.draw(frame, best_box, best_mask, color=(0, 255, 0))
            cx = (best_box[0] + best_box[2]) / 2
            cy = (best_box[1] + best_box[3]) / 2
        if best_box_histo is not None:
            self.detector.draw(frame, best_box_histo, best_mask_histo, color=(0, 255, 0))
            cx = (best_box_histo[0] + best_box_histo[2]) / 2
            cy = (best_box_histo[1] + best_box_histo[3]) / 2

        if cx != None and cy != None:
            self.eval.time.append(rospy.get_time())

            self.eval.n_success += 1

            self.lost_target_count = 0
            depth = self.depth_image[int(cy)][int(cx)] # in mm
            distance_x = (depth / 10) * math.cos(math.radians(50))
            distance_error_x = distance_x - self.desired_distance
            self.eval.distance_vec.append(distance_x)
            distance_error_y_pxl = cx - 320

            distance_error_y = distance_error_y_pxl * (math.radians(self.FOV_y) / 640)
            self.eval.angle_error_vec.append(distance_error_y)

            # print(distance_x, distance_error_y)
            linear_speed = self.linear_PID.update(distance_error_x / 21)
            self.eval.velocity.append(linear_speed*27)

            angular_speed = self.angular_PID.update(distance_error_y / (-math.pi / 4))
            self.eval.angular_velocity.append(angular_speed)
            twist_msg = Twist()
            twist_msg.angular.z = angular_speed
            twist_msg.linear.x = linear_speed

            toc = rospy.get_time()
            if hasattr(self, 'tic'):
                self.eval.total_distance_travelled += linear_speed * 27 * (toc - self.tic) 
                print(toc - self.tic)
                self.tic = toc

            
            if abs(distance_error_x) < self.x_thresh and abs(distance_error_y) < self.y_thresh:
                # save_params = False
                print("Reached")
                # sys.exit()
                return

            self.pub.publish(twist_msg)
            if not hasattr(self, 'tic'):
                self.tic = rospy.get_time()
            self.rate.sleep()

        self.lost_target_count += 1
        self.eval.n_fail += 1




        if self.lost_target_count > 30:
            twist_msg = Twist()
            twist_msg.angular.z = 0.35 # * pi/4 per second
            twist_msg.linear.x = 0
            self.pub.publish(twist_msg)
            self.rate.sleep()

        if save_params:
            self.eval.save_params()

        cv2.imshow("Tracking", frame)
        cv2.waitKey(1)
def main():
    rospy.init_node('human_tracker', anonymous=True)
    tracker = HumanTracker()
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, tracker.depth_callback)
    rospy.Subscriber('/camera/color/image_raw', Image, tracker.callback)
    rospy.spin()

if __name__ == '__main__':
    main()