import cv2

#image.shape[0] = width
#image.shape[1] = height
class VideoStream:
    def __init__(self,camera_ID:int=0,img_width:int=640,img_height:int=480) -> None:
        """Class to handle camera stream

        Args:
            camera_ID (int, optional): camera index to choose the camera. Defaults to 0.
            img_width (int, optional): image width. Defaults to 640.
            img_height (int, optional): image height. Defaults to 480.

        Raises:
            IOError: If unable to open the camera 
        """
        self.camera_ON =False
        self.camera_ID = camera_ID
        self.img_height = img_height
        self.img_width = img_width
        self.cap = cv2.VideoCapture(camera_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        if not(self.cap.isOpened()):
            raise IOError("Failed to open the camera")
            
        else:
            self.camera_ON = True
        
    def camera_isON(self):
        """To check if the camera is succesfully ON

        Returns:
            Bool: True if camera is ON
        """
        return self.camera_ON
    
    def next_frame(self):
        """Returns the image/frame from the camera feed

        Returns:
            numpy.ndarray: image/frame from the camera feed
        """
        ret, image = self.cap.read()
        if not ret:
            print("Failed to read image")
        else:
            image = cv2.resize(src=image, dsize=(self.img_width,self.img_height), interpolation = cv2.INTER_AREA)
        return image
    
    def camera_OFF(self):
        """Switches off the camera when called
        """
        print("Switching off the camera")
        self.camera_ON =False
        self.cap.release()
        cv2.destroyAllWindows()
        
      
def main():
    camera = VideoStream(-1)

    while(camera.camera_isON()):
        image = camera.next_frame()
        cv2.imshow("TEST",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera.camera_OFF()
    
    print("Complete")

if __name__ == '__main__':
    main()