'''
TensorRT code was taken from stackoverflow: 
https://stackoverflow.com/questions/59280745/inference-with-tensorrt-engine-file-on-python, last visited 26.01.2022


Camera code was taken from JetsonHacksNano:
https://github.com/JetsonHacksNano/CSI-Camera, last-visited 26.01.2022 
'''


import collections
import numpy as np
import os
import time
import cv2
import os
import time
import tensorrt as trt
import tensorrt as trt
import numpy as np
import os
import pycuda.driver as cuda
import board
import busio
import adafruit_pca9685


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

                
                
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self,x:np.ndarray,batch_size=2):
        
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        
        self.stream.synchronize()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]


def gstreamer_pipeline(
    capture_width=320,
    capture_height=192,
    display_width=320,
    display_height=192,
    framerate=120,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    global frames
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        last_action = None
        skip_frames = 0
        pause_frames = 0
        start = time.time()

        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            _, img = cap.read()
            img_flipped = cv2.rotate(img, cv2.ROTATE_180) 
            pred_depth_img = pydnet_model(img_flipped / 255.0, 1)

        
            depth_pred_cpu = np.squeeze(pred_depth_img)
            gray_scaled_img = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2GRAY) / 255.0
            img_to_show = np.hstack((gray_scaled_img, pred_depth_img[0].reshape((192, 320))))

            d_min = np.min(depth_pred_cpu)
            d_max = np.max(depth_pred_cpu)
            depth_pred_col = (depth_pred_cpu - d_min) / (d_max - d_min)
            image_buffer.appendleft(depth_pred_col)
            
            if skip_frames <= 0:
                if pause_frames > 0:
                    pca.channels[0].duty_cycle = THROTTLE_NEUTRAL_DUTY_CYCLE
                    pause_frames -= 1
                else:

                    if len(image_buffer) == 4:
                        state = np.stack(image_buffer, axis = 0)
                        actions = q_model(state,1)

                        action = np.argmax(actions)

                        pca.channels[0].duty_cycle = THROTTLE_MAX_DUTY_CYCLE

                        if action == 0:
                            if last_action != action:
                                pca.channels[1].duty_cycle = STEERING_LEFT_DUTY_CYCLE
                            skip_frames = 3

                        elif action == 1:
                            if last_action != action:
                                pca.channels[1].duty_cycle = STEERING_RIGHT_DUTY_CYCLE
                            skip_frames = 3
            
                        elif action == 2:
                            if last_action != action:
                                pca.channels[1].duty_cycle = STEERING_LEFT_DUTY_CYCLE
                            skip_frames = 2
                    
                        elif action == 3:
                            if last_action != action:
                                pca.channels[1].duty_cycle = STEERING_RIGHT_DUTY_CYCLE
                            skip_frames = 2
                        
                        elif action == 4:
                            if last_action != action:                    
                                pca.channels[1].duty_cycle = STEERING_FORWARD_DUTY_CYCLE 
                            skip_frames = 3

                        elif action == 5:
                            if last_action != action:                    
                                pca.channels[1].duty_cycle = STEERING_FORWARD_DUTY_CYCLE
                            skip_frames = 2
                        
                        last_action = action
                        pause_frames = 10



            skip_frames -= 1


                
            cv2.imshow("CSI Camera", img_to_show)

            # This also acts as
            keyCode = cv2.waitKey(1) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
            
            frames += 1

        end = time.time()

        print('Average FPS: {}'.format(int(frames / (end - start))))
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


CONSECUTIVE_IMAGES = 4
frames = 0
image_buffer = collections.deque(maxlen = CONSECUTIVE_IMAGES)

# Should be configured for every rc-car
THROTTLE_MAX_DUTY_CYCLE = 3000
THROTTLE_NEUTRAL_DUTY_CYCLE = 4900

STEERING_FORWARD_DUTY_CYCLE = 5000
STEERING_LEFT_DUTY_CYCLE = 3550
STEERING_RIGHT_DUTY_CYCLE = 6450

i2c = busio.I2C(board.SCL, board.SDA)
pca = adafruit_pca9685.PCA9685(i2c_bus = i2c, address = 0x40)
pca.frequency = 50

pca.channels[0].duty_cycle = THROTTLE_NEUTRAL_DUTY_CYCLE
pca.channels[1].duty_cycle = STEERING_FORWARD_DUTY_CYCLE


batch_size = 1
q_model_engine_path = os.path.join(".","optimized_models","q_model_650.trt")
q_model = TrtModel(q_model_engine_path)
pydnet_model_engine_path = os.path.join(".","optimized_models","pydnet.trt")
pydnet_model = TrtModel(pydnet_model_engine_path)


show_camera()

