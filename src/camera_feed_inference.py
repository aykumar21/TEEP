import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image as PILImage

class FloodClassifier:
    def __init__(self, model_path):
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 classes: Flooded, Not Flooded
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, frame):
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)
            _, predicted = torch.max(output, 1)

        return "Flooded" if predicted.item() == 0 else "Non-Flooded"


# (Optional) Retain your standalone ROS node
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraInferenceNode(Node):
    def __init__(self):
        super().__init__('camera_inference_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10
        )
        self.classifier = FloodClassifier('flood_detection_model.pth')

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().info(f"Error converting image: {str(e)}")
            return

        label = self.classifier.predict(cv_image)

        cv2.putText(cv_image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Camera Feed", cv_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CameraInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
