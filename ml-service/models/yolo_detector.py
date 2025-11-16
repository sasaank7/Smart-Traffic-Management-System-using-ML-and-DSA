"""
YOLOv8 Vehicle Detection Module

Detects and counts vehicles from CCTV frames
"""
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Detection:
    """Vehicle detection result"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3],
            },
            "center": {"x": self.center[0], "y": self.center[1]},
            "timestamp": self.timestamp.isoformat(),
        }


class VehicleDetector:
    """
    YOLOv8-based vehicle detection

    Uses Ultralytics YOLOv8 for real-time vehicle detection
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        try:
            from ultralytics import YOLO
            self.yolo_available = True
        except ImportError:
            print("Warning: ultralytics not installed. Using mock detector.")
            self.yolo_available = False
            return

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Vehicle classes from COCO dataset
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            1: "bicycle",
        }

        # Load model
        if Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Download pretrained model
            print(f"Downloading YOLOv8 model...")
            self.model = YOLO("yolov8n.pt")  # nano model

        # Statistics
        self.total_detections = 0
        self.frame_count = 0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles in a single frame

        Args:
            frame: Input image as numpy array (BGR format)

        Returns:
            List of Detection objects
        """
        if not self.yolo_available:
            return self._mock_detection(frame)

        self.frame_count += 1

        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        # Parse detections
        detections = []
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get class
                cls_id = int(box.cls[0])

                # Filter vehicle classes only
                if cls_id not in self.vehicle_classes:
                    continue

                # Get confidence
                confidence = float(box.conf[0])

                # Get bounding box (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Convert to xywh
                width = x2 - x1
                height = y2 - y1

                # Calculate center
                center_x = int(x1 + width / 2)
                center_y = int(y1 + height / 2)

                # Create detection
                detection = Detection(
                    class_name=self.vehicle_classes[cls_id],
                    confidence=confidence,
                    bbox=(x1, y1, width, height),
                    center=(center_x, center_y),
                    timestamp=datetime.utcnow(),
                )

                detections.append(detection)

        self.total_detections += len(detections)

        return detections

    def detect_and_draw(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        """
        Detect vehicles and draw bounding boxes on frame

        Args:
            frame: Input image

        Returns:
            (annotated_frame, detections)
        """
        detections = self.detect(frame)
        annotated = frame.copy()

        for det in detections:
            x, y, w, h = det.bbox

            # Draw bounding box
            color = self._get_color(det.class_name)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Draw vehicle count
        count_text = f"Vehicles: {len(detections)}"
        cv2.putText(
            annotated,
            count_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return annotated, detections

    def count_vehicles(self, frame: np.ndarray) -> Dict[str, int]:
        """
        Count vehicles by type

        Args:
            frame: Input image

        Returns:
            Dictionary of vehicle_type -> count
        """
        detections = self.detect(frame)

        counts = {vehicle_type: 0 for vehicle_type in self.vehicle_classes.values()}
        counts["total"] = 0

        for det in detections:
            counts[det.class_name] += 1
            counts["total"] += 1

        return counts

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        frame_skip: int = 1,
    ) -> Dict:
        """
        Process video and detect vehicles

        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            frame_skip: Process every Nth frame (for speed)

        Returns:
            Statistics dictionary
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Processing
        frame_idx = 0
        all_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            # Detect
            annotated, detections = self.detect_and_draw(frame)
            all_detections.extend(detections)

            # Write
            if writer:
                writer.write(annotated)

            frame_idx += 1

            # Progress
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

        # Cleanup
        cap.release()
        if writer:
            writer.release()

        # Statistics
        stats = {
            "total_frames": total_frames,
            "processed_frames": frame_idx,
            "total_detections": len(all_detections),
            "avg_vehicles_per_frame": len(all_detections) / frame_idx if frame_idx > 0 else 0,
        }

        return stats

    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for vehicle type (BGR format)"""
        colors = {
            "car": (0, 255, 0),  # Green
            "truck": (0, 0, 255),  # Red
            "bus": (255, 0, 0),  # Blue
            "motorcycle": (255, 255, 0),  # Cyan
            "bicycle": (0, 255, 255),  # Yellow
        }
        return colors.get(class_name, (128, 128, 128))

    def _mock_detection(self, frame: np.ndarray) -> List[Detection]:
        """Mock detection for when YOLO is not available"""
        # Return random detections for testing
        import random

        num_vehicles = random.randint(5, 15)
        height, width = frame.shape[:2]

        detections = []
        for _ in range(num_vehicles):
            class_name = random.choice(["car", "truck", "bus", "motorcycle"])
            confidence = random.uniform(0.5, 0.99)
            x = random.randint(0, width - 100)
            y = random.randint(0, height - 100)
            w = random.randint(50, 100)
            h = random.randint(50, 100)

            detection = Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=(x, y, w, h),
                center=(x + w // 2, y + h // 2),
                timestamp=datetime.utcnow(),
            )
            detections.append(detection)

        return detections

    def get_stats(self) -> Dict:
        """Get detector statistics"""
        return {
            "total_detections": self.total_detections,
            "frame_count": self.frame_count,
            "avg_detections_per_frame": self.total_detections / self.frame_count if self.frame_count > 0 else 0,
        }
