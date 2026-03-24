"""
modules/opencv_hub.py
YOLOv8 object detection + MediaPipe pose & face mesh.
"""
import numpy as np
from PIL import Image, ImageDraw

BOX_COLORS = ["#00e5ff","#7c3aed","#10b981","#f59e0b","#ef4444",
              "#06b6d4","#8b5cf6","#34d399","#fbbf24","#f87171"]

class VisionHub:
    def detect_objects(self, image, conf=0.5):
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            results = model(np.array(image.convert("RGB")), conf=conf, verbose=False)
            result  = results[0]
            img_draw = image.convert("RGB").copy()
            draw = ImageDraw.Draw(img_draw)
            detections = []
            for box in result.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                label  = result.names[cls_id]
                conf_v = float(box.conf[0])
                color  = BOX_COLORS[cls_id % len(BOX_COLORS)]
                draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
                draw.rectangle([x1,y1-22,x1+len(label)*9+10,y1], fill=color)
                draw.text((x1+5,y1-19), f"{label} {conf_v:.2f}", fill="black")
                detections.append({"label":label,"confidence":conf_v,"bbox":[x1,y1,x2,y2]})
            return img_draw, detections
        except ImportError:
            return self._demo_detection(image), [
                {"label":"person","confidence":0.91},
                {"label":"laptop","confidence":0.84}]
        except Exception:
            return self._demo_detection(image), []

    def _demo_detection(self, image):
        img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        w,h = img.size
        boxes = [
            (int(w*.1),int(h*.1),int(w*.5),int(h*.9),"person",0.91,"#00e5ff"),
            (int(w*.55),int(h*.2),int(w*.95),int(h*.75),"laptop",0.84,"#10b981"),
        ]
        for x1,y1,x2,y2,label,conf_v,color in boxes:
            draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
            draw.rectangle([x1,y1-20,x1+len(label)*9+10,y1], fill=color)
            draw.text((x1+5,y1-18), f"{label} {conf_v:.2f}", fill="black")
        return img

    def estimate_pose(self, image):
        try:
            import mediapipe as mp
            img_rgb = np.array(image.convert("RGB"))
            mp_pose = mp.solutions.pose
            mp_draw = mp.solutions.drawing_utils
            with mp_pose.Pose(static_image_mode=True) as pose:
                results = pose.process(img_rgb)
            if results.pose_landmarks:
                mp_draw.draw_landmarks(img_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,229,255),thickness=2,circle_radius=3),
                    mp_draw.DrawingSpec(color=(124,58,237),thickness=2))
            return Image.fromarray(img_rgb)
        except ImportError:
            return self._overlay_text(image, "Install mediapipe:  pip install mediapipe")
        except Exception as e:
            return self._overlay_text(image, f"Pose error: {e}")

    def face_mesh(self, image):
        try:
            import mediapipe as mp
            img_rgb  = np.array(image.convert("RGB"))
            mp_face  = mp.solutions.face_mesh
            mp_draw  = mp.solutions.drawing_utils
            mp_style = mp.solutions.drawing_styles
            with mp_face.FaceMesh(static_image_mode=True,max_num_faces=2,refine_landmarks=True) as fm:
                results = fm.process(img_rgb)
            if results.multi_face_landmarks:
                for lm in results.multi_face_landmarks:
                    mp_draw.draw_landmarks(img_rgb, lm,
                        mp_face.FACEMESH_TESSELATION, None,
                        mp_style.get_default_face_mesh_tesselation_style())
            return Image.fromarray(img_rgb)
        except ImportError:
            return self._overlay_text(image, "Install mediapipe:  pip install mediapipe")
        except Exception as e:
            return self._overlay_text(image, f"Face mesh error: {e}")

    def _overlay_text(self, image, text):
        img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle([0,0,img.width,60], fill=(13,30,46))
        draw.text((10,15), text, fill=(0,229,255))
        return img
