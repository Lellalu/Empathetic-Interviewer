import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import time

# ============================================================
# MODEL ARCHITECTURE (must match training architecture)
# ============================================================

class SpatialAttentionHead(nn.Module):
    def __init__(self, feature_dim=384, num_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.emotion_query = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, patch_features):
        batch_size = patch_features.size(0)
        query = self.emotion_query.expand(batch_size, -1, -1)
        attended, attention_weights = self.attention(query, patch_features, patch_features)
        attended = self.norm1(attended + query)
        attended = self.norm2(attended + self.ffn(attended))
        return attended.squeeze(1), attention_weights.squeeze(1)


class EmotionClassificationHead(nn.Module):
    def __init__(self, feature_dim=384, hidden_dim=512, num_classes=7, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, features):
        return self.classifier(features)


class AdvancedEmotionRecognizer(nn.Module):
    def __init__(self, dino_model, num_classes=7, feature_dim=384, freeze_backbone=True, dropout=0.3):
        super().__init__()
        self.dino_model = dino_model
        if freeze_backbone:
            for param in self.dino_model.parameters():
                param.requires_grad = False
        self.spatial_attention = SpatialAttentionHead(feature_dim=feature_dim, num_heads=8, dropout=dropout)
        self.classification_head = EmotionClassificationHead(feature_dim=feature_dim, hidden_dim=512, num_classes=num_classes, dropout=dropout)
        
    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.dino_model(pixel_values, output_hidden_states=True)
        patch_features = outputs.last_hidden_state[:, 1:, :]
        attended_features, attention_weights = self.spatial_attention(patch_features)
        logits = self.classification_head(attended_features)
        return logits, attention_weights


class EmotionDetector:
    
    EMOTION_NAMES = {0: 'surprise', 1: 'fear', 2: 'disgust', 3: 'happiness', 4: 'sadness', 5: 'anger', 6: 'neutral'}
    
    def __init__(self, model_path='empathetic_interviewer_perception.pth', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[EmotionDetector] Using device: {self.device}")
        
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        dino_backbone = AutoModel.from_pretrained("facebook/dinov2-small")
        
        self.model = AdvancedEmotionRecognizer(dino_model=dino_backbone, num_classes=7, feature_dim=384, freeze_backbone=True, dropout=0.3).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"[EmotionDetector] Model loaded! Test accuracy: {checkpoint.get('test_accuracy', 'N/A'):.2%}")
        
        self.history = []
        self.history_size = 5
    
    def predict_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(frame_rgb), return_tensors="pt")
        with torch.no_grad():
            logits, _ = self.model(inputs['pixel_values'].to(self.device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_id = int(logits.argmax(dim=1).item())
        self.history.append(pred_id)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        smoothed_id = max(set(self.history), key=self.history.count)
        return {'emotion_id': smoothed_id, 'emotion_name': self.EMOTION_NAMES[smoothed_id], 'confidence': float(probs[smoothed_id]), 'all_probabilities': {self.EMOTION_NAMES[i]: float(probs[i]) for i in range(7)}}
    
    def predict_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits, _ = self.model(inputs['pixel_values'].to(self.device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_id = int(logits.argmax(dim=1).item())
        return {'emotion_id': pred_id, 'emotion_name': self.EMOTION_NAMES[pred_id], 'confidence': float(probs[pred_id]), 'all_probabilities': {self.EMOTION_NAMES[i]: float(probs[i]) for i in range(7)}}
    
    def run_webcam(self, camera_id=0, show_display=True):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        print("Webcam started. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = self.predict_frame(frame)
            if show_display:
                cv2.putText(frame, f"{result['emotion_name']} ({result['confidence']*100:.0f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = EmotionDetector('empathetic_interviewer_perception.pth')
    detector.run_webcam()