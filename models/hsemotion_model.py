"""
models/hsemotion_model.py - HSEmotion Pre-trained Model Wrapper
=================================================================
Wrapper for the HSEmotion library's pre-trained facial emotion recognition model.

HSEmotion provides EfficientNet-based models pre-trained on AffectNet dataset.
We wrap it to be compatible with our training/evaluation pipeline.

- Pre-trained on AffectNet (large-scale FER dataset)
- Supports 8 emotions natively, mapped to 6 Ekman emotions
- Can be fine-tuned on FER2013/RAF-DB for better performance

Reference:
    Savchenko, A.V. (2022). HSEmotion: High-Speed facial Emotion recognition
    
Install:
    pip install hsemotion
"""

import torch
import torch.nn as nn
import torchvision.models as models

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# HSEmotion 8-class -> 6 Ekman mapping
# HSEmotion classes: ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
# Our classes:       [0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Sad, 5:Surprise]
HSEMOTION_TO_EKMAN = {
    0: 0,   # Anger     -> Angry
    2: 1,   # Disgust   -> Disgust
    3: 2,   # Fear      -> Fear
    4: 3,   # Happiness -> Happy
    6: 4,   # Sadness   -> Sad
    7: 5,   # Surprise  -> Surprise
    # 1: Contempt -> excluded
    # 5: Neutral  -> excluded
}


class HSEmotionWrapper(nn.Module):
    """
    HSEmotion model wrapped for our 6-class Ekman emotion pipeline.

    Uses EfficientNet-B0 pretrained on AffectNet as backbone,
    then adds a new classifier for 6 Ekman emotions.

    Input: [batch, 3, 260, 260] (RGB)
    Output: [batch, 6] (6 emotion class logits)
    """

    def __init__(self, num_classes=6, freeze_backbone=True):
        """
        Args:
            num_classes (int): Number of output classes (default: 6)
            freeze_backbone (bool): Freeze pretrained backbone
        """
        super(HSEmotionWrapper, self).__init__()

        # Use EfficientNet-B0 as backbone (same architecture as HSEmotion)
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # Try to load HSEmotion pretrained weights
        self._load_hsemotion_weights()

        # Replace classifier for 6 Ekman classes
        in_features = self.backbone.classifier[1].in_features  # 1280

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

    def _load_hsemotion_weights(self):
        """
        Try to load HSEmotion pretrained weights into the backbone.
        Falls back to ImageNet weights if HSEmotion is not available.
        """
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            print("[INFO] HSEmotion library found. Loading AffectNet pretrained weights...")

            # Create temporary recognizer to get the model weights
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            recognizer = HSEmotionRecognizer(
                model_name=config.HSEMOTION_MODEL_NAME,
                device=device_str
            )

            # Extract backbone weights from HSEmotion model
            if hasattr(recognizer, 'model'):
                hsemotion_state = recognizer.model.state_dict()

                # Load matching keys (backbone features only)
                model_state = self.backbone.state_dict()
                matched_keys = 0
                for key in model_state:
                    if key in hsemotion_state and model_state[key].shape == hsemotion_state[key].shape:
                        model_state[key] = hsemotion_state[key]
                        matched_keys += 1

                self.backbone.load_state_dict(model_state)
                print(f"[INFO] Loaded {matched_keys} weight tensors from HSEmotion.")
            else:
                print("[WARNING] Could not extract HSEmotion model weights. Using ImageNet weights.")

            del recognizer

        except ImportError:
            print("[WARNING] hsemotion not installed. Using ImageNet weights.")
            print("[TIP] Install with: pip install hsemotion")
        except Exception as e:
            print(f"[WARNING] HSEmotion weight loading failed: {e}")
            print("[INFO] Falling back to ImageNet pretrained weights.")

    def _freeze_backbone(self):
        """Freeze backbone feature extractor, keep classifier trainable."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Unfreeze last 2 blocks for some adaptation
        total_blocks = len(self.backbone.features)
        for i in range(max(0, total_blocks - 2), total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

        # Classifier always trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Make all layers trainable (full fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): [batch, 3, H, W] (RGB input)

        Returns:
            Tensor: [batch, num_classes] logit output
        """
        return self.backbone(x)

    def get_feature_vector(self, x):
        """
        Returns the feature vector before the final FC layer.

        Returns:
            Tensor: Feature vector of shape [batch, 256]
        """
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # Pass through first layers of classifier up to 256-dim
        for i in range(6):  # Dropout -> Linear(512) -> ReLU -> BN -> Dropout -> Linear(256)
            x = self.backbone.classifier[i](x)

        return x


def get_hsemotion_model(num_classes=None, pretrained_path=None,
                        freeze_backbone=True):
    """
    HSEmotion model factory function.

    Args:
        num_classes (int): Number of classes
        pretrained_path (str): Path to previously fine-tuned model
        freeze_backbone (bool): Freeze backbone

    Returns:
        HSEmotionWrapper: Model instance
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    model = HSEmotionWrapper(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )

    # Load previously fine-tuned model if available
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[INFO] Loading HSEmotion fine-tuned model: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=config.DEVICE)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print("[INFO] HSEmotion model loaded successfully.")

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    print(f"\n[MODEL] HSEmotion (AffectNet Pretrained + Fine-tune)")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")
    print(f"  Number of classes:    {num_classes}")
    print(f"  Backbone frozen:      {freeze_backbone}")

    return model
