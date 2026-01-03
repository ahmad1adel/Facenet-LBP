"""
Training script for masked dataset with FaceNet:
- FaceNet architecture (instead of MobileNet)
- YOLO detector only
- 20 epochs
- Batch size 16
- Learning rate 0.01
- Fine-tuning enabled
- Gaussian filtering enabled
- 160x160 input size (FaceNet standard)
"""

from src_masked_facenet.pipeline import FaceRecognitionPipeline
import os

def main():
    print("=" * 70)
    print("Face Recognition Training - Masked Dataset with FaceNet")
    print("FaceNet | YOLO | 20 Epochs | Batch 16 | LR 0.01 | WITH FILTERING")
    print("=" * 70)
    
    # Configuration
    train_dir = r'..\self-built-masked-face-recognition-dataset\AFDB_masked_face_dataset'  # Relative path
    model_dir = 'models/masked_facenet_model'
    
    # Check dataset
    if not os.path.exists(train_dir):
        print(f"\nError: Dataset not found at '{train_dir}'")
        return
    
    print(f"\nDataset: {train_dir}")
    print(f"Output: {model_dir}")
    print("\nConfiguration:")
    print("  - Architecture: FaceNet (Inception blocks)")
    print("  - Input Size: 160x160 (FaceNet standard)")
    print("  - Detector: YOLO (fixed)")
    print("  - Identification: Cosine Similarity (threshold: 0.55)")
    print("  - Fine-tuning: ENABLED")
    print("  - Epochs: 20")
    print("  - Batch Size: 16")
    print("  - Learning Rate: 0.01")
    print("  - Filtering: Gaussian (ENABLED)")
    print("  - Mask Detection: ENABLED")
    print("=" * 70)
    
    # Initialize pipeline with FaceNet
    print("\n[1/3] Initializing FaceNet pipeline...")
    pipeline = FaceRecognitionPipeline(
        target_size=(160, 160),     # FaceNet uses 160x160
        remove_bg=True,
        filter_type='gaussian',     # FILTERING ENABLED
        detector_type='yolo',       # YOLO only
        similarity_threshold=0.55,  # Cosine similarity threshold
        embedding_dim=128           # FaceNet embedding dimension
    )
    print("✓ FaceNet pipeline initialized with filtering and cosine similarity")
    
    # Train with fine-tuning
    print("\n[2/3] Training with FaceNet fine-tuning...")
    print("-" * 70)
    try:
        pipeline.train(
            train_dir=train_dir,
            val_dir=None,
            fine_tune_embedder=True,  # Enable FaceNet fine-tuning
            epochs=20,                 # 20 epochs as requested
            batch_size=16,             # Batch size 16 as requested
            learning_rate=0.01         # Learning rate 0.01 as requested
        )
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save models
    print("\n[3/3] Saving models...")
    os.makedirs(model_dir, exist_ok=True)
    pipeline.save_pipeline(model_dir)
    
    print("\n" + "=" * 70)
    print("✓ Training completed successfully!")
    print(f"✓ Models saved to: {model_dir}")
    print("=" * 70)
    print("\nYou can now use the trained FaceNet model for predictions.")
    print("This model includes:")
    print("  ✓ Cosine similarity for person identification")
    print("  ✓ FaceNet architecture (better than MobileNet)")
    print("  ✓ Gaussian filtering for noise reduction")
    print("  ✓ Mask detection capability")
    print("  ✓ Fine-tuned for masked faces")

if __name__ == '__main__':
    main()
