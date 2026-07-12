import kagglehub

print("⏳ Connecting to Kaggle and downloading ASL-Citizen-Keypoints...")

# This pulls the pre-extracted dictionary landmarks directly to your machine
path = kagglehub.dataset_download("nguyenchitinh/asl-citizen")

print("\n🎉 Download Complete!")
print("=========================================")
print("Your dataset is safely stored locally at:")
print(path)
print("=========================================")