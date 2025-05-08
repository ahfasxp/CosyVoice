from modelscope import snapshot_download

print("Downloading CosyVoice-300M model...")
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
print("Finished downloading CosyVoice-300M model.")

# print("Downloading CosyVoice-ttsfrd model...")
# snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
# print("Finished downloading CosyVoice-ttsfrd model.")

print("All models downloaded.")
