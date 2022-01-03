install:
	pip install -r requirements.txt

cartoons:
	python3	 cartoon_image_downloader.py

cartoons-smooth:
	python3 cartoon_image_smoothing.py

photos:
	python3 photo_downloader.py

install-transform:
	pip install --no-cache-dir -r requirements-transform.txt

transform:
	python3 transform.py $(IMAGE)

transformVideo:
	python3 transformVideo.py $(IMAGE)

extractFrames:
	python3 extract_frames.py
	
cleanupImageData:
	python3 remove_duplicate_images.py

detectBlur:
	python3 detect_blur.py --images blur_test --delete false