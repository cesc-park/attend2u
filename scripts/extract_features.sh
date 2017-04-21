python generate_dataset.py
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --input_fname ../data/caption_dataset/train.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --input_fname ../data/caption_dataset/test1.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --input_fname ../data/caption_dataset/test2.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --input_fname ../data/hashtag_dataset/train.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --input_fname ../data/hashtag_dataset/test1.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --input_fname ../data/hashtag_dataset/test2.txt
