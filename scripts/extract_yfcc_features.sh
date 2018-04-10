python generate_yfcc100m_dataset.py
python generate_yfcc100m_hashtag_dataset.py
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --image_dir ../data_yfcc/images/ --input_fname ../data_yfcc/caption_dataset/train.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --image_dir ../data_yfcc/images/ --input_fname ../data_yfcc/caption_dataset/test.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --image_dir ../data_yfcc/images/ --input_fname ../data_yfcc/hashtag_dataset/train.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --image_dir ../data_yfcc/images/ --input_fname ../data_yfcc/hashtag_dataset/test1.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --image_dir ../data_yfcc/images/ --input_fname ../data_yfcc/hashtag_dataset/test2.txt
