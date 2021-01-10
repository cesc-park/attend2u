# Attend2u

![alt tag](./assets/attend2u_cvpr.png)

This project hosts the code for our **CVPR 2017** paper and **TPAMI 2018** paper.

- Cesc Chunseong Park, Byeongchang Kim and Gunhee Kim. *Attend to You*: Personalized Image Captioning with Context Sequence Memory Networks. In *CVPR*, 2017. (**Spotlight**) [[arxiv]](https://arxiv.org/abs/1704.06485)
- Cesc Chunseong Park, Byeongchang Kim and Gunhee Kim. Towards Personalized Image Captioning via Multimodal Memory Networks. In *IEEE TPAMI*, 2018. [[pdf]](https://ieeexplore.ieee.org/abstract/document/8334621/)

We address personalization issues of image captioning, which have not been discussed yet in previous research.
For a query image, we aim to generate a descriptive sentence, accounting for prior knowledge such as the user's active vocabularies in previous documents.
As applications of personalized image captioning, we tackle two post automation tasks: hashtag prediction and post generation, on our newly collected Instagram dataset, consisting of 1.1M posts from 6.3K users.
We propose a novel captioning model named Context Sequence Memory Network (CSMN).

## Reference

If you use this code or dataset as part of any published research, please refer one of the following papers.

```
@inproceedings{attend2u:2017:CVPR,
    author    = {Park, Cesc Chunseong and Kim, Byeongchang and Kim, Gunhee},
    title     = "{Attend to You: Personalized Image Captioning with Context Sequence Memory Networks}",
    booktitle = {CVPR},
    year      = 2017
}
```

```
@inproceedings{attend2u:2018:TPAMI,
    author    = {Park, Cesc Chunseong and Kim, Byeongchang and Kim, Gunhee},
    title     = "{Towards Personalized Image Captioning via Multimodal Memory Networks}",
    booktitle = {IEEE TPAMI},
    year      = 2018
}
```

## Running Code

### Get our code

```
git clone https://github.com/cesc-park/attend2u
```

### Prerequisites

1. Install python modules

```
pip install -r requirements.txt
```

2. Download pre-trained resnet checkpoint

```
cd ${project_root}/scripts
./download_pretrained_resnet_101.sh
```

3. Download our version of *YFCC100M* dataset

You can download our personalized image captioning split of [*YFCC100M*](https://arxiv.org/abs/1503.01817) dataset

Download data from the links below and save it to `${project_root}/data_yfcc`.

[[Download json (YFCC100M)]](https://drive.google.com/uc?export=download&id=1e148C7RTetbOmz982gZvbHdnbw84LDxS)
[[Download images (YFCC100M)]](https://drive.google.com/uc?export=download&id=1TXHt-0oLig4MbAHMRuBezdVjclYZzLFc)

```
cd ${project_root}/data_yfcc
tar -xvf yfcc_json.tar.gz
tar -xvf yfcc_images.tar.gz
```

4. Generate formatted dataset and extract Resnet-101 pool5 features

```
cd ${project_root}/scripts
./extract_yfcc_features.sh
```

### Training

Run training script.
You can train the model with multiple gpus.

```
python -m train --num_gpus 4 --batch_size 200 --data_dir ./data_yfcc/caption_dataset
```

### Evaluation

Run evaluation script.
You can evaluate the model with multiple gpus

```
python -m eval --num_gpus 2 --batch_size 500 --data_dir ./data_yfcc/caption_dataset
```

## *InstaPIC-1.1M* Dataset

Temporarily not supported.

## *YFCC100M* Dataset

[*YFCC100M*](https://arxiv.org/abs/1503.01817) (Yahoo Flickr Creative Commons 100 Million Dataset) consists of 100 million Flickr user-uploaded images and videos between 2004 and 2014
along with their corresponding metadata including titles, descriptions, camera types and usertags.
We processed a series of filtering to make personalized image captioning split of *YFCC100M*.
We regard the titles and descriptions as captions and usertags as hashtags.

Key statistics of personalized image captioning splitted *YFCC100M* dataset are outlined below.
We also show average and median (in parentheses) values.
The total unique posts and users in our dataset are (867,922/11,093)

| Dataset       | #posts        | #users        | #posts/user | #words/post |
|:-------------:|:-------------:|:-------------:|:-----------:|:-----------:|
| caption       | 462,036       | 6,197         | 74.6 (40)   | 6.30 (5)    |
| hashtag       | 434,936       | 5,495         | 79.2 (49)   | 7.46 (6)    |

If you download and uncompress the dataset correctly, structure of dataset will follow the below structure.

```
{project_root}/data_yfcc
├── json
│   ├── yfcc-caption-train.json
│   ├── yfcc-caption-test.json
│   ├── yfcc-hashtag-train.json
│   ├── yfcc-hashtag-test1.json
│   └── yfcc-hashtag-test2.json
└── images
    ├── {user1_id}_{post1_id}
    ├── {user1_id}_{post2_id}
    ├── {user2_id}_{post1_id}
    └── ...
```

We provide one type of test set for image captioning and two types of test set for hashtag prediction.

## Examples

Here are post generation examples:

![alt tag](./assets/examples_best_post-1.png)

Here are hashtag generation examples:

![alt tag](./assets/examples_best_hash-1.png)

Here are hashtag and post generation examples with query images and multiple predictions by different users:

![alt tag](./assets/examples_byaimg-1.png)

Here are (little bit wrong but) interesting post generation examples:

![alt tag](./assets/post_example_interest_3-1.png)

Here are (little bit wrong but) interesting hashtag generation examples:

![alt tag](./assets/hash_example_interest_1-1.png)



## Acknowledgement

We implement our model using [tensorflow](http://tensorflow.org) package. Thanks for tensorflow developers. :)

We also thank Instagram for their API and Instagram users for their valuable posts.

Additionally, we thank [coco-caption](https://github.com/tylin/coco-caption) developers for providing caption evaluation tools.

We also appreciate [Juyong Kim](http://juyongkim.com), [Yunseok Jang](https://yunseokjang.github.io) and [Jongwook Choi](https://wook.kr) for helpful comments and discussions.

We are further thankful to Hyunjae Woo for help with YFCC100M dataset preprocessing and [Amelie Schmidt-Colberg](http://vision.snu.ac.kr/people/amelie.html) for carefully correcting our English writing.

## Authors

[Cesc Chunseong Park](http://cesc-park.github.io/), [Byeongchang Kim](http://vision.snu.ac.kr/people/byeongchangkim.html) and [Gunhee Kim](http://vision.snu.ac.kr/~gunhee/)

[Vision and Learning Lab](http://vision.snu.ac.kr/) @ Computer Science and Engineering, Seoul National University, Seoul, Korea

## License

MIT license
