# Attend2u

![alt tag](./assets/attend2u_cvpr.png)

This project hosts the code for our **CVPR 2017** paper.

- Cesc Chunseong Park, Byeongchang Kim and Gunhee Kim. *Attend to You*: Personalized Image Captioning with Context Sequence Memory Networks. In *CVPR*, 2017. (**Spotlight**) [[arxiv]](https://arxiv.org/abs/1704.06485)

We address personalization issues of image captioning, which have not been discussed yet in previous research.
For a query image, we aim to generate a descriptive sentence, accounting for prior knowledge such as the user's active vocabularies in previous documents.
As applications of personalized image captioning, we tackle two post automation tasks: hashtag prediction and post generation, on our newly collected Instagram dataset, consisting of 1.1M posts from 6.3K users.
We propose a novel captioning model named Context Sequence Memory Network (CSMN).

## Reference

If you use this code as part of any published research, please refer the following paper.

```
@inproceedings{attend2u:2017:CVPR,
    author    = {Cesc Chunseong Park and Byeongchang Kim and Gunhee Kim},
    title     = "{Attend to You: Personalized Image Captioning with Context Sequence Memory Networks}"
    booktitle = {CVPR},
    year      = 2017
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

3. Download our dataset (*coming soon*)

```
cd ${project_root}/scripts
./download_dataset.sh
```

4. Generate formatted dataset and extract Resnet-101 pool5 features

```
cd ${project_root}/scripts
./extract_features.sh
```

### Training

Run training script.
You can train the model with multiple gpus.

```
python -m train --num_gpus 4 --batch_size 200
```

### Evaluation

Run evaluation script.
You can evaluate the model with multiple gpus

```
python -m eval --num_gpus 2 --batch_size 500
```

## Examples

Here are post generation examples:

![alt tag](./assets/post_example_1-1.png)

Here are hashtag generation examples:

![alt tag](./assets/hash_example_1-1.png)

Here are (little bit wrong but) interesting post generation examples:

![alt tag](./assets/post_example_interest_3-1.png)

Here are (little bit wrong but) interesting hashtag generation examples:

![alt tag](./assets/hash_example_interest_1-1.png)



## Acknowledgement

We implement our model using [tensorflow](http://tensorflow.org) package. Thanks for tensorflow developers. :)

We also thank Instagram for their API and Instagram users for their valuable posts.

Additionally, we thank [coco-caption](https://github.com/tylin/coco-caption) developers for providing caption evaluation tools.

## Authors

[Cesc Chunseong Park](http://vision.snu.ac.kr/cesc/), Byeongchang Kim and [Gunhee Kim](http://www.cs.cmu.edu/~gunhee/)

[Vision and Learning Lab](http://vision.snu.ac.kr/) @ Computer Science and Engineering, Seoul National University, Seoul, Korea

## License

MIT license
