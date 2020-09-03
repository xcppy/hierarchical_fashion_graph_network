# Dataset.
Our experiment is based on the [POG datasest](https://github.com/wenyuer/POG).

we reprocess the data and save the files here.

# Data Format

## train_uo.txt
The training outfits for each user.
```
 user_id outfit_id outfit_id ...
```

## test_uo.txt
The test outfits for each user in recommendation task.
```
 user_id outfit_id outfit_id ...
```

## fltb_train_outfit_item.txt
The composed items for all the outfits.
```
 outfit_id item_id item_id ...
```

## fltb_feat_resnet152_02.npy
The visual features for all the items. The size is N * D.

>N is the number of items, D is the dimensionality of the feature.<br>
>The model we used to extract the visual features is downloaded from [here](https://github.com/tensorflow/models/tree/master/research/slim).<br>
>We use the model *ResNet V1 152* to extract 2048D visual features.

## fltb_item_cate2.txt
The category files for items.
```
 item_id category_id
```

## fltb_test_file.txt
The test files for fill in the blank task.
```
 outfit_id; outfit_length(int); mask_position(int); pos_outfit; neg_outfit; neg_outfit; neg_outfit
 
 pos_outfit: item_id, item_id, ...
 neg_outfit: item_id, item_id, ...
```





