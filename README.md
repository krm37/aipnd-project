# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


## Examples train.py

Help:
```bash
python ./train.py -h
```

Train on **CPU** with default **vgg19**:
```bash
python ./train.py ./flowers/
```

Train on **GPU** with **densenet169** with one **256** node layer:
```bash
python ./train.py ./flowers/train --gpu --arch "densenet169" --hidden_units 1024 --epochs 5
```

Additional hidden layers with checkpoint saved to densenet169 directory.
```bash
python ./train.py ./flowers/train --gpu --arch densenet169 --hidden_units 1024,512 --save_dir saved_models/
```

## Examples predict.py

Help
```bash
python ./predict.py -h
```

Basic Prediction
```bash
python ./predict.py flowers/valid/5/image_05192.jpg checkpoint.pth
```

Prediction with Top 10 Probabilities
```bash
python ./predict.py flowers/valid/5/image_05192.jpg save_models/checkpoint.pth --top_k 10
```

Prediction with GPU
```bash
python ./predict.py flowers/valid/5/image_05192.jpg checkpoint.pth --gpu
```


## Part 2 Scripts

### [train.py]

**Options:**

- Set directory to save checkpoints
    - `python train.py data_dir --save_dir save_directory`
- Choose architecture
    - `python train.py data_dir --arch "vgg19"`
- Set hyperparameters
    - `python train.py data_dir --learning_rate 0.01 --hidden_units 512,256 --epochs 20`
- Use GPU for training
    - `python train.py data_dir --gpu`

**Help** - `python train.py -h`:
```plain
usage: python ./train.py ./flowers/train --gpu --learning_rate 0.001 --hidden_units 512 --epochs 5

Train and save an image classification model.

positional arguments:
  data_directory

Arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Directory to save training checkpoint file (default:save_models/
                        )

  --category_names CATEGORIES_JSON
                        Path to file containing the categories. (default:
                        cat_to_name.json)

  --arch ARCH           Supported architectures: vgg19(25088-input_unit_size), mobilenet_v2(1280-input_unit_size), densenet169(1664-input_unit_size)
                        (default: vgg19)
  --gpu                 Use GPU (default: False)

hyperparameters:
  --learning_rate LEARNING_RATE
                        Learning rate (default: 0.01)
  --hidden_units [HIDDEN_UNITS ...]
                        Hidden layer units (default: [512])
  --epochs EPOCHS       Epochs (default: 20)
```

### [predict.py]

- Basic usage
    - `python predict.py /path/to/image checkpoint`
- Options
    - Return top KK most likely classes
        - `python predict.py input checkpoint --top_k 3`
    - Use a mapping of categories to real name
        - `python predict.py input checkpoint --category_names cat_to_name.json`
    - Use GPU for inference
        - `python predict.py input checkpoint --gpu`

**Help** - `python ./predict.py -h`:
```plain
usage: python ./predict.py /path/to/image.jpg checkpoint.pth

Image prediction.

positional arguments:
  input            path to input image.
  checkpoint       Path to checkpoint file.

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Directory to save training checkpoint file (default:saved_model/)
  --top_k TOP_K         Return top KK most likely classes. (default: 5)
  --category_names CATEGORIES_JSON
                        Path to file containing the categories. (default:
                        cat_to_name.json)
  --gpu                 Use GPU (default: False)

```

