#Pipeline for Megapixel Resolution Whole Slide Image Analysis
This project provides a comprehensive pipeline for the analysis of whole slide images at megapixel resolution. It leverages the power of deep learning and high-resolution image processing to facilitate advanced research and applications in the field of digital pathology and related areas.

##Installation
To set up the project environment, you need to install the required dependencies:
```pip install -r requirements.txt```
##Usage
The main script of the pipeline is executed using the torchrun command. Below is a detailed description of the arguments that can be used:

--project: The name of the project (type: string, required).
--shift: cross validation fold shift index, with options 0, 1, 2, 3, 4 (type: integer, default: 0).
--epochs: Number of training epochs (type: integer, default: 250).
--data_dir: Directory for input data (type: string, required).
--models_dir: Directory to save or retrieve the model's weights (type: string, required).
--type: Type of the neural network model, e.g., 'vit[2048]' (type: string, default: "vit[2048]").
--device: The device to run the model on, either 'cuda' or 'cpu' (type: string, default: 'cuda').
--batch_size: Batch size for training/testing (type: function check_batch_size, required).
--lr: Learning rate for training (type: float, required).
--oversample: Oversampling rate (type: float, default: 0.5).
-t: Test only run, if set (action: "store_true", default: False).
-i: Train with smaller images, if set (action: "store_true", default: False).
-p: Pin Memory for more efficient memory management, if set (action: "store_true", default: False).
--clinical_data: Path to additional clinical data file for the classification net (type: string, default: None).
Example Execution
```torchrun main.py --project "ExampleProject" --shift 2 --epochs 100 --data_dir "/path/to/data" --models_dir "/path/to/models" --type "vit[2048]" --device cuda --batch_size 32 --lr 0.001 --oversample 0.3 -t --clinical_data "/path/to/clinical_data.csv"```
This command runs the pipeline with specified project name, shift mode, number of epochs, data directories, model type, device, batch size, learning rate, oversampling rate, in test mode, and with an optional clinical data file.
