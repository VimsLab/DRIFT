This code is confidential, please do not distribute.
Due to the limitation of file sizes,
we could not include any weight files or datasets.
We will release full version of code, weights, and dataset once paper is accepted. 
(It's on github, but we can't reveal it due to double blind policy)

But we provide code for you to generate dataset and run our code.

Please first install cocoapi.

We use anaconda to set up the enviroment, and the enviroment settings is exported in 
"environment.yml"

Enverioments are requirements are listed

Generating dataset:
    =================
    synthetic dataset
    =================
    You can create your own synthetic dataset by running: 
    create_synthetic_data.py

    Line 126 to Line 147 you can change setting for synthetic data.
        kernel_size_list = [3, 5,7,11] ->> choose kernel_size ->> width
        cp_steps = [15,30,60] ->> choose stepsize  
        widths = [256,512] ->> choose image size

    You can create two sets of dataset with different image size. and train on 256 and inference 512.

    =================
    Worm dataset
    =================
    convert_worm_COCO.py will convert instnace lables to coco format with sequence of control points.
    Please download the datset from the website (https://bbbc.broadinstitute.org/BBBC010) and set the path in the code.

Training:
    =================
    For synthetic datset
    =================
    Set path in config_Synthetic.py
    Run train_synthetic.py

    =================
    For worm datset
    =================
    Set path in config_worm
    Run train_worm.py

Evaluation:
    =================
    For synthetic datset
    =================
    Set path in config_Synthetic.py
    Run eval_synthetic.py

    =================
    For worm datset
    =================
    Set path in config_worm
    Run eval_worm.py

Visualized evaluation will be saved in 'demo_output' folder

if you would like to visualize the inferene process shown in the demo videos. 
You can set visual to True





