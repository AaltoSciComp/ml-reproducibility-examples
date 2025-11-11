# **Reproducibility and Open Science Practices in Machine Learning**

*A workshop for the AI-DOC doctoral researchers, 11/11/2025, Aalto University*  
This repository contains the materials used in the workshop and examples related to machine learning reproducibility.

## **Schedule**

* 9:00 || Motivational intro: [Fundamentals of reproducibility in Machine Learning](https://zenodo.org/records/17575944) (Enrico Glerean)  
* 9:20 || Practicalities (EG):  
  * [https://noppe.2.rahtiapp.fi/welcome](https://noppe.2.rahtiapp.fi/welcome) \-\> HAKA login (noppe workspace password given during the workshop)  
* 9:45 || [Jupyter reproducibility](https://github.com/AaltoSciComp/ml-reproducibility-examples/blob/main/jupyter_reproducibility.ipynb) (Luca Ferranti)  
* 10:00 || Environment reproducibility (Simo Tuomisto)  
  * Motivation: environment reproducibility good \[pip install in default bad\]  
    * Demo:  
      * Show environment  
      * Create container from environment: [https://github.com/simo-tuomisto/micromamba-apptainer](https://github.com/simo-tuomisto/micromamba-apptainer)  
      * Create apptainer in CSC machine [https://coderefinery.github.io/hpc-containers/intro\_and\_motivation/\#](https://coderefinery.github.io/hpc-containers/intro_and_motivation/#)  
* *10:30 || break*  
* 10:45 How do you track your work / training: MLflow works for trad ML and DL (Hossein Firooz and ST)  
  * [Why do we track the training process? What are the approaches? WandB, MLFlow](https://github.com/AaltoSciComp/ml-reproducibility-examples/blob/main/model_tracking/Monitor%20Training%20W%26B%20and%20MLflow.pdf) (HF)  
  * [Examples demo with MLflow](https://github.com/AaltoSciComp/ml-reproducibility-examples/tree/main/model_tracking) (ST)  
* 11:30 || [Overfitting & overreproducing](https://github.com/AaltoSciComp/ml-reproducibility-examples/blob/main/overfitting.pdf) (LF)  
* *12:00 || Lunch \- on your own*  
* 13:00 || [How do you create reproducible DL training? How do you reproduce the training & creation?](https://github.com/AaltoSciComp/ml-reproducibility-examples/tree/main/modular_training) Lightning (ST) (30min)  
  * Motivation:  
    * Dataset  
    * Model  
    * Trainer  
    * CLI: “main()”  
    * configuration management  
    * checkpointing  
  * Example model in PyTorch Lightning  
* *13:45 || break*  
* 14:00 || [Model sharing](https://github.com/AaltoSciComp/ml-reproducibility-examples/tree/main/model_sharing) How do you share models? Huggingface (ST)  
  * Model card  
    * Model parameters  
    * Model structure as a code  
  * Model weights & used tokenizers  
    * Storage formats: safetensors  
* 14:15 || [Scaling](https://github.com/AaltoSciComp/ml-reproducibility-examples/tree/main/scaling_multi_gpu): How do you read how big players are doing it and how do you get there? (HF)  
  * How to understand the hardware?  
  * DataLoader  
    * Parallelism and workers.  
    * [Sampling and generator](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)  
    * [Random number handling](https://docs.pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading)  
  * DP \-\> DDP \-\> Model / Tensor Parallel \-\> Deepspeed  
    * Demo on triton  
* 14:55 || Outro “What next?”, good reproducibility practices of any research project (ie. the coderefinery workshop). [https://coderefinery.github.io/reproducible-research/](https://coderefinery.github.io/reproducible-research/) (EG)  
* *15:00 || The end*

