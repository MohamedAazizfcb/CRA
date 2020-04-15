import sys  #System.

import os   #Operating system.

import argparse #Argument parsing.

    """
        To ease the writing of command-line interfaces.
        It parses the defined arguments from the sys.
        Automatically generates help and usage messages.
        Issues errors when users give the program invalid arguments.
    """

import logging #Write log easily
    """
        provides a flexible framework for emitting the load of writing log messages.
        The module provides a way to configure different log handlers and a way of routing log messages to these handlers.
    """

from collections import defaultdict #Collections are containers that are used to store collections of data.
    """
        The defaultdict tool is a container in the collections.
        It's similar to the usual dictionary (dict) container.
        But the value fields' data type is specified upon initialization.
        For example: d = defaultdict(list) d['python'].
    """

import matplotlib #For Plotting

import numpy as np #General-purpose array-processing package

import soundfile as sf 
    """
        Is an audio library based on libsndfile, CFFI and NumPy.
        Can read and write sound files.
        It is accessed through CFFI.
    """

import torch #The core machine learning library we have used in training 

import yaml #YAML format

    """
        Is a human-readable data-serialization language.
        It is commonly used for configuration files.
        But it is also used in data storage (e.g. debugging output) or transmittion (e.g. document headers).
    """

from tensorboardX import SummaryWriter #This allows a training program to call methods to add data to the file directly from the training loop, without slowing down training
    """
        TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow.
        It enables tracking experiment metrics like loss and accuracy.
        visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.
        We used only the SummaryWriter with pytorch.
        SummaryWriter class provides a high-level API to create an event file in a given directory and add summaries and events to it.
        The class updates the file contents asynchronously.
    """

from torch.utils.data import DataLoader
    """
        Part of the application's data fetching layer.
        Provides a simplified and consistent API over remote data sources such as databases via batching and caching.
    """

from tqdm import tqdm #A progress bar library with good support for nested loops and notebooks.


#importing our modules
import parallel_wavegan 
import parallel_wavegan.models 
from parallel_wavegan.datasets import AudioMelDataset   
from parallel_wavegan.datasets import AudioMelSCPDataset
from parallel_wavegan.losses import MultiResolutionSTFTLoss #STFT loss function
from parallel_wavegan.optimizers import RAdam #R_Adam optimizer.
from parallel_wavegan.utils import read_hdf5 #Utility functions.


matplotlib.use("Agg") #To avoid matplotlib error in command-line interface (CLI) environment


class Trainer(object):
    """this module is used to train parallel wavegan"""

    def __init__(self,
                 steps,                         #(int): Initial number of global steps.
                 epochs,                        #(int): Initial number of global epochs.
                 data_loader,                   #(dict): Dictionary of data loaders (contrains "train" and "dev" loaders).
                 model,                         #(dict): Dictionary of models (contains "generator" and "discriminator" models).
                 criterion,                     #(dict): Dictionary of criterions (contrains "stft" and "mse" criterions).
                 optimizer,                     #(dict): Dictionary of optimizers (contrains "generator" and "discriminator" optimizers).
                 scheduler,                     #(dict): Dictionary of schedulers (contrains "generator" and "discriminator" schedulers).
                 config,                        #(dict): Config dictionary loaded from yaml format configuration file.
                 device=torch.device("cpu"),    #(torch.deive): Pytorch device instance.
                 ):
        
        #initialzations:
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        self.writer = SummaryWriter(config["outdir"]) #to write the events while training asynchronously.
        self.finish_train = False # true when training is finished
        self.total_train_loss = defaultdict(float) # the stft loss function results on training phase 
        self.total_eval_loss = defaultdict(float) # the stft loss function results on eavaluating phase 

    def run(self):
        """Run training epoch by epoch."""
        self.tqdm = tqdm(initial=self.steps,
                         total=self.config["train_max_steps"],
                         desc="[train]") # progress bar of steps done from maximum number of steps

        while not self.finish_train: # loop till the training is finished
            self._train_epoch() #training of one epoch 

        self.tqdm.close() # close the progress bar when training is finished
        logging.info("Training is complete!, Please Be Happy :)!") 



    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint of comlpeted training results."""

        """
        Arguments:
            checkpoint_path (str): the path inwhich we save Checkpoint.
        """

        ####################################
        """
        we make a state dictionary that has the following archetecture:
            optimizer{state dictionary of the generator, state dictionary of the discriminator},
            scheduler{state dictionary of the generator, state dictionary of the discriminator},
            steps
            epochs
            model{state dictionary of the generator, state dictionary of the discriminator},
        """
        ####################################

        state_dict = { 
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }

        if self.config["distributed"]:# if we confiured distributed
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(), # get the state of the module 
                "discriminator": self.model["discriminator"].module.state_dict(),# get the state of the module
            }
        else:# if we confiured NOT distributed
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(), #get the state of the whole model
                "discriminator": self.model["discriminator"].state_dict(),#get the state of the whole model
            }

        if not os.path.exists(os.path.dirname(checkpoint_path)): #if the checkpoint_path is not found
            os.makedirs(os.path.dirname(checkpoint_path))   #make the path
        torch.save(state_dict, checkpoint_path)  # save the state dictionary in the checkpoint path


    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load a checkpoint to start from it."""
        
        """
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.
        """

        state_dict = torch.load(checkpoint_path, map_location="cpu") #load the checkpoint to the cpu
        if self.config["distributed"]: # if we confiured distributed
            self.model["generator"].module.load_state_dict(state_dict["model"]["generator"]) #load the state of the generator module of the model
            self.model["discriminator"].module.load_state_dict(state_dict["model"]["discriminator"]) #load the state of the discriminator module of the model
        else:# if we confiured NOT distributed
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])#load the state of the generator model
            self.model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])#load the state of the discriminator model
        if not load_only_params:
            self.steps = state_dict["steps"] #load number of steps
            self.epochs = state_dict["epochs"]#load number of epochs
            self.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])#load the optimizer generator state
            self.optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])#load the optimizer discriminator state
            state_dict["scheduler"]["generator"].update(**self.config["generator_scheduler_params"])#overwrite schedular generator argument parameters
            state_dict["scheduler"]["discriminator"].update(**self.config["discriminator_scheduler_params"])#overwrite schedular discriminator argument parameters
            self.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])#load the scheduler generator state
            self.scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])#load the scheduler discriminator state
