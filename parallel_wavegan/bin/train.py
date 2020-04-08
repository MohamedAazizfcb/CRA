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