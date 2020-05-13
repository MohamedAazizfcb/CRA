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
from parallel_wavegan.optimizers.radam import RAdam #R_Adam optimizer.
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


    def _train_step(self, batch):
        """To train the model by one step."""
        
        x, y = batch # parsing the batch
        x = tuple([x_.to(self.device) for x_ in x]) 
        y = y.to(self.device)

        
        """      Generator      """
        
        # 1-calculate generator loss
        y_ = self.model["generator"](*x)
        y, y_ = y.squeeze(1), y_.squeeze(1)
        sc_loss, mag_loss = self.criterion["stft"](y_, y)
        gen_loss = sc_loss + mag_loss
        if self.steps > self.config["discriminator_train_start_steps"]:
            # keep compatibility
            gen_loss *= self.config.get("lambda_aux_after_introduce_adv_loss", 1.0)
            p_ = self.model["discriminator"](y_.unsqueeze(1))
            if not isinstance(p_, list):
                # for standard discriminator
                adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()
            else:
                # for multi-scale discriminator
                adv_loss = 0.0
                for i in range(len(p_)):
                    adv_loss += self.criterion["mse"](
                        p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
                adv_loss /= (i + 1)
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()

                # feature matching loss
                if self.config["use_feat_match_loss"]:
                    # no need to track gradients
                    with torch.no_grad():
                        p = self.model["discriminator"](y.unsqueeze(1))
                    fm_loss = 0.0
                    for i in range(len(p_)):
                        for j in range(len(p_[i]) - 1):
                            fm_loss += self.criterion["l1"](p_[i][j], p[i][j].detach())
                    fm_loss /= (i + 1) * (j + 1)
                    self.total_train_loss["train/feature_matching_loss"] += fm_loss.item()
                    adv_loss += self.config["lambda_feat_match"] * fm_loss

            gen_loss += self.config["lambda_adv"] * adv_loss

        self.total_train_loss["train/spectral_convergence_loss"] += sc_loss.item()
        self.total_train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()
        self.total_train_loss["train/generator_loss"] += gen_loss.item()

        # update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["generator_grad_norm"])
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            # calculate discriminator loss
            p = self.model["discriminator"](y.unsqueeze(1))
            p_ = self.model["discriminator"](y_.unsqueeze(1).detach())
            if not isinstance(p, list):
                # for standard discriminator
                real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
                fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
                dis_loss = real_loss + fake_loss
                self.total_train_loss["train/real_loss"] += real_loss.item()
                self.total_train_loss["train/fake_loss"] += fake_loss.item()
                self.total_train_loss["train/discriminator_loss"] += dis_loss.item()
            else:
                # for multi-scale discriminator
                real_loss = 0.0
                fake_loss = 0.0
                for i in range(len(p)):
                    real_loss += self.criterion["mse"](
                        p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                    fake_loss += self.criterion["mse"](
                        p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
                real_loss /= (i + 1)
                fake_loss /= (i + 1)
                dis_loss = real_loss + fake_loss
                self.total_train_loss["train/real_loss"] += real_loss.item()
                self.total_train_loss["train/fake_loss"] += fake_loss.item()
                self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"])
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                     f"({self.train_steps_per_epoch} steps per epoch).")

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x)
        p_ = self.model["discriminator"](y_)
        y, y_ = y.squeeze(1), y_.squeeze(1)
        sc_loss, mag_loss = self.criterion["stft"](y_, y)
        aux_loss = sc_loss + mag_loss
        if self.steps > self.config["discriminator_train_start_steps"]:
            # keep compatibility
            aux_loss *= self.config.get("lambda_aux_after_introduce_adv_loss", 1.0)
        if not isinstance(p_, list):
            # for standard discriminator
            adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
            gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss
        else:
            # for multi-scale discriminator
            adv_loss = 0.0
            for i in range(len(p_)):
                adv_loss += self.criterion["mse"](
                    p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
            adv_loss /= (i + 1)
            gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss

            # feature matching loss
            if self.config["use_feat_match_loss"]:
                p = self.model["discriminator"](y.unsqueeze(1))
                fm_loss = 0.0
                for i in range(len(p_)):
                    for j in range(len(p_[i]) - 1):
                        fm_loss += self.criterion["l1"](p_[i][j], p[i][j])
                fm_loss /= (i + 1) * (j + 1)
                self.total_eval_loss["eval/feature_matching_loss"] += fm_loss.item()
                gen_loss += self.config["lambda_adv"] * self.config["lambda_feat_match"] * fm_loss

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](y.unsqueeze(1))
        p_ = self.model["discriminator"](y_.unsqueeze(1))
        if not isinstance(p_, list):
            # for standard discriminator
            real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
            fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
            dis_loss = real_loss + fake_loss
        else:
            # for multi-scale discriminator
            real_loss = 0.0
            fake_loss = 0.0
            for i in range(len(p)):
                real_loss += self.criterion["mse"](
                    p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                fake_loss += self.criterion["mse"](
                    p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
            real_loss /= (i + 1)
            fake_loss /= (i + 1)
            dis_loss = real_loss + fake_loss

        # add to total eval loss
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
        self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.data_loader["dev"], desc="[eval]"), 1):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(f"(Steps: {self.steps}) Finished evaluation "
                     f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        # generate
        x_batch, y_batch = batch
        x_batch = tuple([x.to(self.device) for x in x_batch])
        y_batch = y_batch.to(self.device)
        y_batch_ = self.model["generator"](*x_batch)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(figname.replace(".png", "_ref.wav"), y,
                     self.config["sampling_rate"], "PCM_16")
            sf.write(figname.replace(".png", "_gen.wav"), y_,
                     self.config["sampling_rate"], "PCM_16")

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(self,
                 batch_max_steps=20480,
                 hop_size=256,
                 aux_context_window=2,
                 use_noise_input=False,
                 ):
        """Initialize customized collater for PyTorch DataLoader.
        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.
        """
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input

    def __call__(self, batch):
        """Convert into batch tensors.
        Args:
            batch (list): list of tuple of the pair of audio and features.
        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where T = (T' - 2 * aux_context_window) * hop_size
            Tensor: Target signal batch (B, 1, T).
        """
        # time resolution check
        y_batch, c_batch = [], []
        for idx in range(len(batch)):
            x, c = batch[idx]
            x, c = self._adjust_length(x, c)
            self._check_length(x, c, self.hop_size, 0)
            if len(c) - 2 * self.aux_context_window > self.batch_max_frames:
                # randomly pickup with the batch_max_steps length of the part
                interval_start = self.aux_context_window
                interval_end = len(c) - self.batch_max_frames - self.aux_context_window
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * self.hop_size
                y = x[start_step: start_step + self.batch_max_steps]
                c = c[start_frame - self.aux_context_window:
                      start_frame + self.aux_context_window + self.batch_max_frames]
                self._check_length(y, c, self.hop_size, self.aux_context_window)
            else:
                logging.warn(f"Removed short sample from batch (length={len(x)}).")
                continue
            y_batch += [y.astype(np.float32).reshape(-1, 1)]  # [(T, 1), (T, 1), ...]
            c_batch += [c.astype(np.float32)]  # [(T' C), (T' C), ...]

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = torch.FloatTensor(np.array(y_batch)).transpose(2, 1)  # (B, 1, T)
        c_batch = torch.FloatTensor(np.array(c_batch)).transpose(2, 1)  # (B, C, T')

        # make input noise signal batch tensor
        if self.use_noise_input:
            z_batch = torch.randn(y_batch.size())  # (B, 1, T)
            return (z_batch, c_batch), y_batch
        else:
            return (c_batch,), y_batch

    def _adjust_length(self, x, c):
        """Adjust the audio and feature lengths.
        NOTE that basically we assume that the length of x and c are adjusted
        in preprocessing stage, but if we use ESPnet processed features, this process
        will be needed because the length of x is not adjusted.
        """
        if len(x) < len(c) * self.hop_size:
            x = np.pad(x, (0, len(c) * self.hop_size - len(x)), mode="edge")
        return x, c

    @staticmethod
    def _check_length(x, c, hop_size, context_window):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == (len(c) - 2 * context_window) * hop_size


def main():
    """The main function that runs training process."""
    # initialize the argument parser
    parser = argparse.ArgumentParser(description="Train Parallel WaveGAN.") # just a description of the job that the parser is used to support.

    # Add arguments to the parser
        #first is name of the argument
        #default: The value produced if the argument is absent from the command line
        #type: The type to which the command-line argument should be converted
        #help: hint that appears when the user doesnot know what is this argument [-h]
        #required: Whether or not the command-line option may be omitted (optionals only)
        #nargs:The number of command-line arguments that should be consumed
            # "?" One argument will be consumed from the command line if possible, and produced as a single item. If no command-line argument is present, 
            # the value from default will be produced. 
            # Note that for optional arguments, there is an additional case - the option string is present but not followed by a command-line argument. In this case the value from const will be produced.

    parser.add_argument("--train-wav-scp", default=None, type=str,
                        help="kaldi-style wav.scp file for training. "
                             "you need to specify either train-*-scp or train-dumpdir.")

    parser.add_argument("--train-feats-scp", default=None, type=str,
                        help="kaldi-style feats.scp file for training. "
                             "you need to specify either train-*-scp or train-dumpdir.")

    parser.add_argument("--train-segments", default=None, type=str,
                        help="kaldi-style segments file for training.")

    parser.add_argument("--train-dumpdir", default=None, type=str,
                        help="directory including training data. "
                             "you need to specify either train-*-scp or train-dumpdir.")

    parser.add_argument("--dev-wav-scp", default=None, type=str,
                        help="kaldi-style wav.scp file for validation. "
                             "you need to specify either dev-*-scp or dev-dumpdir.")

    parser.add_argument("--dev-feats-scp", default=None, type=str,
                        help="kaldi-style feats.scp file for vaidation. "
                             "you need to specify either dev-*-scp or dev-dumpdir.")

    parser.add_argument("--dev-segments", default=None, type=str,
                        help="kaldi-style segments file for validation.")

    parser.add_argument("--dev-dumpdir", default=None, type=str,
                        help="directory including development data. "
                             "you need to specify either dev-*-scp or dev-dumpdir.")

    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save checkpoints.")

    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")

    parser.add_argument("--pretrain", default="", type=str, nargs="?",
                        help="checkpoint file path to load pretrained params. (default=\"\")")

    parser.add_argument("--resume", default="", type=str, nargs="?",
                        help="checkpoint file path to resume training. (default=\"\")")

    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")

    parser.add_argument("--rank", "--local_rank", default=0, type=int,
                        help="rank for distributed training. no need to explictly specify.")

    # parse all the input arguments 
    args = parser.parse_args()
    args.distributed = False

    if not torch.cuda.is_available(): #if gpu is not available 
        device = torch.device("cpu") #train on cpu
    else: #GPU
        device = torch.device("cuda")#train on gpu
        torch.backends.cudnn.benchmark = True # effective when using fixed size inputs (no conditional layers or layers inside loops),benchmark mode in cudnn,faster runtime
        torch.cuda.set_device(args.rank) # sets the default GPU for distributed training
        if "WORLD_SIZE" in os.environ:#determine max number of parallel processes (distributed)
            args.world_size = int(os.environ["WORLD_SIZE"]) #get the world size from the os
            args.distributed = args.world_size > 1 #set distributed if woldsize > 1 
        if args.distributed: 
            torch.distributed.init_process_group(backend="nccl", init_method="env://") #Use the NCCL backend for distributed GPU training (Rule of thumb)
                #NCCL:since it currently provides the best distributed GPU training performance, especially for multiprocess single-node or multi-node distributed training

    # suppress logging for distributed training
    if args.rank != 0: #if process is not p0
        sys.stdout = open(os.devnull, "w")#DEVNULL is Special value that can be used as the stdin, stdout or stderr argument to

    # set logger
    if args.verbose > 1: #if level of logging is heigher then 1
        logging.basicConfig( #configure the logging
            level=logging.DEBUG, stream=sys.stdout, #heigh logging level,detailed information, typically of interest only when diagnosing problems.
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s") #format includes Time,module,line#,level,and message.
    elif args.verbose > 0:#if level of logging is between 0,1
        logging.basicConfig(#configure the logging
            level=logging.INFO, stream=sys.stdout,#moderate logging level,Confirmation that things are working as expected.
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")#format includes Time,module,line#,level,and message.
    else:#if level of logging is 0
        logging.basicConfig(#configure the logging
            level=logging.WARN, stream=sys.stdout,#low logging level,An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’).
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")#format includes Time,module,line#,level,and message.
        logging.warning("Skip DEBUG/INFO messages")#tell the user that he will skip logging DEBUG/INFO messages by choosing this level.

    # check directory existence
    if not os.path.exists(args.outdir): #directory to save checkpoints
        os.makedirs(args.outdir)

    # check arguments
    if (args.train_feats_scp is not None and args.train_dumpdir is not None) or  (args.train_feats_scp is None and args.train_dumpdir is None):
            # if the user chooses both training data files (examples) or
            # the user doesnot choose any training data file
        raise ValueError("Please specify either --train-dumpdir or --train-*-scp.") #raise an error to tell the user to choose one training file
    if (args.dev_feats_scp is not None and args.dev_dumpdir is not None) or \
            (args.dev_feats_scp is None and args.dev_dumpdir is None):
            # if the user chooses both validatation data files (examples) or
            # the user doesnot choose any validatation data file
        raise ValueError("Please specify either --dev-dumpdir or --dev-*-scp.") #raise an error to tell the user to choose one validation data file

    # load config
    with open(args.config) as f:#open configuration file (yaml format)
        config = yaml.load(f, Loader=yaml.Loader) #load configuration file (yaml format to python object)
    # update config
    config.update(vars(args))#update arguments in configuration file
    config["version"] = parallel_wavegan.__version__  # add parallel wavegan version info
    # save config
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:#open outdir/config.yml
        yaml.dump(config, f, Dumper=yaml.Dumper) #dump function accepts a Python object and produces a YAML document.
    # add config info to the high level logger.
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    if config["remove_short_samples"]:#if configuration tells to remove short samples from training.
        mel_length_threshold = config["batch_max_steps"] // config["hop_size"] + \
            2 * config["generator_params"].get("aux_context_window", 0)#th of length = floor(batch_max_steps/hop_size) + 2 * (generator_params.aux_context_window)
    else:
        mel_length_threshold = None # No th.
    if args.train_wav_scp is None or args.dev_wav_scp is None: #if at least one of training or evaluating datasets = None
        if config["format"] == "hdf5":# format of data = hdf5
            audio_query, mel_query = "*.h5", "*.h5" # audio and text queries = "...".h5
            #lambda example:
            #x = lambda a, b: a * b
            #x(5, 6)-->x(a=5,b=6)=a*b=5*6=30
            audio_load_fn = lambda x: read_hdf5(x, "wave")  # The function to load data,NOQA
            mel_load_fn = lambda x: read_hdf5(x, "feats")  # The function to load data,NOQA
        elif config["format"] == "npy":# format of data = npy
            audio_query, mel_query = "*-wave.npy", "*-feats.npy" #audio query = "..."-wave.npy and text query = "..."-feats.h5
            audio_load_fn = np.load#The function to load data.
            mel_load_fn = np.load#The function to load data.
        else:#if any other data format
            raise ValueError("support only hdf5 or npy format.") #raise error to tell the user the data format is not supported.

    if args.train_dumpdir is not None: # if training ds is not None
        train_dataset = AudioMelDataset( # define the training dataset
            root_dir=args.train_dumpdir,#the directory of ds.
            audio_query=audio_query,#audio query according to format above.
            mel_query=mel_query,#mel query according to format above.
            audio_load_fn=audio_load_fn,#load the function that loads the audio data according to format above.
            mel_load_fn=mel_load_fn,#load the function that loads the mel data according to format above.
            mel_length_threshold=mel_length_threshold,#th to remove short samples -calculated above-.
            allow_cache=config.get("allow_cache", False),  # keep compatibility.
        )
    else:# if training ds is None
        train_dataset = AudioMelSCPDataset(# define the training dataset
            wav_scp=args.train_wav_scp,
            feats_scp=args.train_feats_scp,
            segments=args.train_segments, #segments of dataset
            mel_length_threshold=mel_length_threshold,#th to remove short samples -calculated above-.
            allow_cache=config.get("allow_cache", False),  # keep compatibility
        )
    logging.info(f"The number of training files = {len(train_dataset)}.") # add length of trainning data set to the logger.
    if args.dev_dumpdir is not None: #if evaluating ds is not None
        dev_dataset = AudioMelDataset( # define the evaluating dataset
            root_dir=args.dev_dumpdir,#the directory of ds.
            audio_query=audio_query,#audio query according to format above.
            mel_query=mel_query,#mel query according to format above.
            audio_load_fn=audio_load_fn,#load the function that loads the audio data according to format above.
            mel_load_fn=mel_load_fn,#load the function that loads the mel data according to format above.
            mel_length_threshold=mel_length_threshold,#th to remove short samples -calculated above-.
            allow_cache=config.get("allow_cache", False),  # keep compatibility
        )
    else:# if evaluating ds is None
        dev_dataset = AudioMelSCPDataset(
            wav_scp=args.dev_wav_scp,
            feats_scp=args.dev_feats_scp,
            segments=args.dev_segments,#segments of dataset
            mel_length_threshold=mel_length_threshold,#th to remove short samples -calculated above-.
            allow_cache=config.get("allow_cache", False),  # keep compatibility
        )
    logging.info(f"The number of development files = {len(dev_dataset)}.") # add length of evaluating data set to the logger.
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    } #define the whole dataset used which is divided into training and evaluating datasets
    # get data loader
    collater = Collater(
        batch_max_steps=config["batch_max_steps"],
        hop_size=config["hop_size"],
        # keep compatibility
        aux_context_window=config["generator_params"].get("aux_context_window", 0),
        # keep compatibility
        use_noise_input=config.get(
            "generator_type", "ParallelWaveGANGenerator") != "MelGANGenerator",
    )
    train_sampler, dev_sampler = None, None
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        dev_sampler = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=train_sampler,
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=dev_sampler,
            pin_memory=config["pin_memory"],
        ),
    }

    # define models and optimizers
    generator_class = getattr(
        parallel_wavegan.models,
        # keep compatibility
        config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    discriminator_class = getattr(
        parallel_wavegan.models,
        # keep compatibility
        config.get("discriminator_type", "ParallelWaveGANDiscriminator"),
    )
    model = {
        "generator": generator_class(
            **config["generator_params"]).to(device),
        "discriminator": discriminator_class(
            **config["discriminator_params"]).to(device),
    }
    criterion = {
        "stft": MultiResolutionSTFTLoss(
            **config["stft_loss_params"]).to(device),
        "mse": torch.nn.MSELoss().to(device),
    }
    if config.get("use_feat_match_loss", False):  # keep compatibility
        criterion["l1"] = torch.nn.L1Loss().to(device)
    optimizer = {
        "generator": RAdam(
            model["generator"].parameters(),
            **config["generator_optimizer_params"]),
        "discriminator": RAdam(
            model["discriminator"].parameters(),
            **config["discriminator_optimizer_params"]),
    }
    scheduler = {
        "generator": torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"]),
        "discriminator": torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer["discriminator"],
            **config["discriminator_scheduler_params"]),
    }
    if args.distributed:
        # wrap model for distributed training
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError("apex is not installed. please check https://github.com/NVIDIA/apex.")
        model["generator"] = DistributedDataParallel(model["generator"])
        model["discriminator"] = DistributedDataParallel(model["discriminator"])
    logging.info(model["generator"])
    logging.info(model["discriminator"])

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.pretrain}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl"))
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
