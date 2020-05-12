import argparse
import logging
import os

import kaldiio
import librosa
import numpy as np
import yaml
import soundfile as sf

from joblib import delayed
from joblib import Parallel
from tqdm import tqdm
from parallel_wavegan.datasets import AudioDataset
from parallel_wavegan.utils import write_hdf5




# calc log-Mel filterbank feature
def logmelfilterbank(audio,           # audio signal
                     sampling_rate,   # sampling_rate
                     fft_size=1024,   # length of windowed signal
                     hop_size=256,    # number of audio samples between adjacent STFT columns
                     win_length=None, # window length = fft_size
                     window="hann",   # window func type
                     num_mels=80,     # number of mels in filterbank
                     fmin=None,       # min frequency in filterbank
                     fmax=None,       # max frequency in filterbank
                     eps=1e-10,       # epsiln value to avoid infinity in log calculation 
                     ):


    # get short-time fourier transform of audio signal
    x_stft = librosa.stft(audio,n_fft=fft_size,hop_length=hop_size,
                            win_length=win_length,window=window,pad_mode="reflect")

    # get amplitude spectrogram
    spc = np.abs(x_stft).T 

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate,fft_size,num_mels,fmin,fmax)
    

    return np.log10(np.maximum(eps,np.dot(spc,mel_basis.T)))
    
    
def main():
    parser = argparse.ArgumentParser(description="Preprocess audio and extract features (see detail in parallel_wavegan/bin/preprocess.py ")
    parser.add_argument("--wav-scp","--scp",default=None,type=str,
                        help="kaldi-styke wav.scp file. you need to specify either scp or rootdir.")
    parser.add_argument("--segments",default=None,type=str,
                        help="kaldi-style segments file. if use you must specify both scp and segments.")
    parser.add_argument("--rootdir",default=None,type=str,
                        help="directory icluding wav files. you need to specify either scp or rootdir.")
    parser.add_argument("--dumpdir",type=str,required=True,
                        help="directory to dump feature files.")
    parser.add_argument("--config",type=str,required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--n_jobs", type=int, default=16,
                        help="number of parallel jobs. (default=16)")
    parser.add_argument("--verbose",type=int,default=1,
                        help="logging level. higher is more logging.")
    args = parser.parse_args()


    # setting logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN,format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning('skip DEBUG/INFO messages')


    # loading config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # checking arguments
    if (args.wav_scp is not None and args.rootdir is not None) or \
            (args.wav_scp is None and args.rootdir is None):
        raise ValueError("Please specify either --wav_scp or --rootdir")


   # get dataset
    if args.wav_scp is not None:
        dataset = kaldiio.ReadHelper(f"scp:{args.wav_scp}",
                                     segments=args.segments)
    else:
        dataset = AudioDataset(args.rootdir, "*.wav",
                               audio_load_fn=sf.read,
                               return_filename=True)


    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # define function for parallel processing
    def _process_single_file(data):
        # parse inputs
        if args.wav_scp is not None:
            utt_id, (fs, audio) = data
            audio = audio.astype(np.float32)
            audio /= (1 << (16 - 1))  # assume that wav is PCM 16 bit
        else:
            name, (audio, fs) = data
            utt_id = os.path.basename(name).replace(".wav", "")

        # check
        assert len(audio.shape) == 1, \
            f"{utt_id} seems to be multi-channel signal."
        assert fs == config["sampling_rate"], \
            f"{utt_id} seems to have a different sampling rate."
        assert np.abs(audio).max() <= 1.0, \
            f"{utt_id} seems to be different from 16 bit PCM."

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(audio,
                                            top_db=config["trim_threshold_in_db"],
                                            frame_length=config["trim_frame_size"],
                                            hop_length=config["trim_hop_size"])

        # extract feature
        mel = logmelfilterbank(audio, fs,
                               fft_size=config["fft_size"],
                               hop_size=config["hop_size"],
                               win_length=config["win_length"],
                               window=config["window"],
                               num_mels=config["num_mels"],
                               fmin=config["fmin"],
                               fmax=config["fmax"])

        # make sure the audio length and feature length are matched
        audio = np.pad(audio, (0, config["fft_size"]), mode="edge")
        audio = audio[:len(mel) * config["hop_size"]]
        assert len(mel) * config["hop_size"] == len(audio)

        # apply global gain
        if config["global_gain_scale"] > 0.0:
            audio *= config["global_gain_scale"]
            if np.abs(audio).max() > 1.0:
                logging.warn(f"{utt_id} causes clipping. "
                             f"it is better to re-consider global gain scale.")
                return

        # save
        if config["format"] == "hdf5":
            write_hdf5(os.path.join(args.dumpdir, f"{utt_id}.h5"), "wave", audio.astype(np.float32))
            write_hdf5(os.path.join(args.dumpdir, f"{utt_id}.h5"), "feats", mel.astype(np.float32))
        elif config["format"] == "npy":
            np.save(os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
                    audio.astype(np.float32), allow_pickle=False)
            np.save(os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
                    mel.astype(np.float32), allow_pickle=False)
        else:
            raise ValueError("support only hdf5 or npy format.")

    # process in parallel
    Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
        [delayed(_process_single_file)(data) for data in tqdm(dataset)])


if __name__ == "__main__":
    main()
