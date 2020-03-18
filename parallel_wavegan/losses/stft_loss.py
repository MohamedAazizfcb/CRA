import torch
import torch.nn.functional as F


def stft(inp_sig, fft_size, hop_size, win_len, window):  # convert to magnitude spectrogram.
    """
    Arguments:
        inp_sig (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_len (int): Window length.
        window (str): Window function type.
    """
    # first perform Short-time Fourier transform on the input signal tensor
    sig_stft = torch.stft(inp_sig, fft_size, hop_size, win_len, window)
    real_part = sig_stft[..., 0]
    imag_part = sig_stft[..., 1]

    # clamp to avoid nan or inf
    clamped = torch.clamp(real_part ** 2 + imag_part ** 2, min=1e-7)

    # returns Magnitude spectrogram (B, #frames, fft_size // 2 + 1) --> (Tensor).
    return torch.sqrt(clamped).transpose(2, 1)


class Spectral_Convergence_Loss(torch.nn.Module):

    def __init__(self):
        # Initialize spectral convergence loss module.
        super(Spectral_Convergence_Loss, self).__init__()

    def forward_propagation(self, pred_mag_spect, gnd_mag_spect):  # Compute forward propagation
        """
        Arguments:
            pred_mag_spect (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            gnd_mag_spect (Tensor): Magnitude spectrogram of ground truth signal (B, #frames, #freq_bins).
        """
        # returns Spectral convergence loss value --> (Tensor).
        return torch.norm(gnd_mag_spect - pred_mag_spect, p="fro") / torch.norm(gnd_mag_spect, p="fro")


class Log_STFT_Magnitude_Loss(torch.nn.Module):

    def __init__(self):
        # Initialize los STFT magnitude loss module.
        super(Log_STFT_Magnitude_Loss, self).__init__()

    def forward_propagation(self, pred_mag_spect, gnd_mag_spect):  # Compute forward propagation
        """
        Arguments:
            pred_mag_spect (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            gnd_mag_spect (Tensor): Magnitude spectrogram of ground truth signal (B, #frames, #freq_bins).
        """
        # returns Log STFT magnitude loss value --> (Tensor).
        return F.l1_loss(torch.log(gnd_mag_spect), torch.log(pred_mag_spect))


class STFT_Loss(torch.nn.Module):

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFT_Loss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergence_loss = Spectral_Convergence_Loss()
        self.log_stft_magnitude_loss = Log_STFT_Magnitude_Loss()

    def forward(self, pred_Signal, gnd_signal):  # Compute forward propagation
        """
        Arguments:
            pred_Signal (Tensor): Predicted signal (B, T).
            gnd_signal (Tensor): Ground truth signal (B, T).
        """
        pred_mag_spect = stft(pred_Signal, self.fft_size, self.shift_size, self.win_length, self.window)
        gnd_mag_spect = stft(gnd_signal, self.fft_size, self.shift_size, self.win_length, self.window)

        spect_conv_loss = self.spectral_convergence_loss(pred_mag_spect, gnd_mag_spect)
        log_mag_loss = self.log_stft_magnitude_loss(pred_mag_spect, gnd_mag_spect)

        """
        returns : 
                1- Spectral convergence loss value --> (Tensor).
                2- Log STFT magnitude loss value --> (Tensor).
        """
        return spect_conv_loss, log_mag_loss


class Multi_Resolution_STFT_Loss(torch.nn.Module):

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window"):
        # Initialize Multi resolution STFT loss module.
        """
        Arguments:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(Multi_Resolution_STFT_Loss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFT_Loss(fs, ss, wl, window)]

    def forward(self, pred_signal, gnd_signal):  # Compute forward propagation
        """
        Arguments:
            pred_signal (Tensor): Predicted signal (B, T).
            gnd_signal (Tensor): Ground truth signal (B, T).
        """
        spect_conv_loss = 0.0
        log_mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(pred_signal, gnd_signal)
            spect_conv_loss += sc_l
            log_mag_loss += mag_l
        spect_conv_loss /= len(self.stft_losses)
        log_mag_loss /= len(self.stft_losses)
        """
        returns : 
                1- Multi resolution spectral convergence loss value --> (Tensor).
                2- Multi resolution log STFT magnitude loss value --> (Tensor).
        """
        return spect_conv_loss, log_mag_loss
