import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math

from .constants import *
from time import time

""" Initial code was from https://github.com/jongwook/onsets-and-frames, now reimplemented without numpy """

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length, hop_length, win_length=None, window='hann', padding=True):
        super(STFT, self).__init__()
        if win_length is None:
            win_length = filter_length

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_name = window
        self.padding = padding
        
        # Create window tensor
        if window == 'hann':
            window_tensor = torch.hann_window(win_length)
        else:
            # Default to hann if not specified
            window_tensor = torch.hann_window(win_length)
        
        # Register both window and forward_basis to match the original model state dict
        self.register_buffer('window', window_tensor)
        
        # Create forward basis (matching the original implementation's structure)
        n_fft = filter_length
        forward_basis = torch.empty(n_fft, n_fft, dtype=torch.complex64)
        for i in range(n_fft):
            for j in range(n_fft):
                angle = -2 * math.pi * i * j / n_fft
                forward_basis[i, j] = torch.complex(
                    torch.cos(torch.tensor(angle)), 
                    torch.sin(torch.tensor(angle))
                )
        self.register_buffer('forward_basis', forward_basis)
        
    def forward(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        # Pad if needed
        if self.padding:
            pad_length = int(self.filter_length // 2)
            input_data = F.pad(input_data, (pad_length, pad_length), mode='reflect')
        
        # Use torch's built-in STFT
        complex_stft = torch.stft(
            input_data, 
            n_fft=self.filter_length, 
            hop_length=self.hop_length, 
            win_length=self.win_length,
            window=self.window, 
            center=False, 
            return_complex=True
        )
        
        # Calculate magnitude
        magnitude = torch.abs(complex_stft)
        
        # Transpose to get the expected shape
        magnitude = magnitude.transpose(1, 2)
        
        return magnitude


class MelSpectrogram(torch.nn.Module):
    def __init__(self, n_mels, sample_rate, filter_length, hop_length,
                 win_length=None, mel_fmin=0.0, mel_fmax=None):
        super(MelSpectrogram, self).__init__()
        self.stft = STFT(filter_length, hop_length, win_length)
        
        if mel_fmax is None:
            mel_fmax = sample_rate // 2
        
        # Create PyTorch-based mel filter bank
        mel_basis = self._create_mel_filterbank(sample_rate, filter_length, n_mels, mel_fmin, mel_fmax)
        self.register_buffer('mel_basis', mel_basis)
    
    def _create_mel_filterbank(self, sr, n_fft, n_mels, fmin, fmax):
        """
        Creates a mel filterbank matrix using PyTorch
        """
        # Convert to mel scale
        m_min = 2595.0 * torch.log10(torch.tensor(1.0 + fmin / 700.0))
        m_max = 2595.0 * torch.log10(torch.tensor(1.0 + fmax / 700.0))
        
        # Create mel points equally spaced in mel scale
        m_points = torch.linspace(m_min, m_max, n_mels + 2)
        
        # Convert back to Hz
        f_points = 700.0 * (10.0**(m_points / 2595.0) - 1.0)
        
        # Convert Hz to FFT bins
        bins = torch.floor((n_fft + 1) * f_points / sr).long()
        
        # Create filterbank matrix
        fbank = torch.zeros(n_mels, n_fft // 2 + 1)
        
        for m in range(1, n_mels + 1):
            f_m_minus = bins[m - 1]
            f_m = bins[m]
            f_m_plus = bins[m + 1]
            
            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus:
                    fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            
            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m:
                    fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        
        return fbank

    def forward(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, T, n_mels)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)
        with torch.no_grad():
            # Use PyTorch's STFT implementation instead of librosa
            magnitudes = self.stft(y)
            
            # Convert to mel scale using the mel basis
            mel_output = torch.matmul(self.mel_basis, magnitudes)
            mel_output = torch.log(torch.clamp(mel_output, min=1e-5))
            return mel_output