"""
FIS Modules for Deep JSCC
File: fis_modules.py (NEW FILE - add to root directory)
"""

import torch
import torch.nn as nn
import numpy as np


class FIS_ImportanceAssessment(nn.Module):
    """
    FIS Layer 1: Assess importance of each spatial location

    Input: Feature maps (B, C, H, W)
    Output: Importance map (B, H, W) with values in [0, 1]
    """

    def __init__(self):
        super().__init__()

        # Try to use scikit-fuzzy, fallback to neural network
        try:
            import skfuzzy as fuzz
            from skfuzzy import control as ctrl
            self.use_fuzzy = True
            self._setup_fuzzy_system()
            print("FIS Layer 1: Using fuzzy inference system")
        except ImportError:
            print("Warning: scikit-fuzzy not installed. Using NN approximation.")
            self.use_fuzzy = False
            self._setup_nn_approximation()

    def _setup_fuzzy_system(self):
        """Setup fuzzy inference system"""
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl

        # Input variables
        self.magnitude = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'magnitude')
        self.variance = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'variance')
        self.gradient = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'gradient')

        # Output variable
        self.importance = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'importance')

        # Membership functions
        for var in [self.magnitude, self.variance, self.gradient]:
            var['low'] = fuzz.trimf(var.universe, [0, 0, 0.4])
            var['medium'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
            var['high'] = fuzz.trimf(var.universe, [0.6, 1, 1])

        self.importance['very_low'] = fuzz.trimf(self.importance.universe, [0, 0, 0.25])
        self.importance['low'] = fuzz.trimf(self.importance.universe, [0.15, 0.35, 0.5])
        self.importance['medium'] = fuzz.trimf(self.importance.universe, [0.4, 0.5, 0.6])
        self.importance['high'] = fuzz.trimf(self.importance.universe, [0.5, 0.7, 0.85])
        self.importance['very_high'] = fuzz.trimf(self.importance.universe, [0.75, 1, 1])

        # Fuzzy rules
        self.rules = [
            ctrl.Rule(self.magnitude['high'] & self.variance['high'],
                      self.importance['very_high']),
            ctrl.Rule(self.magnitude['high'] & self.variance['medium'],
                      self.importance['high']),
            ctrl.Rule(self.gradient['high'],
                      self.importance['high']),
            ctrl.Rule(self.magnitude['low'] & self.variance['low'],
                      self.importance['very_low']),
            ctrl.Rule(self.magnitude['medium'] & self.variance['medium'],
                      self.importance['medium']),
        ]

        self.ctrl_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.ctrl_system)

    def _setup_nn_approximation(self):
        """Fallback: NN to approximate FIS"""
        self.nn_approx = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        features: (B, C, H, W) from encoder
        Returns: (B, H, W) importance map
        """
        B, C, H, W = features.shape
        importance_map = torch.zeros(B, H, W, device=features.device)

        if self.use_fuzzy:
            # Use fuzzy inference
            features_np = features.cpu().detach().numpy()

            for b in range(B):
                for i in range(H):
                    for j in range(W):
                        f_ij = features_np[b, :, i, j]

                        # Compute features
                        mag = np.linalg.norm(f_ij) / np.sqrt(C)
                        var = np.var(f_ij)
                        grad = np.std(f_ij)

                        # Normalize
                        mag = np.clip(mag, 0, 1)
                        var = np.clip(var / (np.mean(f_ij ** 2) + 1e-8), 0, 1)
                        grad = np.clip(grad / (np.mean(np.abs(f_ij)) + 1e-8), 0, 1)

                        # FIS inference
                        try:
                            self.simulation.input['magnitude'] = float(mag)
                            self.simulation.input['variance'] = float(var)
                            self.simulation.input['gradient'] = float(grad)
                            self.simulation.compute()
                            importance = self.simulation.output['importance']
                        except:
                            importance = 0.4 * mag + 0.3 * var + 0.3 * grad

                        importance_map[b, i, j] = importance
        else:
            # Use NN approximation
            for b in range(B):
                for i in range(H):
                    for j in range(W):
                        f_ij = features[b, :, i, j]

                        mag = torch.norm(f_ij) / np.sqrt(C)
                        var = torch.var(f_ij)
                        grad = torch.std(f_ij)

                        mag = torch.clamp(mag, 0, 1)
                        var = torch.clamp(var, 0, 1)
                        grad = torch.clamp(grad, 0, 1)

                        input_vec = torch.stack([mag, var, grad]).unsqueeze(0)
                        importance = self.nn_approx(input_vec).squeeze()

                        importance_map[b, i, j] = importance

        return importance_map


class FIS_BitAllocation(nn.Module):
    """
    FIS Layer 2: Bit allocation with GLOBAL rate constraint
    """

    def __init__(self, b_min=1, b_max=8):
        super().__init__()
        self.b_min = b_min
        self.b_max = b_max

    def forward(self, importance_map, SNR_dB, target_rate):
        """
        importance_map: (B, H, W) in [0,1]
        SNR_dB: scalar (float)
        target_rate: (0,1]  --- GLOBAL rate constraint
        """

        # ===============================
        # 1. Normalize importance
        # ===============================
        I = importance_map
        I = (I - I.min()) / (I.max() - I.min() + 1e-8)

        # ===============================
        # 2. Normalize SNR (0–30 dB)
        # ===============================
        snr_norm = torch.clamp(
            torch.tensor(SNR_dB / 30.0, device=I.device),
            0.0, 1.0
        )

        # ===============================
        # 3. Soft fuzzy memberships
        # ===============================
        low_I  = torch.relu(0.5 - I)
        high_I = torch.relu(I - 0.5)

        low_SNR  = torch.relu(0.5 - snr_norm)
        high_SNR = torch.relu(snr_norm - 0.5)

        # ===============================
        # 4. Fuzzy rules
        # ===============================
        rule_low  = low_I * low_SNR
        rule_mid  = (low_I * high_SNR) + (high_I * low_SNR)
        rule_high = high_I * high_SNR

        # ===============================
        # 5. Raw bit scores (NO target_rate here!)
        # ===============================
        bits_raw = (
            rule_low  * self.b_min +
            rule_mid  * ((self.b_min + self.b_max) / 2.0) +
            rule_high * self.b_max
        )

        # ===============================
        # 6. GLOBAL rate constraint
        # ===============================
        B, H, W = bits_raw.shape
        total_positions = H * W

        target_total_bits = (
            target_rate * self.b_max * total_positions
        )

        scale = target_total_bits / (
            bits_raw.sum(dim=(1, 2), keepdim=True) + 1e-8
        )

        bits = bits_raw * scale

        # ===============================
        # 7. Quantize + clamp
        # ===============================
        bits = torch.clamp(
            bits.round(),
            min=self.b_min,
            max=self.b_max
        )

        return bits



class AdaptiveQuantizer(nn.Module):
    """
    Adaptive quantization based on bit allocation
    """

    def __init__(self):
        super().__init__()

    def forward(self, features, bit_allocation):
        """
        features: (B, C, H, W)
        bit_allocation: (B, H, W)

        Returns: (B, C, H, W) quantized features
        """
        B, C, H, W = features.shape
        features_quantized = features.clone()

        for b in range(B):
            for i in range(H):
                for j in range(W):
                    bits = bit_allocation[b, i, j].item()

                    # Quantize all channels at this location
                    f_ij = features[b, :, i, j]

                    # Uniform quantization
                    levels = 2 ** bits
                    f_min = f_ij.min()
                    f_max = f_ij.max()

                    if f_max - f_min > 1e-8:
                        f_norm = (f_ij - f_min) / (f_max - f_min)
                        f_quant = torch.round(f_norm * (levels - 1)) / (levels - 1)
                        f_dequant = f_quant * (f_max - f_min) + f_min
                        features_quantized[b, :, i, j] = f_dequant

        return features_quantized
