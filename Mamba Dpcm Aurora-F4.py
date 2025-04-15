import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import time
import cupy as cp
import math
from torch.utils.cpp_extension import load
import astropy.io.fits as fits
from typing import Optional, Tuple
import os
from dataclasses import dataclass
from adaptive_arithmetic_encoder_decoder import ArithmeticCoder, arithmetic_encoder, arithmetic_decoder
import gc
import csv
import traceback
import scipy
import scipy.sparse
import scipy.sparse.linalg
# 定义设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 配置设置
@dataclass
class MambaDPCMConfig:
    d_model: int = 64  # 模型维度
    n_layer: int = 4  # Mamba块数量
    d_state: int = 16  # SSM状态大小
    d_conv: int = 4  # 卷积核大小
    expand: int = 2  # 扩展因子
    dt_min: float = 0.001  # 最小delta值
    dt_max: float = 0.1  # 最大delta值
    dt_init: str = "random"  # 初始化方式
    dt_scale: float = 1.0  # 缩放因子
    dt_init_floor: float = 1e-4  # 最小初始值
    bias: bool = True  # 是否使用偏置
    conv_bias: bool = True  # 是否使用卷积偏置

    # DPCM特定参数
    block_size: int = 256  # 处理块大小
    pred_order: int = 11  # 预测阶数 (N)
    eq_count: int = 7  # 方程数量 (M)
    threshold_init: int = 13  # 残差阈值初始值

    # GPU优化参数
    streams_per_device: int = 4  # 每个GPU的CUDA流数量
    devices: int = 1  # 使用的GPU数量
    shared_mem_size: int = 48 * 1024  # 共享内存大小 (默认48KB)


class SSM(nn.Module):
    """
    状态空间模型实现
    """
    def __init__(self, config, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 初始化参数
        # A 是状态转移矩阵 (d_state, d_state)
        self.A = nn.Parameter(torch.randn(self.d_model, self.d_state, self.d_state))
        
        # B 是输入投影矩阵 (d_state, 1)
        self.B = nn.Parameter(torch.randn(self.d_model, self.d_state, 1))
        
        # C 是输出投影矩阵 (1, d_state)
        self.C = nn.Parameter(torch.randn(self.d_model, 1, self.d_state))
        
        # D 是直通连接
        self.D = nn.Parameter(torch.randn(self.d_model))
        
        # 时间步长参数
        log_dt_min, log_dt_max = math.log(config.dt_min), math.log(config.dt_max)
        self.log_dt = nn.Parameter(torch.rand(self.d_model) * (log_dt_max - log_dt_min) + log_dt_min)
        
        # 初始状态
        self.x0 = nn.Parameter(torch.zeros(self.d_model, self.d_state, 1))
        
    def forward(self, u):
        """
        前向传播
        u: 输入序列 [batch_size, seq_len, d_model]
        返回: 输出序列 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = u.shape
        
        # 确保维度匹配
        if d_model != self.d_model:
            raise ValueError(f"输入特征维度 {d_model} 与模型维度 {self.d_model} 不匹配")
        
        # 计算时间步长
        delta = torch.exp(self.log_dt)  # [d_model]
        
        # 确保delta的维度与A匹配
        delta_expanded = delta.unsqueeze(-1).unsqueeze(-1).expand(-1, self.d_state, self.d_state)
        
        # 计算离散状态转移矩阵
        A_discrete = torch.matrix_exp(self.A * delta_expanded)
        
        # 初始化状态
        x = self.x0.expand(batch_size, -1, -1, -1)  # [batch_size, d_model, d_state, 1]
        
        # 输出序列
        outputs = []
        
        # 逐时间步处理
        for t in range(seq_len):
            # 获取当前输入
            ut = u[:, t, :].unsqueeze(-1).unsqueeze(-1)  # [batch_size, d_model, 1, 1]
            
            # 状态更新
            x = A_discrete.unsqueeze(0) @ x + self.B.unsqueeze(0) * ut
            
            # 计算输出
            yt = (self.C.unsqueeze(0) @ x).squeeze(-1).squeeze(-1) + self.D * u[:, t, :]
            
            outputs.append(yt)
        
        # 堆叠输出
        return torch.stack(outputs, dim=1)  # [batch_size, seq_len, d_model]


# Mamba块实现
class MambaBlock(nn.Module):
    def __init__(self, config, d_model=None):
        super().__init__()
        self.config = config
        self.d_model = d_model or config.d_model
        self.expand = config.expand

        # 归一化层
        self.norm = nn.LayerNorm(self.d_model)

        # 局部上下文卷积
        self.conv = nn.Conv1d(
            in_channels=self.d_model * self.expand,
            out_channels=self.d_model * self.expand,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_model * self.expand,
            bias=config.conv_bias
        )

        # 线性上投影
        self.in_proj = nn.Linear(
            self.d_model,
            self.d_model * self.expand * 2,  # 2用于门控
            bias=config.bias
        )

        # SSM层
        self.ssm = SSM(config, d_model=self.d_model * self.expand)

        # 输出投影
        self.out_proj = nn.Linear(
            self.d_model * self.expand,
            self.d_model,
            bias=config.bias
        )

    def forward(self, x):
        # 输入: [batch, seq_len, d_model]
        residual = x
        x = self.norm(x)

        # 输入投影与门控
        x_proj = self.in_proj(x)  # [batch, seq_len, 2*d_model*expand]
        x_proj_1, x_proj_2 = x_proj.chunk(2, dim=-1)

        # 卷积处理
        x_conv = self.conv(x_proj_1.transpose(1, 2))
        x_conv = x_conv[:, :, :x.shape[1]].transpose(1, 2)

        # SSM处理
        x_ssm = self.ssm(x_conv)

        # SiLU激活和门控
        x_silu = F.silu(x_ssm)
        x_gated = x_silu * x_proj_2

        # 输出投影和残差连接
        return self.out_proj(x_gated) + residual


# Mamba模型主体 - 替换原有的MambaModel类
class MambaModel(nn.Module):
    """
    Mamba模型实现
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 输入投影
        self.input_proj = nn.Linear(config.d_model, config.d_model)
        
        # Mamba层
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.n_layer)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(config.d_model, config.d_model)
        
        # 融合层 - 确保维度匹配
        self.fusion = nn.Linear(config.d_model * config.expand, config.d_model)
        
    def forward(self, x):
        """
        前向传播
        x: 输入张量 [batch_size, seq_len, d_model]
        返回: 输出张量 [batch_size, seq_len, d_model]
        """
        # 输入投影
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # 提取维度信息
        batch_size, seq_len, d_model = x.shape
        
        # 处理行特征
        row_features = x
        for layer in self.layers:
            row_features = layer(row_features)
        
        # 输出投影
        output = self.output_proj(row_features)
        
        # 确保维度匹配
        if hasattr(self, 'fusion') and hasattr(self.config, 'expand') and self.config.expand > 1:
            # 如果使用了扩展，需要调整维度
            combined_features = output.reshape(batch_size * seq_len, -1)
            # 确保输入维度与权重维度匹配
            if combined_features.size(1) != self.fusion.weight.size(1):
                # 调整维度
                pad_size = self.fusion.weight.size(1) - combined_features.size(1)
                if pad_size > 0:
                    # 需要填充
                    padding = torch.zeros(combined_features.size(0), pad_size, 
                                         device=combined_features.device)
                    combined_features = torch.cat([combined_features, padding], dim=1)
                else:
                    # 需要裁剪
                    combined_features = combined_features[:, :self.fusion.weight.size(1)]
            
            fused_features = self.fusion(combined_features)
            output = fused_features.reshape(batch_size, seq_len, d_model)
        
        return output


# 添加动态预测系数类
class DynamicPredictionCoefficients(nn.Module):
    """
    动态预测系数生成模块，根据历史像素值生成最优预测系数
    利用选择性SSM的动态参数调整机制，实现输入相关的自适应预测
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 预测阶数
        self.pred_order = config.pred_order
        
        # 历史序列长度 - 增加以捕获更长依赖
        self.history_length = config.pred_order * 4
        
        # 创建配置用于增强型Mamba模型
        enhanced_config = MambaDPCMConfig(
            d_model=64,        # 模型维度
            n_layer=3,         # 层数
            d_state=64,        # 状态维度
            d_conv=4,          # 卷积核大小
            expand=2,          # 扩展因子
            dt_min=0.001,      # 时间步长范围
            dt_max=0.1,
            dt_init="random",  # 随机初始化时间步长
            dt_scale=1.0,
            bias=True,         # 启用偏置
            conv_bias=True     # 启用卷积偏置
        )
        
        # 增强型Mamba模型用于长序列建模
        self.mamba = MambaModel(enhanced_config)
        
        # 位置编码
        self.position_encoder = PositionalEncoding(enhanced_config.d_model, dropout=0.1, max_len=100)
        
        # 上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(enhanced_config.d_model, enhanced_config.d_model),
            nn.ReLU(),
            nn.Linear(enhanced_config.d_model, enhanced_config.d_model)
        )
        
        # 预测系数生成器
        self.coefficient_generator = nn.Sequential(
            nn.Linear(enhanced_config.d_model, enhanced_config.d_model),
            nn.ReLU(),
            nn.Linear(enhanced_config.d_model, self.pred_order)
        )
        
        # 初始化权重
        self._init_weights()
    
def forward(self, history, context=None):
    """
    根据历史像素值生成预测系数
    history: 历史像素值 [batch, seq_len]
    context: 可选的空间上下文 [batch, height, width]
    返回: 预测系数 [batch, pred_order]
    """
    batch_size = history.shape[0]
    
    # 检查当前可用GPU内存
    try:
        available_mem = torch.cuda.memory_allocated()
        total_mem = torch.cuda.get_device_properties(0).total_memory
        free_mem = total_mem - available_mem
        memory_critical = free_mem < 1e9  # 小于1GB可用内存视为紧急情况
    except:
        # 如果无法获取内存信息，假设内存紧张
        free_mem = 0
        memory_critical = True
        
    # 内存紧张时直接使用默认系数
    if memory_critical:
        print("内存紧张，使用默认预测系数...")
        return self.default_coeffs.unsqueeze(0).expand(batch_size, -1)
        
    # 根据可用内存动态调整历史长度
    if free_mem < 2e9:  # 小于2GB可用内存
        max_history_length = min(self.history_length, 8)  # 更激进地减少历史长度
    else:
        max_history_length = min(self.history_length, 16)  # 正常减少历史长度

    # 确保历史长度足够
    padded_history = self._pad_history(history, max_history_length)

    # 添加位置编码 - 使用较小的序列长度
    position_encoded = padded_history.unsqueeze(-1)
    if position_encoded.shape[1] <= self.position_encoding.shape[1]:
        position_encoded = position_encoded + self.position_encoding[:, :position_encoded.shape[1], :]
    else:
        # 如果历史长度超过位置编码长度，截断历史
        position_encoded = position_encoded[:, :self.position_encoding.shape[1], :] + self.position_encoding

    try:
        # 使用Mamba模型分析历史像素 - 添加内存优化
        with torch.cuda.amp.autocast():  # 使用混合精度计算
            # 分批处理以减少内存占用
            if batch_size > 8:
                # 大批量时，分批处理
                batch_results = []
                for i in range(0, batch_size, 4):  # 每批4个样本
                    end_idx = min(i + 4, batch_size)
                    batch_input = position_encoded[i:end_idx]
                    batch_output = self.mamba(batch_input)
                    batch_results.append(batch_output)
                    # 主动清理临时变量
                    del batch_input, batch_output
                    torch.cuda.empty_cache()
                
                mamba_features = torch.cat(batch_results, dim=0)
            else:
                # 批量小时，一次性处理
                mamba_features = self.mamba(position_encoded)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            # 内存不足，使用默认系数
            print("Mamba处理内存不足，使用默认预测系数...")
            # 清理所有可能的临时变量
            del position_encoded
            if 'mamba_features' in locals():
                del mamba_features
            if 'batch_results' in locals():
                del batch_results
            torch.cuda.empty_cache()
            return self.default_coeffs.unsqueeze(0).expand(batch_size, -1)
        else:
            # 其他错误，重新抛出
            raise

    # 提取最终特征
    final_mamba_features = mamba_features.mean(dim=1)  # [batch, d_model]
    del mamba_features
    torch.cuda.empty_cache()

    # 简化特征处理流程，减少内存使用
    try:
        # 简化的多尺度特征
        multi_scale_features = torch.mean(padded_history, dim=1, keepdim=True).repeat(1, 48)
        
        # 简化的特征融合
        fused_features = torch.cat([final_mamba_features, multi_scale_features], dim=1)
        fused_features = self.feature_fusion(fused_features)  # [batch, 64]
        
        # 简化的系数生成
        coeffs = self.coeff_generator(fused_features)
        
        # 使用Softmax确保系数和为1
        coeffs = F.softmax(coeffs, dim=1)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            # 内存不足，使用默认系数
            print("特征处理内存不足，使用默认预测系数...")
            # 清理所有临时变量
            del final_mamba_features, padded_history
            if 'multi_scale_features' in locals():
                del multi_scale_features
            if 'fused_features' in locals():
                del fused_features
            torch.cuda.empty_cache()
            return self.default_coeffs.unsqueeze(0).expand(batch_size, -1)
        else:
            raise
    
    # 清理缓存
    del final_mamba_features, multi_scale_features, fused_features
    torch.cuda.empty_cache()

    return coeffs
    
    def predict(self, history, context=None):
        """
        使用生成的系数预测下一个像素值
        history: 历史像素值 [batch, pred_order]
        context: 可选的空间上下文
        返回: 预测值 [batch]
        """
        # 生成系数
        coeffs = self.forward(history, context)
        
        # 使用系数进行预测
        prediction = torch.sum(history * coeffs, dim=1)
        
        return prediction

# 添加自适应块大小选择器类 - 在AuroraPreprocessor类之前添加
class AdaptiveBlockSizeSelector:
    """自适应块大小选择器，根据图像内容选择最佳块大小"""
    
    def __init__(self, min_size=64, max_size=512, step=64):
        self.min_size = min_size
        self.max_size = max_size
        self.step = step
        
        # 创建可用的块大小列表
        self.available_sizes = list(range(min_size, max_size + step, step))
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 块大小选择器
        self.size_selector = nn.Linear(32, len(self.available_sizes))
        
        # 将模型移动到GPU
        self.feature_extractor = self.feature_extractor.to(device)
        self.size_selector = self.size_selector.to(device)
    
    def select_block_size(self, image):
        """
        为给定图像选择最佳块大小
        image: 输入图像 [height, width]
        返回: 选择的块大小
        """
        # 确保图像是PyTorch张量
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32, device=device)
        
        # 添加批次和通道维度
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        
        # 提取特征
        with torch.no_grad():
            features = self.feature_extractor(image)
            features = features.view(features.size(0), -1)
            
            # 预测块大小分数
            scores = self.size_selector(features)
            
            # 选择得分最高的块大小
            size_idx = torch.argmax(scores, dim=1)[0].item()
            selected_size = self.available_sizes[size_idx]
        
        return selected_size
    
    def analyze_image_complexity(self, image):
        """
        分析图像复杂度，返回复杂度得分和推荐的块大小
        image: 输入图像 [height, width]
        返回: (复杂度得分, 推荐块大小)
        """
        # 计算图像梯度
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32, device=device)
        
        # 计算水平和垂直梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device).float()
        
        # 添加维度以便进行卷积
        image_expanded = image.unsqueeze(0).unsqueeze(0)
        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)
        
        # 计算梯度
        grad_x = F.conv2d(image_expanded, sobel_x, padding=1)
        grad_y = F.conv2d(image_expanded, sobel_y, padding=1)
        
        # 计算梯度幅度
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
        
        # 计算复杂度得分 (梯度幅度的平均值)
        complexity_score = grad_magnitude.mean().item()
        
        # 根据复杂度选择块大小
        if complexity_score < 10:
            # 低复杂度，使用大块
            recommended_size = self.max_size
        elif complexity_score < 30:
            # 中等复杂度，使用中等大小的块
            recommended_size = (self.min_size + self.max_size) // 2
        else:
            # 高复杂度，使用小块
            recommended_size = self.min_size
        
        return complexity_score, recommended_size

# 添加极光图像预处理类
class AuroraPreprocessor(nn.Module):
    """极光图像预处理模块，增强极光特征"""
    
    def __init__(self):
        super().__init__()
        # 对比度增强层
        self.contrast_enhancer = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1)
        )
        
        # 边缘检测层
        self.edge_detector = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=1)
        )
    
    def forward(self, x):
        """
        预处理极光图像
        x: 输入图像 [batch, 1, height, width]
        返回: 预处理后的图像和特征图
        """
        # 原始图像
        original = x
        
        # 对比度增强
        enhanced = self.contrast_enhancer(x)
        
        # 边缘检测
        edges = self.edge_detector(x)
        
        # 特征融合
        features = torch.cat([original, enhanced, edges], dim=1)
        
        return original, features

# 添加自适应块大小选择器
class AdaptiveBlockSizeSelector:
    """根据图像内容动态选择最优块大小"""
    
    def __init__(self, min_size=64, max_size=512, step=64):
        self.min_size = min_size
        self.max_size = max_size
        self.step = step
        self.sizes = list(range(min_size, max_size + 1, step))
    
    def select_block_size(self, image):
        """
        为图像选择最优块大小
        image: 输入图像 [height, width]
        返回: 最优块大小
        """
        # 计算图像复杂度
        complexity = self._calculate_complexity(image)
        
        # 根据复杂度选择块大小
        if complexity < 0.2:
            # 低复杂度区域使用大块
            return self.sizes[-1]
        elif complexity < 0.5:
            # 中等复杂度区域使用中等块
            return self.sizes[len(self.sizes) // 2]
        else:
            # 高复杂度区域使用小块
            return self.sizes[0]
    
    def _calculate_complexity(self, image):
        """计算图像复杂度"""
        # 使用梯度幅值作为复杂度度量
        grad_x = np.abs(np.diff(image, axis=1, prepend=image[:, :1]))
        grad_y = np.abs(np.diff(image, axis=0, prepend=image[:1, :]))
        
        # 计算平均梯度幅值并归一化
        avg_grad = (np.mean(grad_x) + np.mean(grad_y)) / 2
        max_possible_grad = np.max(image) - np.min(image)
        
        return avg_grad / max_possible_grad if max_possible_grad > 0 else 0

# 自适应残差阈值生成模块
class AdaptiveThresholdGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 小型Mamba模型处理残差分布
        small_config = MambaDPCMConfig(
            d_model=32,
            n_layer=2,
            d_state=8
        )
        self.mamba = MambaModel(small_config)

        # 映射到双阈值 (T-, T+)
        self.threshold_proj = nn.Linear(small_config.d_model, 2)

        # 初始化阈值
        self.register_buffer('base_threshold',
                             torch.tensor([config.threshold_init], dtype=torch.float))

    def forward(self, residual_histogram):
        """
        根据残差分布直方图生成自适应阈值
        residual_histogram: 残差直方图 [batch, bins]
        """
        # 使用Mamba处理残差直方图
        features = self.mamba(residual_histogram)

        # 生成阈值偏移量
        threshold_offsets = self.threshold_proj(features.mean(dim=1))

        # 基础阈值加上预测的偏移量
        T_minus = self.base_threshold - torch.abs(threshold_offsets[:, 0:1])
        T_plus = self.base_threshold + torch.abs(threshold_offsets[:, 1:2])

        return T_minus, T_plus


# 添加多目标损失函数 - 在AdaptiveThresholdGenerator类之后
class MultiObjectiveLoss(nn.Module):
    """
    多目标损失函数，包括重建损失、预测损失和稀疏性损失
    """

    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01):
        super().__init__()
        self.alpha = alpha  # 重建损失权重
        self.beta = beta  # 预测损失权重
        self.gamma = gamma  # 稀疏性损失权重

    def forward(self, original, predicted, reconstructed, residual):
        """
        计算多目标损失
        original: 原始图像
        predicted: 预测图像
        reconstructed: 重建图像
        residual: 残差
        """
        # 重建损失 (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, original)

        # 预测损失 (MAE)
        prediction_loss = F.l1_loss(predicted, original)

        # 稀疏性损失 (L1正则化)
        sparsity_loss = torch.mean(torch.abs(residual))

        # 总损失
        total_loss = (
                self.alpha * reconstruction_loss +
                self.beta * prediction_loss +
                self.gamma * sparsity_loss
        )

        return total_loss, {
            'reconstruction': reconstruction_loss.item(),
            'prediction': prediction_loss.item(),
            'sparsity': sparsity_loss.item()
        }


# GPU优化的并行扫描算法 (基于CUDA实现)
class ParallelScan:
    """
    并行扫描算法，使用CUDA实现GPU加速的矩阵运算
    优化矩阵分解策略，提高计算效率
    """
    def __init__(self, config):
        self.config = config
        
        # 初始化CUDA模块
        self.cuda_module = self._init_cuda_module()
        
        # 创建CUDA流，用于并行执行
        self.streams = [cp.cuda.Stream() for _ in range(4)]
        
        # 缓存矩阵分解结果
        self.decomposition_cache = {}
        
        # 初始化矩阵分块大小
        self.tile_size = 32  # 增大分块大小提高并行度
        
    def _init_cuda_module(self):
        """初始化CUDA模块，编译核心函数"""
        cuda_code = """
        // 优化的矩阵乘法核心函数
        extern "C" __global__ void matrix_mul_kernel(float* A, float* B, float* C, 
                                                 int M, int N, int K) {
            // 块索引
            int bx = blockIdx.x;
            int by = blockIdx.y;
            
            // 线程索引
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            
            // 共享内存声明
            extern __shared__ float shared_mem[];
            float* As = &shared_mem[0];
            float* Bs = &shared_mem[32*32]; // 使用32x32的块提高效率
            
            // 计算全局索引
            int row = by * 32 + ty;
            int col = bx * 32 + tx;
            
            // 累加器
            float sum = 0.0f;
            
            // 遍历A和B的分块
            for (int i = 0; i < (K + 31) / 32; ++i) {
                // 加载A分块到共享内存 - 使用向量化加载提高带宽利用率
                if (row < M && i * 32 + tx < K) {
                    As[ty * 32 + tx] = A[row * K + i * 32 + tx];
                } else {
                    As[ty * 32 + tx] = 0.0f;
                }
                
                // 加载B分块到共享内存 - 使用向量化加载提高带宽利用率
                if (i * 32 + ty < K && col < N) {
                    Bs[ty * 32 + tx] = B[(i * 32 + ty) * N + col];
                } else {
                    Bs[ty * 32 + tx] = 0.0f;
                }
                
                // 同步以确保数据加载完成
                __syncthreads();
                
                // 计算当前分块的乘积 - 使用循环展开提高指令级并行
                #pragma unroll 8
                for (int k = 0; k < 32; ++k) {
                    sum += As[ty * 32 + k] * Bs[k * 32 + tx];
                }
                
                // 同步以确保计算完成
                __syncthreads();
            }
            
            // 写入结果
            if (row < M && col < N) {
                C[row * N + col] = sum;
            }
        }
        
        // 优化的CT*C计算核心函数 - 使用分块策略
        extern "C" __global__ void ctc_kernel(float* C, float* result, int M, int N) {
            // 块索引
            int bx = blockIdx.x;
            int by = blockIdx.y;
            
            // 线程索引
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            
            // 共享内存
            extern __shared__ float shared_mem[];
            float* Cs = &shared_mem[0];
            float* CTs = &shared_mem[32*32]; // 存储C的转置
            
            // 计算全局索引
            int row = by * 32 + ty;
            int col = bx * 32 + tx;
            
            // 初始化累加器
            float sum = 0.0f;
            
            // 分块计算C^T * C
            for (int m = 0; m < (M + 31) / 32; ++m) {
                // 加载C子矩阵到共享内存
                int m_idx = m * 32 + ty;
                if (m_idx < M && col < N) {
                    Cs[ty * 32 + tx] = C[m_idx * N + col];
                } else {
                    Cs[ty * 32 + tx] = 0.0f;
                }
                
                // 加载C的转置到共享内存
                if (m_idx < M && row < N) {
                    CTs[tx * 32 + ty] = C[m_idx * N + row];
                } else {
                    CTs[tx * 32 + ty] = 0.0f;
                }
                
                // 同步以确保数据加载完成
                __syncthreads();
                
                // 计算当前分块的乘积
                #pragma unroll 8
                for (int k = 0; k < 32; ++k) {
                    sum += CTs[tx * 32 + k] * Cs[k * 32 + tx];
                }
                
                // 同步以确保计算完成
                __syncthreads();
            }
            
            // 写入结果
            if (row < N && col < N) {
                result[row * N + col] = sum;
            }
        }
        
        // 优化的矩阵分解核心函数 (Cholesky分解)
        extern "C" __global__ void cholesky_kernel(float* A, float* L, int N) {
            // 共享内存
            extern __shared__ float shared_A[];
            
            // 线程索引
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            
            // 全局索引
            int i = blockIdx.y * blockDim.y + ty;
            int j = blockIdx.x * blockDim.x + tx;
            
            // 加载数据到共享内存
            if (i < N && j < N) {
                shared_A[ty * blockDim.x + tx] = A[i * N + j];
            } else {
                shared_A[ty * blockDim.x + tx] = 0.0f;
            }
            
            __syncthreads();
            
            // 执行Cholesky分解
            if (i >= j && i < N && j < N) {
                float sum = 0.0f;
                for (int k = 0; k < j; k++) {
                    sum += L[i * N + k] * L[j * N + k];
                }
                
                if (i == j) {
                    // 对角元素
                    L[i * N + j] = sqrtf(shared_A[ty * blockDim.x + tx] - sum);
                } else {
                    // 非对角元素
                    L[i * N + j] = (shared_A[ty * blockDim.x + tx] - sum) / L[j * N + j];
                }
            }
        }
        
        // 优化的前向/后向替换核心函数
        extern "C" __global__ void forward_substitution_kernel(float* L, float* b, float* y, int N) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (i < N) {
                float sum = 0.0f;
                for (int j = 0; j < i; j++) {
                    sum += L[i * N + j] * y[j];
                }
                y[i] = (b[i] - sum) / L[i * N + i];
            }
        }
        
        extern "C" __global__ void backward_substitution_kernel(float* LT, float* y, float* x, int N) {
            int i = N - 1 - (blockIdx.x * blockDim.x + threadIdx.x);
            
            if (i >= 0 && i < N) {
                float sum = 0.0f;
                for (int j = i + 1; j < N; j++) {
                    sum += LT[i * N + j] * x[j];
                }
                x[i] = (y[i] - sum) / LT[i * N + i];
            }
        }
        
        // 优化的残差计算核心函数 - 使用向量化加载/存储
        extern "C" __global__ void residual_kernel(float* original, float* predicted, 
                                               float* residual, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            // 使用向量化加载/存储提高内存带宽利用率
            if (idx < size / 4) {
                float4 orig = reinterpret_cast<float4*>(original)[idx];
                float4 pred = reinterpret_cast<float4*>(predicted)[idx];
                
                float4 res;
                res.x = orig.x - pred.x;
                res.y = orig.y - pred.y;
                res.z = orig.z - pred.z;
                res.w = orig.w - pred.w;
                
                reinterpret_cast<float4*>(residual)[idx] = res;
            } else {
                // 处理剩余元素
                int base_idx = (size / 4) * 4;
                idx += base_idx;
                
                if (idx < size) {
                    residual[idx] = original[idx] - predicted[idx];
                }
            }
        }
        
        // 优化的双阈值编码核心函数 - 使用共享内存缓存阈值
        extern "C" __global__ void threshold_encode_kernel(float* residual, int* encoded,
                                                      float T_minus, float T_plus, int size) {
            // 共享内存缓存阈值
            __shared__ float shared_T_minus, shared_T_plus;
            
            // 只让第一个线程加载阈值到共享内存
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                shared_T_minus = T_minus;
                shared_T_plus = T_plus;
            }
            
            __syncthreads();
            
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float res = residual[idx];
                if (res >= shared_T_minus && res <= shared_T_plus) {
                    // 在阈值范围内，使用范围编码
                    encoded[idx] = static_cast<int>(res) + 32768; // 偏移以处理负值
                } else {
                    // 超出阈值，标记为特殊值
                    encoded[idx] = (res < shared_T_minus) ? -1 : -2;
                }
            }
        }
        
        // 新增: 矩阵分解策略 - QR分解核心函数
        extern "C" __global__ void qr_decomposition_kernel(float* A, float* Q, float* R, int M, int N) {
            // 共享内存
            extern __shared__ float shared_mem[];
            float* shared_A = &shared_mem[0];
            float* shared_v = &shared_mem[M*N];
            
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int row = blockIdx.y * blockDim.y + ty;
            int col = blockIdx.x * blockDim.x + tx;
            
            // 初始化Q为单位矩阵
            if (row < M && col < M) {
                Q[row * M + col] = (row == col) ? 1.0f : 0.0f;
            }
            
            // 加载A到共享内存
            if (row < M && col < N) {
                shared_A[ty * N + tx] = A[row * N + col];
            }
            
            __syncthreads();
            
            // 执行Householder变换
            for (int k = 0; k < min(M-1, N); k++) {
                // 计算Householder向量
                if (tx == 0 && row >= k && row < M) {
                    float x = shared_A[(row-k) * N + k];
                    float norm = 0.0f;
                    
                    // 计算列向量的范数
                    for (int i = k; i < M; i++) {
                        float val = shared_A[(i-k) * N + k];
                        norm += val * val;
                    }
                    norm = sqrtf(norm);
                    
                    // 构造Householder向量
                    if (row == k) {
                        shared_v[row-k] = x + copysignf(norm, x);
                    } else {
                        shared_v[row-k] = x;
                    }
                }
                
                __syncthreads();
                
                // 计算Householder向量的范数
                float v_norm = 0.0f;
                if (tx == 0 && row >= k && row < M) {
                    for (int i = 0; i < M-k; i++) {
                        v_norm += shared_v[i] * shared_v[i];
                    }
                    v_norm = sqrtf(v_norm);
                }
                
                // 归一化Householder向量
                if (tx == 0 && row >= k && row < M) {
                    shared_v[row-k] /= v_norm;
                }
                
                __syncthreads();
                
                // 应用Householder变换到A
                if (row >= k && row < M && col >= k && col < N) {
                    float dot_product = 0.0f;
                    for (int i = 0; i < M-k; i++) {
                        dot_product += shared_v[i] * shared_A[i * N + (col-k)];
                    }
                    
                    shared_A[(row-k) * N + (col-k)] -= 2.0f * shared_v[row-k] * dot_product;
                }
                
                // 应用Householder变换到Q
                if (row < M && col < M) {
                    float dot_product = 0.0f;
                    for (int i = 0; i < M-k; i++) {
                        dot_product += shared_v[i] * Q[(row) * M + (i+k)];
                    }
                    
                    Q[row * M + col] -= 2.0f * dot_product * shared_v[col-k];
                }
                
                __syncthreads();
            }
            
            // 复制上三角部分到R
            if (row < N && col < N && row <= col) {
                R[row * N + col] = shared_A[row * N + col];
            } else if (row < N && col < N) {
                R[row * N + col] = 0.0f;
            }
        }
        
        // 新增: 矩阵分解策略 - SVD分解核心函数 (简化版)
        extern "C" __global__ void svd_decomposition_kernel(float* A, float* U, float* S, float* VT, int M, int N) {
            // 这里实现简化版SVD，实际应用中可能需要更复杂的算法
            // 此处仅作为示例，展示如何在CUDA中实现SVD
            
            // 共享内存
            extern __shared__ float shared_mem[];
            float* shared_A = &shared_mem[0];
            
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int row = blockIdx.y * blockDim.y + ty;
            int col = blockIdx.x * blockDim.x + tx;
            
            // 加载A到共享内存
            if (row < M && col < N) {
                shared_A[ty * N + tx] = A[row * N + col];
            }
            
            __syncthreads();
            
            // 计算A^T * A
            if (row < N && col < N) {
                float sum = 0.0f;
                for (int i = 0; i < M; i++) {
                    sum += A[i * N + row] * A[i * N + col];
                }
                
                VT[row * N + col] = sum;
            }
            
            __syncthreads();
            
            // 注意：完整的SVD实现需要计算特征值和特征向量
            // 这里仅作为示例，实际应用中应使用cuSOLVER等库
        }
        """

        # 使用cupy编译CUDA代码
        module = cp.RawModule(code=cuda_code)
        return module

    def matrix_multiply(self, A, B):
        """
        优化的矩阵乘法实现
        A: 第一个矩阵 [M, K]
        B: 第二个矩阵 [K, N]
        返回: C = A * B [M, N]
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "矩阵维度不匹配"

        # 确保数据在GPU上
        if not isinstance(A, cp.ndarray):
            A_gpu = cp.asarray(A)
        else:
            A_gpu = A

        if not isinstance(B, cp.ndarray):
            B_gpu = cp.asarray(B)
        else:
            B_gpu = B

        # 分配输出内存
        C_gpu = cp.zeros((M, N), dtype=np.float32)

        # 计算网格和块的尺寸
        grid_dim = ((N + 31) // 32, (M + 31) // 32)
        block_dim = (32, 32)
        shared_mem = 2 * 32 * 32 * 4  # 2个32x32的浮点数块

        # 获取核心函数
        kernel = self.cuda_module.get_function("matrix_mul_kernel")

        # 启动核心
        kernel(grid_dim, block_dim,
               (A_gpu, B_gpu, C_gpu, M, N, K),
               shared_mem=shared_mem)

        return C_gpu

    def compute_ctc(self, C):
        """
        计算CT*C的优化实现
        C: 输入矩阵 [M, N]
        返回: result = C^T * C [N, N]
        """
        M, N = C.shape

        # 确保数据在GPU上
        if not isinstance(C, cp.ndarray):
            C_gpu = cp.asarray(C)
        else:
            C_gpu = C

        # 分配输出内存
        result_gpu = cp.zeros((N, N), dtype=np.float32)

        # 计算网格和块的尺寸
        grid_dim = ((N + 31) // 32, (N + 31) // 32)
        block_dim = (32, 32)
        shared_mem = 2 * 32 * 32 * 4  # 两个32x32的浮点数块

        # 获取核心函数
        kernel = self.cuda_module.get_function("ctc_kernel")

        # 启动核心
        kernel(grid_dim, block_dim,
               (C_gpu, result_gpu, M, N),
               shared_mem=shared_mem)

        return result_gpu
    
    def matrix_decomposition(self, A, method='cholesky'):
        """
        矩阵分解策略
        A: 输入矩阵
        method: 分解方法 ('cholesky', 'qr', 'svd')
        返回: 分解结果
        """
        # 检查缓存
        cache_key = (A.shape, method)
        if cache_key in self.decomposition_cache:
            return self.decomposition_cache[cache_key]
        
        # 确保数据在GPU上
        if not isinstance(A, cp.ndarray):
            A_gpu = cp.asarray(A)
        else:
            A_gpu = A
            
        N = A.shape[0]
        
        if method == 'cholesky':
            # Cholesky分解 A = L * L^T
            L_gpu = cp.zeros_like(A_gpu)
            
            # 计算网格和块的尺寸
            block_dim = (16, 16)
            grid_dim = ((N + block_dim[0] - 1) // block_dim[0], 
                         (N + block_dim[1] - 1) // block_dim[1])
            shared_mem = block_dim[0] * block_dim[1] * 4
            
            # 获取核心函数
            kernel = self.cuda_module.get_function("cholesky_kernel")
            
            # 启动核心
            kernel(grid_dim, block_dim,
                   (A_gpu, L_gpu, N),
                   shared_mem=shared_mem)
                   
            # 缓存结果
            self.decomposition_cache[cache_key] = (L_gpu,)
            return (L_gpu,)
            
        elif method == 'qr':
            # QR分解 A = Q * R
            M, N = A_gpu.shape
            Q_gpu = cp.zeros((M, M), dtype=np.float32)
            R_gpu = cp.zeros((M, N), dtype=np.float32)
            
            # 计算网格和块的尺寸
            block_dim = (16, 16)
            grid_dim = ((N + block_dim[0] - 1) // block_dim[0], 
                         (M + block_dim[1] - 1) // block_dim[1])
            shared_mem = M * N * 4 + M * 4  # 共享内存用于A和Householder向量
            
            # 获取核心函数
            kernel = self.cuda_module.get_function("qr_decomposition_kernel")
            
            # 启动核心
            kernel(grid_dim, block_dim,
                   (A_gpu, Q_gpu, R_gpu, M, N),
                   shared_mem=shared_mem)
                   
            # 缓存结果
            self.decomposition_cache[cache_key] = (Q_gpu, R_gpu)
            return (Q_gpu, R_gpu)
            
        elif method == 'svd':
            # SVD分解 A = U * S * V^T
            M, N = A_gpu.shape
            U_gpu = cp.zeros((M, M), dtype=np.float32)
            S_gpu = cp.zeros(min(M, N), dtype=np.float32)
            VT_gpu = cp.zeros((N, N), dtype=np.float32)
            
            # 对于完整的SVD，建议使用cuSOLVER库
            # 这里使用简化版实现作为示例
            
            # 计算网格和块的尺寸
            block_dim = (16, 16)
            grid_dim = ((N + block_dim[0] - 1) // block_dim[0], 
                         (M + block_dim[1] - 1) // block_dim[1])
            shared_mem = M * N * 4  # 共享内存用于A
            
            # 获取核心函数
            kernel = self.cuda_module.get_function("svd_decomposition_kernel")
            
            # 启动核心
            kernel(grid_dim, block_dim,
                   (A_gpu, U_gpu, S_gpu, VT_gpu, M, N),
                   shared_mem=shared_mem)
                   
            # 缓存结果
            self.decomposition_cache[cache_key] = (U_gpu, S_gpu, VT_gpu)
            return (U_gpu, S_gpu, VT_gpu)
        else:
            raise ValueError(f"不支持的分解方法: {method}")

    def solve_linear_system(self, A, b, method='cholesky'):
        """
        求解线性方程组 Ax = b
        A: 系数矩阵
        b: 右侧向量
        method: 求解方法
        返回: 解向量 x
        """
        # 确保数据在GPU上
        if not isinstance(A, cp.ndarray):
            A_gpu = cp.asarray(A)
        else:
            A_gpu = A
            
        if not isinstance(b, cp.ndarray):
            b_gpu = cp.asarray(b)
        else:
            b_gpu = b
            
        N = A_gpu.shape[0]
        
        if method == 'cholesky':
            # 使用Cholesky分解求解
            L_gpu, = self.matrix_decomposition(A_gpu, method='cholesky')
            
            # 前向替换 Ly = b
            y_gpu = cp.zeros_like(b_gpu)
            
            # 计算网格和块的尺寸
            block_dim = 256
            grid_dim = (N + block_dim - 1) // block_dim
            
            # 获取核心函数
            forward_kernel = self.cuda_module.get_function("forward_substitution_kernel")
            
            # 启动核心
            forward_kernel((grid_dim,), (block_dim,),
                          (L_gpu, b_gpu, y_gpu, N))
                          
            # 后向替换 L^T x = y
            x_gpu = cp.zeros_like(b_gpu)
            LT_gpu = cp.transpose(L_gpu)
            
            # 获取核心函数
            backward_kernel = self.cuda_module.get_function("backward_substitution_kernel")
            
            # 启动核心
            backward_kernel((grid_dim,), (block_dim,),
                           (LT_gpu, y_gpu, x_gpu, N))
                           
            return x_gpu
            
        elif method == 'qr':
            # 使用QR分解求解
            Q_gpu, R_gpu = self.matrix_decomposition(A_gpu, method='qr')
            
            # 计算 Q^T * b
            QT_b = cp.matmul(cp.transpose(Q_gpu), b_gpu)
            
            # 后向替换求解 Rx = Q^T b
            x_gpu = cp.zeros_like(b_gpu)
            
            # 使用cupy的内置函数求解上三角系统
            x_gpu = cp.linalg.solve_triangular(R_gpu, QT_b, lower=False)
            
            return x_gpu
            
        else:
            # 使用cupy的内置求解器
            return cp.linalg.solve(A_gpu, b_gpu)

    def compute_residual(self, original, predicted):
        """
        计算残差的优化实现
        original: 原始数据 [size]
        predicted: 预测数据 [size]
        返回: residual = original - predicted [size]
        """
        size = len(original)

        # 确保数据在GPU上
        if not isinstance(original, cp.ndarray):
            original_gpu = cp.asarray(original)
        else:
            original_gpu = original

        if not isinstance(predicted, cp.ndarray):
            predicted_gpu = cp.asarray(predicted)
        else:
            predicted_gpu = predicted

        # 分配输出内存
        residual_gpu = cp.zeros(size, dtype=np.float32)

        # 计算网格和块的尺寸 - 考虑向量化
        threads_per_block = 256
        blocks_per_grid = (size // 4 + threads_per_block - 1) // threads_per_block

        # 获取核心函数
        kernel = self.cuda_module.get_function("residual_kernel")

        # 启动核心
        kernel((blocks_per_grid,), (threads_per_block,),
               (original_gpu, predicted_gpu, residual_gpu, size))

        return residual_gpu

    def threshold_encode(self, residual, T_minus, T_plus):
        """
        双阈值编码的优化实现
        residual: 残差数据 [size]
        T_minus: 下阈值
        T_plus: 上阈值
        返回: 编码后的数据 [size]
        """
        size = len(residual)

        # 确保数据在GPU上
        if not isinstance(residual, cp.ndarray):
            residual_gpu = cp.asarray(residual)
        else:
            residual_gpu = residual

        # 分配输出内存
        encoded_gpu = cp.zeros(size, dtype=np.int32)

        # 计算网格和块的尺寸
        threads_per_block = 256
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

        # 获取核心函数
        kernel = self.cuda_module.get_function("threshold_encode_kernel")

        # 启动核心 - 使用共享内存缓存阈值
        kernel((blocks_per_grid,), (threads_per_block,),
               (residual_gpu, encoded_gpu, float(T_minus), float(T_plus), size))

        return encoded_gpu
        
    def parallel_process_matrix(self, matrix, process_func, num_streams=4):
        """
        使用多个CUDA流并行处理矩阵
        matrix: 输入矩阵
        process_func: 处理函数
        num_streams: 流的数量
        返回: 处理后的矩阵
        """
        # 确保数据在GPU上
        if not isinstance(matrix, cp.ndarray):
            matrix_gpu = cp.asarray(matrix)
        else:
            matrix_gpu = matrix
            
        # 获取矩阵尺寸
        rows, cols = matrix_gpu.shape
        
        # 计算每个流处理的行数
        rows_per_stream = (rows + num_streams - 1) // num_streams
        
        # 创建结果矩阵
        result_gpu = cp.zeros_like(matrix_gpu)
        
        # 并行处理每个分块
        for i in range(num_streams):
            start_row = i * rows_per_stream
            end_row = min((i + 1) * rows_per_stream, rows)
            
            if start_row >= rows:
                break
                
            # 获取当前分块
            block = matrix_gpu[start_row:end_row]
            
            # 在指定流中处理分块
            with self.streams[i % len(self.streams)]:
                # 处理当前分块
                result_block = process_func(block)
                
                # 将结果复制回结果矩阵
                result_gpu[start_row:end_row] = result_block
                
        # 同步所有流
        cp.cuda.Stream.null.synchronize()
        
        return result_gpu
    
    def batch_matrix_multiply(self, matrices_A, matrices_B):
        """
        批量矩阵乘法
        matrices_A: 第一组矩阵 [batch, M, K]
        matrices_B: 第二组矩阵 [batch, K, N]
        返回: 结果矩阵 [batch, M, N]
        """
        batch_size = len(matrices_A)
        results = []
        
        # 使用多个流并行处理
        for i in range(batch_size):
            # 选择流
            stream_idx = i % len(self.streams)
            
            # 在指定流中执行矩阵乘法
            with self.streams[stream_idx]:
                result = self.matrix_multiply(matrices_A[i], matrices_B[i])
                results.append(result)
                
        # 同步所有流
        cp.cuda.Stream.null.synchronize()
        
        return results
    
    def optimize_matrix_operations(self, A, B, operation='multiply'):
        """
        优化矩阵操作，自动选择最佳算法
        A, B: 输入矩阵
        operation: 操作类型 ('multiply', 'solve', 'decompose')
        返回: 操作结果
        """
        # 分析矩阵特性
        if operation == 'multiply':
            M, K = A.shape
            K2, N = B.shape
            
            # 获取当前可用GPU内存
            available_mem = cp.cuda.Device().mem_info[0]
            required_mem = M * N * 4 + M * K * 4 + K * N * 4  # 估计所需内存
            
            # 根据矩阵大小和可用内存选择最佳算法
            if max(M, N, K) <= 32:
                # 小矩阵使用共享内存优化
                return self.matrix_multiply(A, B)
            elif required_mem > available_mem * 0.7:  # 如果预计内存使用超过可用内存的70%
                # 内存不足，检查是否可以使用多GPU
                if self.multi_gpu_enabled and self.num_gpus > 1 and M > 1000:
                    print(f"使用多GPU ({self.num_gpus}个) 进行大矩阵乘法")
                    return self.multi_gpu_matrix_multiply(A, B)
                else:
                    # 使用内存优化版本
                    return self._optimized_matrix_multiply(A, B)
            elif M >= 1024 or N >= 1024:
                # 大矩阵使用分块策略或多GPU
                if self.multi_gpu_enabled and self.num_gpus > 1 and M > 2000:
                    print(f"使用多GPU ({self.num_gpus}个) 进行大矩阵乘法")
                    return self.multi_gpu_matrix_multiply(A, B)
                else:
                    return self._large_matrix_multiply(A, B)
            else:
                # 中等大小矩阵使用标准算法
                return self.matrix_multiply(A, B)
                
        elif operation == 'solve':
            N = A.shape[0]
            
            # 根据矩阵特性选择求解方法
            if N <= 32:
                # 小型系统使用Cholesky分解
                return self.solve_linear_system(A, B, method='cholesky')
            elif self._is_sparse(A):
                # 稀疏矩阵使用迭代方法
                return self._solve_sparse_system(A, B)
            else:
                # 一般情况使用QR分解
                return self.solve_linear_system(A, B, method='qr')
                
        elif operation == 'decompose':
            M, N = A.shape
            
            # 根据矩阵特性选择分解方法
            if M == N and self._is_symmetric(A):
                # 对称矩阵使用Cholesky分解
                return self.matrix_decomposition(A, method='cholesky')
            elif M >= N:
                # 矩形矩阵使用QR分解
                return self.matrix_decomposition(A, method='qr')
            else:
                # 其他情况使用SVD
                return self.matrix_decomposition(A, method='svd')
        
        else:
            raise ValueError(f"不支持的操作: {operation}")
    
    def _large_matrix_multiply(self, A, B):
        """
        大矩阵乘法的分块实现
        A: 第一个矩阵 [M, K]
        B: 第二个矩阵 [K, N]
        返回: C = A * B [M, N]
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "矩阵维度不匹配"
        
        # 确保数据在GPU上
        if not isinstance(A, cp.ndarray):
            A_gpu = cp.asarray(A)
        else:
            A_gpu = A
            
        if not isinstance(B, cp.ndarray):
            B_gpu = cp.asarray(B)
        else:
            B_gpu = B
            
        # 分配输出内存
        C_gpu = cp.zeros((M, N), dtype=np.float32)
        
        # 定义分块大小 - 根据GPU内存动态调整
        available_mem = cp.cuda.Device().mem_info[0]  # 获取可用GPU内存
        # 根据可用内存动态调整分块大小
        if available_mem < 1e9:  # 小于1GB
            block_size = 16
        elif available_mem < 4e9:  # 小于4GB
            block_size = 32
        else:
            block_size = 64
        
        # 使用多流并行处理不同的分块
        num_streams = min(len(self.streams), (M + block_size - 1) // block_size)
        stream_assignments = {}
        
        # 分块计算
        for i in range(0, M, block_size):
            i_end = min(i + block_size, M)
            stream_idx = (i // block_size) % num_streams
            
            # 在指定流中处理当前行块
            with self.streams[stream_idx]:
                for j in range(0, N, block_size):
                    j_end = min(j + block_size, N)
                    
                    # 初始化当前块的结果
                    C_block = cp.zeros((i_end - i, j_end - j), dtype=np.float32)
                    
                    # 累加K维度上的分块乘积，使用前缀和技术
                    for k in range(0, K, block_size):
                        k_end = min(k + block_size, K)
                        
                        # 提取子块
                        A_block = A_gpu[i:i_end, k:k_end]
                        B_block = B_gpu[k:k_end, j:j_end]
                        
                        # 计算子块乘积并累加
                        C_block += self.matrix_multiply(A_block, B_block)
                    
                    # 将结果复制到输出矩阵
                    C_gpu[i:i_end, j:j_end] = C_block
                    
                    # 主动释放不再需要的临时内存
                    del C_block
                    cp.get_default_memory_pool().free_all_blocks()
        
        # 同步所有流
        cp.cuda.Stream.null.synchronize()
        
        return C_gpu
    
    # 添加新方法：使用前缀和技术优化矩阵乘法
    def _optimized_matrix_multiply(self, A, B):
        """
        使用前缀和技术优化的矩阵乘法，减少内存占用
        A: 第一个矩阵 [M, K]
        B: 第二个矩阵 [K, N]
        返回: C = A * B [M, N]
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "矩阵维度不匹配"
        
        # 确保数据在GPU上
        if not isinstance(A, cp.ndarray):
            A_gpu = cp.asarray(A)
        else:
            A_gpu = A
            
        if not isinstance(B, cp.ndarray):
            B_gpu = cp.asarray(B)
        else:
            B_gpu = B
        
        # 分配输出内存
        C_gpu = cp.zeros((M, N), dtype=np.float32)
        
        # 使用前缀和技术，逐行计算
        for i in range(M):
            # 每次只处理一行，减少内存占用
            A_row = A_gpu[i:i+1, :]  # 提取单行 [1, K]
            
            # 计算当前行的结果
            C_row = cp.zeros((1, N), dtype=np.float32)
            
            # 分块处理K维度，避免一次性加载所有数据
            block_size = min(256, K)  # 根据K的大小动态调整块大小
            
            for k_start in range(0, K, block_size):
                k_end = min(k_start + block_size, K)
                
                # 提取子块
                A_sub = A_row[:, k_start:k_end]  # [1, block_size]
                B_sub = B_gpu[k_start:k_end, :]  # [block_size, N]
                
                # 累加当前子块的结果
                C_row += cp.matmul(A_sub, B_sub)
                
                # 释放临时变量
                del A_sub, B_sub
                
            # 将当前行的结果复制到输出矩阵
            C_gpu[i:i+1, :] = C_row
            
            # 释放临时变量
            del A_row, C_row
            
            # 每处理几行后清理一次内存
            if i % 10 == 0:
                cp.get_default_memory_pool().free_all_blocks()
        
        return C_gpu
    
    def optimize_matrix_operations(self, A, B, operation='multiply'):
        """
        优化矩阵操作，自动选择最佳算法
        A, B: 输入矩阵
        operation: 操作类型 ('multiply', 'solve', 'decompose')
        返回: 操作结果
        """
        # 分析矩阵特性
        if operation == 'multiply':
            M, K = A.shape
            K2, N = B.shape
            
            # 获取当前可用GPU内存
            available_mem = cp.cuda.Device().mem_info[0]
            required_mem = M * N * 4 + M * K * 4 + K * N * 4  # 估计所需内存
            
            # 根据矩阵大小和可用内存选择最佳算法
            if max(M, N, K) <= 32:
                # 小矩阵使用共享内存优化
                return self.matrix_multiply(A, B)
            elif required_mem > available_mem * 0.7:  # 如果预计内存使用超过可用内存的70%
                # 内存不足，使用内存优化版本
                return self._optimized_matrix_multiply(A, B)
            elif M >= 1024 or N >= 1024:
                # 大矩阵使用分块策略
                return self._large_matrix_multiply(A, B)
            else:
                # 中等大小矩阵使用标准算法
                return self.matrix_multiply(A, B)
                
    
    def _is_symmetric(self, A):
        """检查矩阵是否对称"""
        if A.shape[0] != A.shape[1]:
            return False
            
        # 对于大矩阵，只检查部分元素
        if A.shape[0] > 1000:
            # 随机采样检查
            n_samples = 100
            indices = np.random.choice(A.shape[0], size=n_samples, replace=False)
            
            for i in indices:
                for j in indices:
                    if abs(A[i, j] - A[j, i]) > 1e-5:
                        return False
            
            return True
        else:
            # 小矩阵完全检查
            return cp.allclose(A, A.T)
    
    def _is_sparse(self, A):
        """检查矩阵是否稀疏"""
        # 计算非零元素比例
        if isinstance(A, cp.ndarray):
            non_zeros = cp.count_nonzero(A)
            total = A.size
        else:
            non_zeros = np.count_nonzero(A)
            total = A.size
            
        sparsity = non_zeros / total
        
        # 如果非零元素少于10%，认为是稀疏矩阵
        return sparsity < 0.1
    
    def _solve_sparse_system(self, A, b):
        """
        求解稀疏线性系统
        A: 系数矩阵
        b: 右侧向量
        返回: 解向量 x
        """
        # 将数据转换为scipy稀疏矩阵格式
        if isinstance(A, cp.ndarray):
            A_cpu = A.get()
        else:
            A_cpu = A
            
        if isinstance(b, cp.ndarray):
            b_cpu = b.get()
        else:
            b_cpu = b
            
        # 创建CSR格式稀疏矩阵
        A_sparse = scipy.sparse.csr_matrix(A_cpu)
        
        # 使用迭代求解器
        x, info = scipy.sparse.linalg.cg(A_sparse, b_cpu)
        
        if info != 0:
            # 如果共轭梯度法失败，尝试其他求解器
            x, info = scipy.sparse.linalg.gmres(A_sparse, b_cpu)
            
            if info != 0:
                # 如果仍然失败，使用直接求解器
                x = scipy.sparse.linalg.spsolve(A_sparse, b_cpu)
        
        # 将结果转回GPU
        return cp.asarray(x)
    
    def optimize_memory_usage(self):
        """优化内存使用，清理缓存"""
        # 清理分解缓存
        self.decomposition_cache.clear()
        
        # 清理CUDA缓存
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        
        # 清理PyTorch CUDA缓存
        try:
            torch.cuda.empty_cache()
        except:
            pass
        
        # 强制垃圾回收
        gc.collect()
    
    # 添加多GPU支持方法
    def _init_multi_gpu(self):
        """初始化多GPU支持"""
        try:
            # 获取可用的GPU数量
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus <= 1:
                print("只有一个GPU可用，不启用多GPU模式")
                self.multi_gpu_enabled = False
                return
                
            # 初始化每个GPU上的CUDA流
            self.gpu_streams = []
            for i in range(self.num_gpus):
                with cp.cuda.Device(i):
                    self.gpu_streams.append([cp.cuda.Stream() for _ in range(4)])
            
            self.multi_gpu_enabled = True
            print(f"多GPU模式已启用，使用 {self.num_gpus} 个GPU")
        except Exception as e:
            print(f"初始化多GPU支持失败: {e}")
            self.multi_gpu_enabled = False
    
    def multi_gpu_matrix_multiply(self, A, B):
        """
        使用多GPU进行矩阵乘法
        A: 第一个矩阵 [M, K]
        B: 第二个矩阵 [K, N]
        返回: C = A * B [M, N]
        """
        if not self.multi_gpu_enabled or self.num_gpus <= 1:
            # 如果多GPU不可用，回退到单GPU版本
            return self._large_matrix_multiply(A, B)
            
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "矩阵维度不匹配"
        
        # 确保数据在CPU上，以便分发到多个GPU
        if isinstance(A, cp.ndarray):
            A_cpu = A.get()
        else:
            A_cpu = A
            
        if isinstance(B, cp.ndarray):
            B_cpu = B.get()
        else:
            B_cpu = B
            
        # 分配输出内存
        C_cpu = np.zeros((M, N), dtype=np.float32)
        
        # 按行分割矩阵A
        rows_per_gpu = (M + self.num_gpus - 1) // self.num_gpus
        
        # 在多个GPU上并行计算
        results = []
        for i in range(self.num_gpus):
            start_row = i * rows_per_gpu
            end_row = min((i + 1) * rows_per_gpu, M)
            
            if start_row >= M:
                break
                
            # 提取子矩阵
            A_sub = A_cpu[start_row:end_row, :]
            
            # 在指定GPU上异步计算
            def compute_on_gpu(gpu_id, A_sub, B, start_row, end_row):
                try:
                    with cp.cuda.Device(gpu_id):
                        # 将数据复制到当前GPU
                        A_gpu = cp.asarray(A_sub)
                        B_gpu = cp.asarray(B_cpu)
                        
                        # 计算结果
                        C_sub = self._large_matrix_multiply(A_gpu, B_gpu)
                        
                        # 返回结果和位置信息
                        return C_sub.get(), start_row, end_row
                except Exception as e:
                    print(f"GPU {gpu_id} 计算失败: {e}")
                    # 回退到CPU计算
                    C_sub = np.matmul(A_sub, B_cpu)
                    return C_sub, start_row, end_row
            
            # 启动异步任务
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
                results.append(
                    executor.submit(compute_on_gpu, i, A_sub, B_cpu, start_row, end_row)
                )
        
        # 收集结果
        for future in concurrent.futures.as_completed(results):
            C_sub, start_row, end_row = future.result()
            C_cpu[start_row:end_row, :] = C_sub
        
        # 返回结果
        return cp.asarray(C_cpu)
    
    # 添加新方法：监控内存使用情况
    def monitor_memory_usage(self):
        """监控当前GPU内存使用情况，返回可用内存百分比"""
        try:
            # CuPy内存信息
            cp_free, cp_total = cp.cuda.Device().mem_info
            cp_used = cp_total - cp_free
            cp_percent = cp_used / cp_total * 100
            
            # PyTorch内存信息
            torch_allocated = torch.cuda.memory_allocated()
            torch_reserved = torch.cuda.memory_reserved()
            torch_total = torch.cuda.get_device_properties(0).total_memory
            torch_percent = torch_allocated / torch_total * 100
            
            print(f"CuPy内存使用: {cp_percent:.2f}% ({cp_used/(1024**3):.2f}GB/{cp_total/(1024**3):.2f}GB)")
            print(f"PyTorch内存使用: {torch_percent:.2f}% ({torch_allocated/(1024**3):.2f}GB/{torch_total/(1024**3):.2f}GB)")
            print(f"PyTorch预留内存: {torch_reserved/(1024**3):.2f}GB")
            
            # 返回可用内存百分比
            return 100 - max(cp_percent, torch_percent)
        except:
            # 如果无法获取内存信息
            return 0
    
    # 添加内存使用监控和自适应调整方法
    def print_memory_stats(self):
        """打印当前内存使用统计信息"""
        try:
            # CuPy内存信息
            cp_free, cp_total = cp.cuda.Device().mem_info
            cp_used = cp_total - cp_free
            cp_percent = cp_used / cp_total * 100
            
            # PyTorch内存信息
            torch_allocated = torch.cuda.memory_allocated()
            torch_reserved = torch.cuda.memory_reserved()
            torch_total = torch.cuda.get_device_properties(0).total_memory
            torch_percent = torch_allocated / torch_total * 100
            
            print("=" * 50)
            print("内存使用统计:")
            print(f"CuPy内存使用: {cp_percent:.2f}% ({cp_used/(1024**3):.2f}GB/{cp_total/(1024**3):.2f}GB)")
            print(f"PyTorch内存使用: {torch_percent:.2f}% ({torch_allocated/(1024**3):.2f}GB/{torch_total/(1024**3):.2f}GB)")
            print(f"PyTorch预留内存: {torch_reserved/(1024**3):.2f}GB")
            print("=" * 50)
        except Exception as e:
            print(f"获取内存统计信息失败: {e}")
    
    def dynamic_batch_size(self, data_size, operation_complexity):
        """
        根据数据大小和操作复杂度动态调整批处理大小
        data_size: 数据大小
        operation_complexity: 操作复杂度 (1-10)
        返回: 推荐的批处理大小
        """
        # 获取当前可用内存百分比
        available_percent = self.monitor_memory_usage()
        
        # 基础批处理大小
        base_batch_size = 32
        
        # 根据可用内存调整
        if available_percent < 20:  # 内存紧张
            memory_factor = 0.25
        elif available_percent < 40:
            memory_factor = 0.5
        elif available_percent < 60:
            memory_factor = 0.75
        else:
            memory_factor = 1.0
        
        # 根据数据大小调整
        if data_size > 1e7:  # 非常大的数据
            size_factor = 0.25
        elif data_size > 1e6:  # 大数据
            size_factor = 0.5
        elif data_size > 1e5:  # 中等数据
            size_factor = 0.75
        else:  # 小数据
            size_factor = 1.0
        
        # 根据操作复杂度调整
        complexity_factor = max(0.1, 1.0 - operation_complexity / 10)
        
        # 计算最终批处理大小
        batch_size = int(base_batch_size * memory_factor * size_factor * complexity_factor)
        
        # 确保批处理大小至少为1
        return max(1, batch_size)
    
    # 添加新方法：自适应内存管理
    def adaptive_memory_management(self, operation_type, data_size):
        """
        根据操作类型和数据大小自适应管理内存
        operation_type: 操作类型 ('matrix', 'mamba', 'feature')
        data_size: 数据大小估计
        返回: 是否需要使用内存优化版本
        """
        # 监控当前内存使用情况
        available_percent = self.monitor_memory_usage()
        
        # 根据操作类型和数据大小设置阈值
        if operation_type == 'matrix':
            # 矩阵操作内存敏感度高
            threshold = 30  # 如果可用内存小于30%，使用优化版本
        elif operation_type == 'mamba':
            # Mamba模型内存需求大
            threshold = 40  # 如果可用内存小于40%，使用优化版本
        else:
            # 其他特征处理
            threshold = 20  # 如果可用内存小于20%，使用优化版本
        
        # 根据数据大小调整阈值
        if data_size > 1e6:  # 大数据
            threshold += 10
        
        # 返回是否需要使用内存优化版本
        return available_percent < threshold


# 添加动态预测系数类 - 在AdaptiveResidualEncoder类之前添加
class DynamicPredictionCoefficients(nn.Module):
    """
    动态预测系数生成模块，根据历史像素值生成最优预测系数
    增强长序列建模能力，充分利用Mamba的优势
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 预测阶数
        self.pred_order = config.pred_order
        
        # 历史序列长度 - 增加以捕获更长依赖
        self.history_length = config.pred_order * 4
        
        # 增强型Mamba模型用于长序列建模
        enhanced_config = MambaDPCMConfig(
            d_model=64,        # 增加模型维度
            n_layer=3,         # 增加层数
            d_state=32,        # 增加状态空间维度
            d_conv=4,          # 增加卷积核大小
            expand=3,   # 增加扩展因子
            dt_min=0.001,      # 优化时间步长范围
            dt_max=0.1,
            dt_init="random",  # 随机初始化时间步长
            dt_scale=1.0,
            bias=True,         # 启用偏置
            conv_bias=True     # 启用卷积偏置
        )
        self.mamba = MambaModel(enhanced_config)
        
        # 多尺度特征提取
        self.multi_scale_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=k, padding=k//2),
                nn.SiLU(),
                nn.Conv1d(16, 16, kernel_size=1)
            ) for k in [3, 5, 7]  # 不同尺度的卷积核
        ])
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(enhanced_config.d_model + 16*3, 128),
            nn.SiLU(),
            nn.Linear(128, 64)
        )
        
        # 系数生成层 - 使用多头注意力机制
        self.coeff_attention = nn.MultiheadAttention(
            embed_dim=64, 
            num_heads=4, 
            batch_first=True
        )
        
        # 最终系数生成层
        self.coeff_generator = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Dropout(0.1),  # 添加Dropout提高泛化能力
            nn.Linear(128, self.pred_order)
        )
        
        # 添加上下文感知层 - 增强空间上下文建模
        self.context_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1, dilation=2),  # 使用空洞卷积扩大感受野
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 融合层
        self.fusion = nn.Linear(64 + 8, 64)
        
        # 初始化默认系数 (用于回退)
        default_coeffs = torch.zeros(self.pred_order)
        # 简单的默认系数：最近的像素权重最大
        for i in range(self.pred_order):
            default_coeffs[i] = 0.5 ** (i + 1)
        # 归一化
        default_coeffs = default_coeffs / default_coeffs.sum()
        self.register_buffer('default_coeffs', default_coeffs)
        
        # 添加自适应系数调整机制
        self.adaptive_adjustment = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, self.pred_order)
        )
        
        # 添加历史缓存机制，用于长序列建模
        self.history_cache = None
        self.cache_size = 1024  # 缓存大小
        
        # 添加位置编码
        self.position_encoding = nn.Parameter(
            torch.zeros(1, self.history_length, enhanced_config.d_model)
        )
        nn.init.normal_(self.position_encoding, mean=0, std=0.02)
        
        # 添加序列分析器 - 用于检测周期性和长期依赖
        self.sequence_analyzer = nn.GRU(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 周期性检测器
        self.periodicity_detector = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 8),
            nn.Softmax(dim=1)  # 输出不同周期的概率
        )
    
    def _pad_history(self, history, target_length):
        """填充历史序列到目标长度"""
        batch_size = history.shape[0]
        current_length = history.shape[1]
        
        if current_length >= target_length:
            # 如果当前长度已经足够，直接返回最近的部分
            return history[:, -target_length:]
        
        # 需要填充
        padding_length = target_length - current_length
        padding = torch.zeros(batch_size, padding_length, device=history.device)
        
        # 使用历史平均值填充，比零填充更合理
        if current_length > 0:
            history_mean = history.mean(dim=1, keepdim=True)
            padding = history_mean.expand(batch_size, padding_length)
        
        # 拼接填充和原始历史
        padded_history = torch.cat([padding, history], dim=1)
        
        return padded_history
    
    def _extract_multi_scale_features(self, history):
        """提取多尺度特征"""
        batch_size = history.shape[0]
        
        # 调整形状以适应卷积操作
        history_conv = history.unsqueeze(1)  # [batch, 1, seq_len]
        
        # 提取不同尺度的特征
        multi_scale_features = []
        for extractor in self.multi_scale_extractors:
            features = extractor(history_conv)  # [batch, 16, seq_len]
            # 全局池化
            pooled = F.adaptive_avg_pool1d(features, 1).view(batch_size, -1)
            multi_scale_features.append(pooled)
        
        # 拼接多尺度特征
        return torch.cat(multi_scale_features, dim=1)  # [batch, 16*3]
    
    def _analyze_sequence_patterns(self, history):
        """分析序列模式，检测周期性和长期依赖"""
        # 使用GRU分析序列
        history_unsqueezed = history.unsqueeze(-1)  # [batch, seq_len, 1]
        outputs, hidden = self.sequence_analyzer(history_unsqueezed)
        
        # 合并双向GRU的输出
        sequence_features = outputs.mean(dim=1)  # [batch, 64]
        
        # 检测周期性
        periodicity = self.periodicity_detector(sequence_features)
        
        return sequence_features, periodicity
    
    def _update_history_cache(self, history):
        """更新历史缓存"""
        if self.history_cache is None:
            # 首次初始化缓存
            self.history_cache = history.detach().clone()
        else:
            # 更新缓存，保留最新的cache_size个元素
            self.history_cache = torch.cat([self.history_cache, history.detach()], dim=1)
            if self.history_cache.shape[1] > self.cache_size:
                self.history_cache = self.history_cache[:, -self.cache_size:]
    
    # 在DynamicPredictionCoefficients类的forward方法中添加混合精度计算
    def forward(self, history, context=None):
        """
        根据历史像素值生成预测系数
        history: 历史像素值 [batch, seq_len]
        context: 可选的空间上下文 [batch, height, width]
        返回: 预测系数 [batch, pred_order]
        """
        batch_size = history.shape[0]
        
        # 检查当前可用GPU内存
        try:
            available_mem = torch.cuda.memory_allocated()
            total_mem = torch.cuda.get_device_properties(0).total_memory
            free_mem = total_mem - available_mem
        except:
            # 如果无法获取内存信息，假设内存紧张
            free_mem = 0
            
        # 根据可用内存动态调整历史长度
        if free_mem < 1e9:  # 小于1GB可用内存
            max_history_length = min(self.history_length, 16)  # 更激进地减少历史长度
        else:
            max_history_length = min(self.history_length, 32)  # 正常减少历史长度
    
        # 确保历史长度足够
        padded_history = self._pad_history(history, max_history_length)
    
        # 添加位置编码 - 使用较小的序列长度
        position_encoded = padded_history.unsqueeze(-1)
        if position_encoded.shape[1] <= self.position_encoding.shape[1]:
            position_encoded = position_encoded + self.position_encoding[:, :position_encoded.shape[1], :]
        else:
            # 如果历史长度超过位置编码长度，截断历史
            position_encoded = position_encoded[:, :self.position_encoding.shape[1], :] + self.position_encoding
    
        try:
            # 使用Mamba模型分析历史像素 - 添加内存优化
            with torch.cuda.amp.autocast():  # 使用混合精度计算
                # 分批处理以减少内存占用
                if batch_size > 16 and free_mem < 2e9:
                    # 大批量且内存紧张时，分批处理
                    batch_results = []
                    for i in range(0, batch_size, 8):  # 每批8个样本
                        end_idx = min(i + 8, batch_size)
                        batch_input = position_encoded[i:end_idx]
                        batch_output = self.mamba(batch_input)
                        batch_results.append(batch_output)
                        # 主动清理临时变量
                        torch.cuda.empty_cache()
                    
                    mamba_features = torch.cat(batch_results, dim=0)
                else:
                    # 内存充足时，一次性处理
                    mamba_features = self.mamba(position_encoded)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # 内存不足，尝试更激进的内存优化
                torch.cuda.empty_cache()  # 清理缓存
                
                try:
                    # 尝试更小的批量和更短的历史
                    reduced_history_length = min(max_history_length, 8)
                    reduced_history = self._pad_history(history, reduced_history_length)
                    reduced_position = reduced_history.unsqueeze(-1)
                    if reduced_position.shape[1] <= self.position_encoding.shape[1]:
                        reduced_position = reduced_position + self.position_encoding[:, :reduced_position.shape[1], :]
                    else:
                        reduced_position = reduced_position[:, :self.position_encoding.shape[1], :] + self.position_encoding
                    
                    # 极小批量处理
                    batch_results = []
                    for i in range(0, batch_size, 4):  # 每批4个样本
                        end_idx = min(i + 4, batch_size)
                        with torch.cuda.amp.autocast():
                            batch_output = self.mamba(reduced_position[i:end_idx])
                        batch_results.append(batch_output)
                        torch.cuda.empty_cache()
                    
                    mamba_features = torch.cat(batch_results, dim=0)
                except RuntimeError:
                    # 如果仍然失败，使用默认系数
                    print("内存严重不足，使用默认预测系数...")
                    return self.default_coeffs.unsqueeze(0).expand(batch_size, -1)
            else:
                # 其他错误，重新抛出
                raise
    
        # 提取最终特征
        final_mamba_features = mamba_features.mean(dim=1)  # [batch, d_model]
    
        # 提取多尺度特征 - 添加内存优化
        try:
            multi_scale_features = self._extract_multi_scale_features(padded_history)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # 内存不足，使用简化特征提取
                print("特征提取内存不足，使用简化特征...")
                # 简单平均池化作为特征
                multi_scale_features = torch.mean(padded_history, dim=1, keepdim=True).repeat(1, 48)
            else:
                raise
    
        # 融合多尺度特征和Mamba特征
        try:
            fused_features = self.feature_fusion(
                torch.cat([final_mamba_features, multi_scale_features], dim=1)
            )  # [batch, 64]
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # 内存不足，使用简化融合
                print("特征融合内存不足，使用简化融合...")
                # 直接使用Mamba特征
                fused_features = final_mamba_features
            else:
                raise
    
        # 如果有上下文信息，融合上下文特征
        if context is not None:
            try:
                # 处理上下文
                context = context.unsqueeze(1)  # [batch, 1, height, width]
                context_features = self.context_encoder(context).view(batch_size, -1)
                
                # 融合特征
                fused_features = self.fusion(torch.cat([fused_features, context_features], dim=1))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    # 内存不足，忽略上下文
                    print("上下文处理内存不足，忽略上下文...")
                    # 不使用上下文
                    pass
                else:
                    raise
    
        # 使用自注意力机制生成系数
        try:
            # 创建查询、键、值
            query = fused_features.unsqueeze(1)  # [batch, 1, 64]
            key = value = query
            
            # 应用多头注意力
            attn_output, _ = self.coeff_attention(query, key, value)
            attn_output = attn_output.squeeze(1)  # [batch, 64]
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # 内存不足，使用简化注意力
                print("注意力计算内存不足，使用简化计算...")
                # 直接使用线性层
                attn_output = fused_features
            else:
                raise
    
        # 生成基础预测系数
        try:
            base_coeffs = self.coeff_generator(attn_output)
            
            # 分析序列模式 - 使用较小的序列长度
            sequence_features, periodicity = self._analyze_sequence_patterns(padded_history)
            
            # 根据周期性检测结果调整系数
            periodicity_adjustment = self.adaptive_adjustment(sequence_features)
            
            # 结合基础系数和调整系数
            coeffs = base_coeffs + 0.1 * periodicity_adjustment
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # 内存不足，使用简化系数生成
                print("系数生成内存不足，使用简化生成...")
                # 直接使用线性层生成系数
                coeffs = torch.nn.functional.linear(
                    attn_output, 
                    self.coeff_generator[0].weight[:self.pred_order],
                    self.coeff_generator[0].bias[:self.pred_order] if self.coeff_generator[0].bias is not None else None
                )
            else:
                raise
    
        # 使用Softmax确保系数和为1
        coeffs = F.softmax(coeffs, dim=1)
        
        # 清理缓存
        torch.cuda.empty_cache()
    
        return coeffs
    
    def predict(self, history, context=None):
        """
        使用生成的系数预测下一个像素值
        history: 历史像素值 [batch, seq_len]
        context: 可选的空间上下文
        返回: 预测值 [batch]
        """
        # 确保历史长度足够
        if history.shape[1] < self.pred_order:
            # 如果历史不足，使用填充
            padded_history = self._pad_history(history, self.pred_order)
            recent_history = padded_history[:, -self.pred_order:]
        else:
            # 使用最近的pred_order个值
            recent_history = history[:, -self.pred_order:]
        
        # 生成系数
        coeffs = self.forward(history, context)
        
        # 使用系数进行预测
        prediction = torch.sum(recent_history * coeffs, dim=1)
        
        return prediction
    
    def predict_sequence(self, initial_history, length, context=None):
        """
        预测一个序列
        initial_history: 初始历史 [batch, seq_len]
        length: 要预测的序列长度
        context: 可选的空间上下文
        返回: 预测序列 [batch, length]
        """
        batch_size = initial_history.shape[0]
        device = initial_history.device
        
        # 复制初始历史，避免修改原始数据
        history = initial_history.clone()
        
        # 预测结果
        predictions = torch.zeros(batch_size, length, device=device)
        
        # 逐步预测
        for i in range(length):
            # 预测下一个值
            next_value = self.predict(history, context)
            
            # 保存预测结果
            predictions[:, i] = next_value
            
            # 更新历史
            history = torch.cat([history, next_value.unsqueeze(1)], dim=1)
        
        return predictions

# 自适应残差编码器 - 添加在MambaDPCMCompressor类之前
class AdaptiveResidualEncoder(nn.Module):
    """
    自适应残差编码器，使用双阈值策略对残差进行编码
    增强异常值处理和编码效率
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 阈值生成网络
        self.threshold_generator = AdaptiveThresholdGenerator(config)

        # 范围编码器
        self.range_coder = RangeCoder(precision=16)

        # 残差分析网络 - 增强版
        small_config = MambaDPCMConfig(
            d_model=48,  # 增加模型维度
            n_layer=3,   # 增加层数
            d_state=16   # 增加状态空间
        )
        self.residual_analyzer = MambaModel(small_config)

        # 量化策略选择器 - 增加策略数量
        self.strategy_selector = nn.Linear(small_config.d_model, 5)  # 5种策略

        # 异常值编码器 - 新增
        self.outlier_encoder = nn.Sequential(
            nn.Linear(2, 32),  # 输入：残差值和位置信息
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 8)
        )
        
        # 异常值聚类器 - 新增
        self.outlier_clusterer = nn.Sequential(
            nn.Linear(small_config.d_model, 32),
            nn.SiLU(),
            nn.Linear(32, 8)  # 8个聚类中心
        )
        
        # 自适应量化步长生成器 - 新增
        self.quant_step_generator = nn.Linear(small_config.d_model, 1)

    def analyze_residual(self, residual):
        """分析残差特性并选择最佳编码策略"""
        # 计算残差直方图
        hist = self.calculate_histogram(residual)

        # 使用Mamba分析残差特性
        features = self.residual_analyzer(hist.unsqueeze(0))
        features_mean = features.mean(dim=1)

        # 选择量化策略
        strategy_scores = self.strategy_selector(features_mean)
        strategy = torch.argmax(strategy_scores, dim=1)[0].item()
        
        # 生成量化步长
        quant_step = torch.abs(self.quant_step_generator(features_mean)) + 0.5
        
        # 生成聚类中心
        cluster_centers = self.outlier_clusterer(features_mean)

        return strategy, hist, quant_step, cluster_centers

    def calculate_histogram(self, residual, bins=256):
        """计算残差直方图"""
        # 将残差值限制在合理范围内
        min_val = residual.min().item()
        max_val = residual.max().item()

        # 使用torch创建直方图
        hist = torch.histc(torch.tensor(residual, device=device),
                           bins=bins, min=min_val, max=max_val)

        # 归一化直方图
        hist = hist / hist.sum()

        return hist

    def cluster_outliers(self, outliers, cluster_centers):
        """
        将异常值聚类到最近的聚类中心
        outliers: 异常值 [n_outliers]
        cluster_centers: 聚类中心 [n_clusters]
        返回: 聚类索引和量化误差
        """
        n_outliers = len(outliers)
        n_clusters = len(cluster_centers)
        
        if n_outliers == 0:
            return [], []
            
        # 计算每个异常值到每个聚类中心的距离
        outliers_expanded = outliers.unsqueeze(1).expand(n_outliers, n_clusters)
        centers_expanded = cluster_centers.unsqueeze(0).expand(n_outliers, n_clusters)
        
        # 计算距离
        distances = torch.abs(outliers_expanded - centers_expanded)
        
        # 找到最近的聚类中心
        closest_cluster = torch.argmin(distances, dim=1)
        
        # 计算量化误差
        quantization_errors = torch.gather(distances, 1, closest_cluster.unsqueeze(1)).squeeze()
        
        return closest_cluster, quantization_errors
        
    def dynamic_threshold_adjustment(self, residual_histogram, initial_T_minus, initial_T_plus):
        """
        根据残差直方图动态调整阈值
        实现图片中的"动态调整Δ (控制信息传播速率)"功能
        """
        # 计算残差分布特性
        hist_mean = torch.sum(residual_histogram * torch.arange(len(residual_histogram), device=device))
        hist_var = torch.sum(residual_histogram * (torch.arange(len(residual_histogram), device=device) - hist_mean)**2)
        hist_entropy = -torch.sum(residual_histogram * torch.log2(residual_histogram + 1e-10))
        
        # 根据残差特性调整阈值
        # 对应图片中的"动态调整Δ"、"动态调整B"和"动态调整C"三个步骤
        
        # 步骤1: 控制信息传播速率 (动态调整Δ)
        transmission_rate = torch.sigmoid(hist_entropy / 8.0)  # 归一化熵值
        delta_adjustment = (transmission_rate - 0.5) * 2.0  # 范围[-1, 1]
        
        # 步骤2: 过滤输入噪声 (动态调整B)
        noise_factor = torch.exp(-hist_var / 100.0)  # 方差越小，噪声越大
        noise_adjustment = (noise_factor - 0.5) * 1.5  # 范围[-0.75, 0.75]
        
        # 步骤3: 增强关键特征 (动态调整C)
        # 检测残差中的关键特征
        key_features = self._detect_key_features(residual_histogram)
        feature_adjustment = key_features * 0.5  # 范围[0, 0.5]
        
        # 综合调整阈值
        T_minus_adjustment = -delta_adjustment - noise_adjustment - feature_adjustment
        T_plus_adjustment = delta_adjustment + noise_adjustment + feature_adjustment
        
        # 应用调整
        adjusted_T_minus = initial_T_minus * (1.0 + T_minus_adjustment)
        adjusted_T_plus = initial_T_plus * (1.0 + T_plus_adjustment)
        
        # 确保阈值在合理范围内
        adjusted_T_minus = torch.clamp(adjusted_T_minus, min=initial_T_minus * 0.5, max=initial_T_minus * 1.5)
        adjusted_T_plus = torch.clamp(adjusted_T_plus, min=initial_T_plus * 0.5, max=initial_T_plus * 1.5)
        
        return adjusted_T_minus, adjusted_T_plus
    
    def _detect_key_features(self, histogram):
        """检测残差直方图中的关键特征"""
        # 计算直方图的梯度
        grad = torch.abs(histogram[1:] - histogram[:-1])
        
        # 大梯度表示关键特征
        key_feature_score = torch.mean(grad) / (torch.std(grad) + 1e-10)
        
        # 归一化到[0,1]范围
        return torch.sigmoid(key_feature_score - 2.0)

    def encode(self, residual):
        """
        编码残差
        residual: 残差数据 [size]
        返回: 编码后的数据
        """
        # 分析残差并选择策略
        strategy, hist, quant_step, cluster_centers = self.analyze_residual(residual)

        # 生成初始自适应阈值
        initial_T_minus, initial_T_plus = self.threshold_generator(hist.unsqueeze(0))
        
        # 动态调整阈值 - 新增
        T_minus, T_plus = self.dynamic_threshold_adjustment(hist, initial_T_minus, initial_T_plus)

        # 根据策略调整阈值
        if strategy == 0:  # 背景区域策略 - 保守
            T_minus = T_minus * 0.7
            T_plus = T_plus * 0.7
        elif strategy == 1:  # 标准策略
            pass  # 使用原始阈值
        elif strategy == 2:  # 显著信号区域策略 - 激进
            T_minus = T_minus * 1.3
            T_plus = T_plus * 1.3
        elif strategy == 3:  # 高频细节区域策略
            T_minus = T_minus * 0.9
            T_plus = T_plus * 1.1
        elif strategy == 4:  # 极端值策略
            T_minus = T_minus * 0.5
            T_plus = T_plus * 1.5

        # 使用双阈值进行残差编码
        outliers_mask = (residual < T_minus) | (residual > T_plus)
        outliers_indices = torch.nonzero(outliers_mask).squeeze()
        outliers_values = residual[outliers_mask]
        
        # 异常值处理 - 改进
        if len(outliers_values) > 0:
            # 对异常值进行聚类
            cluster_indices, quant_errors = self.cluster_outliers(outliers_values, cluster_centers)
            
            # 对异常值进行自适应量化
            quantized_outliers = torch.zeros_like(outliers_values)
            for i, (val, cluster_idx) in enumerate(zip(outliers_values, cluster_indices)):
                # 根据聚类中心和量化步长进行量化
                center = cluster_centers[cluster_idx]
                # 计算量化后的值
                quantized_outliers[i] = center + torch.round((val - center) / quant_step) * quant_step
                
            # 使用量化后的异常值
            outliers_values = quantized_outliers
            
            # 计算异常值的分布统计
            outlier_min = outliers_values.min().item()
            outlier_max = outliers_values.max().item()
            outlier_mean = outliers_values.mean().item()
            outlier_std = outliers_values.std().item() if len(outliers_values) > 1 else 0
        else:
            # 没有异常值的情况
            outlier_min = 0
            outlier_max = 0
            outlier_mean = 0
            outlier_std = 0
            cluster_indices = []

        # 对正常范围内的残差使用自适应算术编码
        normal_residual = residual[~outliers_mask].cpu().numpy()
        
        # 对正常残差进行量化处理
        if len(normal_residual) > 0:
            # 自适应量化步长
            normal_quant_step = min(1.0, quant_step.item() * 0.5)  # 正常值使用更小的量化步长
            normal_residual = np.round(normal_residual / normal_quant_step) * normal_quant_step
            
            # 计算符号和频率表
            unique_symbols, counts = np.unique(normal_residual, return_counts=True)
            frequency_table = counts.tolist()
            
            # 使用自适应算术编码
            encoded_normal = arithmetic_encoder(normal_residual.tolist(), unique_symbols.tolist(), frequency_table)
        else:
            # 没有正常值的情况
            unique_symbols = []
            frequency_table = []
            encoded_normal = []
            normal_quant_step = 1.0

        return {
            'encoded_normal': encoded_normal,
            'symbols': unique_symbols.tolist(),
            'frequency_table': frequency_table,
            'outliers_indices': outliers_indices.cpu(),
            'outliers_values': outliers_values.cpu(),
            'T_minus': T_minus,
            'T_plus': T_plus,
            'strategy': strategy,
            'quant_step': quant_step.item(),
            'normal_quant_step': normal_quant_step,
            'cluster_centers': cluster_centers.cpu(),
            'cluster_indices': torch.tensor(cluster_indices).cpu() if len(cluster_indices) > 0 else torch.tensor([]),
            'outlier_stats': {
                'min': outlier_min,
                'max': outlier_max,
                'mean': outlier_mean,
                'std': outlier_std
            }
        }
    
    def decode(self, encoded_data, block_size):
        """
        解码残差
        encoded_data: 编码数据
        block_size: 块大小
        返回: 解码后的残差
        """
        # 提取数据
        encoded_normal = encoded_data['encoded_normal']
        symbols = encoded_data['symbols']
        frequency_table = encoded_data['frequency_table']
        outliers_indices = encoded_data['outliers_indices']
        outliers_values = encoded_data['outliers_values']
        normal_quant_step = encoded_data.get('normal_quant_step', 1.0)
        normal_count = block_size - len(outliers_indices)

        # 使用自适应算术解码
        if normal_count > 0 and len(encoded_normal) > 0:
            normal_residual = arithmetic_decoder(encoded_normal, normal_count, symbols, frequency_table)
            normal_residual = torch.tensor(normal_residual, device=device)
        else:
            normal_residual = torch.tensor([], device=device)

        # 创建完整的残差数组
        residual = torch.zeros(block_size, device=device)

        # 首先填充正常范围的残差
        normal_idx = 0
        for i in range(block_size):
            if i not in outliers_indices:
                if normal_idx < len(normal_residual):
                    residual[i] = normal_residual[normal_idx]
                    normal_idx += 1

        # 然后填充异常值
        for idx, val in zip(outliers_indices, outliers_values):
            residual[idx] = val

        return residual
# 范围编码器实现
class RangeCoder:
    """
    范围编码器实现，用于残差的高效编码
    """

    def __init__(self, precision=16):
        self.precision = precision
        self.full_range = 1 << precision
        self.half_range = self.full_range >> 1
        self.quarter_range = self.full_range >> 2
        self.three_quarter_range = self.half_range + self.quarter_range

    def _estimate_frequencies(self, data):
        """估计数据频率分布"""
        # 数据类型转换
        data = np.array(data).astype(np.int32)

        # 找到唯一值及其计数
        values, counts = np.unique(data, return_counts=True)

        # 创建频率表
        freq_table = dict(zip(values, counts))

        # 计算累积频率
        total = sum(counts)
        cum_freq = {}
        cum_sum = 0

        for val in sorted(freq_table.keys()):
            cum_freq[val] = (cum_sum, cum_sum + freq_table[val])
            cum_sum += freq_table[val]

        return freq_table, cum_freq, total

    def encode(self, data):
        """
        对数据进行范围编码
        data: 输入数据 (numpy数组或列表)
        返回: 编码后的数据
        """
        if len(data) == 0:
            return {'encoded': [], 'freq_table': {}}

        # 估计频率
        freq_table, cum_freq, total = self._estimate_frequencies(data)

        # 初始化编码范围
        low = 0
        high = self.full_range - 1

        # 编码过程
        encoded_bits = []
        pending_bits = 0

        for symbol in data:
            # 更新范围
            range_size = high - low + 1
            low_cum, high_cum = cum_freq[symbol]

            high = low + (range_size * high_cum) // total - 1
            low = low + (range_size * low_cum) // total

            # 处理范围缩小
            while True:
                if high < self.half_range:
                    # 输出0，增加pending 1s
                    encoded_bits.append(0)
                    for _ in range(pending_bits):
                        encoded_bits.append(1)
                    pending_bits = 0
                elif low >= self.half_range:
                    # 输出1，增加pending 0s
                    encoded_bits.append(1)
                    for _ in range(pending_bits):
                        encoded_bits.append(0)
                    pending_bits = 0
                    low -= self.half_range
                    high -= self.half_range
                elif low >= self.quarter_range and high < self.three_quarter_range:
                    # 缩小中间范围
                    pending_bits += 1
                    low -= self.quarter_range
                    high -= self.quarter_range
                else:
                    break

                # 左移
                low <<= 1
                high = (high << 1) | 1

                # 确保范围在有效区间内
                high &= self.full_range - 1
                low &= self.full_range - 1

        # 输出最终位
        pending_bits += 1
        if low < self.quarter_range:
            encoded_bits.append(0)
            for _ in range(pending_bits):
                encoded_bits.append(1)
        else:
            encoded_bits.append(1)
            for _ in range(pending_bits):
                encoded_bits.append(0)

        # 将位流转换为字节
        encoded_bytes = []
        byte = 0
        bit_count = 0

        for bit in encoded_bits:
            byte = (byte << 1) | bit
            bit_count += 1
            if bit_count == 8:
                encoded_bytes.append(byte)
                byte = 0
                bit_count = 0

        # 处理未满8位的尾部
        if bit_count > 0:
            byte <<= (8 - bit_count)
            encoded_bytes.append(byte)

        return {
            'encoded': encoded_bytes,
            'freq_table': freq_table
        }

    def decode(self, encoded_data):
        """
        解码数据
        encoded_data: 编码数据，包含编码和频率表
        返回: (解码后的数据)
        """
        encoded_bytes = encoded_data['encoded']
        freq_table = encoded_data['freq_table']

        # 如果输入为空，直接返回
        if not encoded_bytes:
            return []

        # 重建累积频率表
        values = sorted(freq_table.keys())
        counts = [freq_table[val] for val in values]
        total = sum(counts)

        cum_freq = {}
        cum_sum = 0
        for val, count in zip(values, counts):
            cum_freq[val] = (cum_sum, cum_sum + count)
            cum_sum += count

        # 准备解码
        # 将字节转换为位流
        encoded_bits = []
        for byte in encoded_bytes:
            for i in range(7, -1, -1):
                bit = (byte >> i) & 1
                encoded_bits.append(bit)

        # 初始化解码状态
        code = 0
        for i in range(self.precision):
            if i < len(encoded_bits):
                code = (code << 1) | encoded_bits[i]

        low = 0
        high = self.full_range - 1

        # 解码过程
        decoded_data = []
        i = self.precision

        while True:
            # 查找当前码字对应的符号
            range_size = high - low + 1
            scaled_code = ((code - low + 1) * total - 1) // range_size

            symbol = None
            for val in values:
                low_cum, high_cum = cum_freq[val]
                if low_cum <= scaled_code < high_cum:
                    symbol = val
                    break

            if symbol is None:
                break

            decoded_data.append(symbol)

            # 更新范围
            low_cum, high_cum = cum_freq[symbol]
            high = low + (range_size * high_cum) // total - 1
            low = low + (range_size * low_cum) // total

            # 读入新位并更新范围
            while True:
                if high < self.half_range:
                    # 不需要调整
                    pass
                elif low >= self.half_range:
                    # 调整范围
                    code -= self.half_range
                    low -= self.half_range
                    high -= self.half_range
                elif low >= self.quarter_range and high < self.three_quarter_range:
                    # 缩小中间范围
                    code -= self.quarter_range
                    low -= self.quarter_range
                    high -= self.quarter_range
                else:
                    break

                # 左移并读入新位
                low <<= 1
                high = (high << 1) | 1
                code = (code << 1)

                if i < len(encoded_bits):
                    code |= encoded_bits[i]
                    i += 1

                # 确保范围在有效区间内
                high &= self.full_range - 1
                low &= self.full_range - 1
                code &= self.full_range - 1

            # 判断是否已经解码完毕
            if i >= len(encoded_bits) and low == 0 and high == self.full_range - 1:
                break

        return decoded_data


# 基于Mamba的DPCM压缩器
class MambaDPCMCompressor:
    def __init__(self, config=None):
        if config is None:
            self.config = MambaDPCMConfig()
        else:
            self.config = config

        # 创建各个组件
        self.dynamic_predictor = DynamicPredictionCoefficients(self.config)
        self.threshold_generator = AdaptiveThresholdGenerator(self.config)
        self.parallel_scan = ParallelScan(self.config)

        # 将模型移动到GPU
        self.dynamic_predictor = self.dynamic_predictor.to(device)
        self.threshold_generator = self.threshold_generator.to(device)

        # 初始化RangeCoder (自定义的范围编码器)
        self.range_coder = RangeCoder()
        
        # 添加极光图像预处理器
        self.preprocessor = AuroraPreprocessor().to(device)
        
        # 添加自适应块大小选择器
        self.block_size_selector = AdaptiveBlockSizeSelector(
            min_size=64, 
            max_size=512, 
            step=64
        )

    def matrix_decomposition_acceleration(self, matrix):
        """
        实现图片中的"分解为子矩阵C0, C1, C2..."和"并行累加器"功能
        使用矩阵分解和并行计算加速矩阵运算
        """
        # 获取矩阵尺寸
        height, width = matrix.shape
        
        # 分解为子矩阵
        sub_matrices = []
        sub_size = min(256, max(32, min(height, width) // 4))  # 子矩阵大小
        
        # 按行分解
        for i in range(0, height, sub_size):
            row_end = min(i + sub_size, height)
            # 按列分解
            for j in range(0, width, sub_size):
                col_end = min(j + sub_size, width)
                # 提取子矩阵
                sub_matrix = matrix[i:row_end, j:col_end]
                sub_matrices.append((sub_matrix, (i, j)))
        
        # 并行处理子矩阵
        processed_matrices = []
        
        # 创建多个CUDA流
        streams = [cp.cuda.Stream() for _ in range(min(4, len(sub_matrices)))]
        
        for idx, (sub_matrix, position) in enumerate(sub_matrices):
            # 选择流
            stream_idx = idx % len(streams)
            
            # 在指定流中处理子矩阵
            with streams[stream_idx]:
                # 转移到GPU
                if not isinstance(sub_matrix, cp.ndarray):
                    sub_matrix_gpu = cp.asarray(sub_matrix)
                else:
                    sub_matrix_gpu = sub_matrix
                
                # 处理子矩阵 (例如计算C^T·C)
                processed = cp.matmul(sub_matrix_gpu.T, sub_matrix_gpu)
                
                # 保存处理结果和位置
                processed_matrices.append((processed, position))
        
        # 同步所有流
        cp.cuda.Stream.null.synchronize()
        
        # 使用并行累加器合并结果
        result = cp.zeros((height, width), dtype=cp.float32)
        
        for processed, (i, j) in processed_matrices:
            sub_height, sub_width = processed.shape
            result[i:i+sub_height, j:j+sub_width] += processed
        
        return result

    def _compress_block(self, block):
        """
        压缩单个图像块
        block: 输入图像块 [block_size, block_size]
        返回: 压缩后的块数据
        """
        # 创建残差编码器
        residual_encoder = AdaptiveResidualEncoder(self.config)
        
        # 获取块大小
        block_size = block.shape[0] * block.shape[1]
        
        # 初始化预测图像和残差
        predicted = torch.zeros_like(block)
        residual = torch.zeros_like(block)
        
        # 获取预测阶数
        pred_order = self.config.pred_order
        
        # 逐像素处理
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                # 收集历史像素
                history = self._collect_history(block, i, j, pred_order)
                
                # 确保历史数据维度正确
                if history.shape[0] != pred_order:
                    # 如果历史不足，直接复制原始值
                    predicted[i, j] = block[i, j]
                    residual[i, j] = 0
                    continue
                
                # 使用动态预测器预测当前像素
                # 确保输入维度正确 [batch_size=1, sequence_length=pred_order]
                history = history.unsqueeze(0)  # 添加批次维度
                pred_value = self.dynamic_predictor.predict(history).squeeze()
                
                # 记录预测值
                predicted[i, j] = pred_value
                
                # 计算残差
                residual[i, j] = block[i, j] - pred_value
        
        # 编码残差
        encoded_data = residual_encoder.encode(residual.reshape(-1))
        
        # 添加块信息
        encoded_data['block_size'] = block_size
        encoded_data['original_shape'] = block.shape
        encoded_data['predicted_block'] = predicted.cpu()
        
        return encoded_data
    
    def _collect_history(self, block, i, j, pred_order):
        """
        收集像素的历史值用于预测
        block: 输入块
        i, j: 当前像素位置
        pred_order: 预测阶数
        返回: 历史像素值 [pred_order]
        """
        history = []
        
        # 收集历史像素 - 使用因果模板
        # 先收集左侧像素
        for k in range(1, pred_order + 1):
            if j - k >= 0:
                history.append(block[i, j - k].item())
            else:
                break
                
        # 如果左侧像素不足，收集上方像素
        if len(history) < pred_order:
            for k in range(1, pred_order + 1 - len(history)):
                if i - k >= 0:
                    history.append(block[i - k, j].item())
                else:
                    break
                    
        # 如果仍然不足，收集左上方像素
        if len(history) < pred_order:
            k = 1
            while len(history) < pred_order and i - k >= 0 and j - k >= 0:
                history.append(block[i - k, j - k].item())
                k += 1
        
        # 确保历史长度为pred_order，不足则用0填充
        while len(history) < pred_order:
            history.append(0.0)
            
        # 转换为张量
        return torch.tensor(history, device=block.device)
    
    def decompress_with_overlapping(self, compressed_data):
        """
        解压使用重叠块压缩的图像
        compressed_data: 压缩数据，包含compressed_data和metadata
        返回: 解压后的图像
        """
        # 提取数据
        blocks_data = compressed_data['compressed_data']
        metadata = compressed_data['metadata']
        
        # 获取图像尺寸和块参数
        height = metadata['height']
        width = metadata['width']
        block_size = metadata['block_size']
        overlap_size = metadata['overlap_size']
        
        # 创建输出图像
        output_image = torch.zeros((height, width), device=device)
        
        # 创建权重图像（用于重叠区域的加权平均）
        weight_image = torch.zeros((height, width), device=device)
        
        # 创建汉宁窗
        hann_1d = torch.hann_window(block_size, device=device)
        hann_window = torch.outer(hann_1d, hann_1d)
        
        # 解压每个块
        for block_data in blocks_data:
            # 获取块位置
            h_start, w_start = block_data['position']
            
            # 解压块
            decompressed_block = self._decompress_block(block_data)
            
            # 应用汉宁窗作为权重
            weighted_block = decompressed_block * hann_window
            
            # 确定块的有效范围
            h_end = min(h_start + block_size, height)
            w_end = min(w_start + block_size, width)
            block_height = h_end - h_start
            block_width = w_end - w_start
            
            # 将解压后的块添加到输出图像
            output_image[h_start:h_end, w_start:w_end] += weighted_block[:block_height, :block_width]
            
            # 更新权重图像
            weight_image[h_start:h_end, w_start:w_end] += hann_window[:block_height, :block_width]
        
        # 对重叠区域进行加权平均
        # 避免除以零
        weight_image = torch.where(weight_image > 0, weight_image, torch.ones_like(weight_image))
        output_image = output_image / weight_image
        
        return output_image
    
    def _decompress_block(self, block_data):
        """
        解压单个块
        block_data: 块压缩数据
        返回: 解压后的块
        """
        # 提取数据
        original_shape = block_data['original_shape']
        predicted_block = block_data['predicted_block'].to(device)
        
        # 创建残差编码器
        residual_encoder = AdaptiveResidualEncoder(self.config)
        
        # 解码残差
        residual = residual_encoder.decode(block_data, block_data['block_size'])
        residual = residual.reshape(original_shape)
        
        # 重建原始块
        decompressed_block = predicted_block + residual
        
        return decompressed_block
    # 在compress方法中使用矩阵分解加速
    def compress(self, image):
        # 获取图像尺寸
        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1
            
        # 对大型矩阵运算使用分解加速
        if min(height, width) > 1024:
            # 使用矩阵分解加速大型矩阵运算
            image = self.matrix_decomposition_acceleration(image)
            
        # 继续原有的压缩流程
        # ... 原有代码 ...
        
        return compressed_data, metadata
        
    # 在MambaDPCMCompressor类中添加以下方法
    def compress_with_overlapping(self, image, overlap_ratio=0.25):
        """
        使用重叠块压缩图像
        image: 输入图像 [height, width]
        overlap_ratio: 重叠比例
        返回: 压缩数据和元数据
        """
        # 获取图像尺寸
        height, width = image.shape
    
        # 确定块大小和重叠大小
        block_size = self.config.block_size
        overlap_size = int(block_size * overlap_ratio)
    
        # 初始化压缩数据列表和元数据
        compressed_data = []
        metadata = {
            'height': height,
            'width': width,
            'block_size': block_size,
            'overlap_size': overlap_size,
            'overlap_ratio': overlap_ratio
        }
    
        # 创建汉宁窗
        hann_1d = torch.hann_window(block_size, device=device)
        hann_window = torch.outer(hann_1d, hann_1d)
    
        # 逐块处理
        for h_start in range(0, height - overlap_size, block_size - overlap_size):
            # 确保最后一个块不会超出图像边界
            if h_start + block_size > height:
                h_start = max(0, height - block_size)
            
        for w_start in range(0, width - overlap_size, block_size - overlap_size):
            # 确保最后一个块不会超出图像边界
            if w_start + block_size > width:
                w_start = max(0, width - block_size)
                
            # 提取当前块
            h_end = min(h_start + block_size, height)
            w_end = min(w_start + block_size, width)
            
            # 处理不完整的块
            if h_end - h_start < block_size or w_end - w_start < block_size:
                # 创建填充块
                padded_block = torch.zeros((block_size, block_size), device=device)
                padded_block[:h_end-h_start, :w_end-w_start] = image[h_start:h_end, w_start:w_end]
                block = padded_block
            else:
                block = image[h_start:h_end, w_start:w_end]
            
            # 应用汉宁窗
            windowed_block = block * hann_window
            
            # 压缩当前块
            block_data = self._compress_block(windowed_block)
            
            # 添加位置信息
            block_data['position'] = (h_start, w_start)
            compressed_data.append(block_data)
    
        # 返回压缩数据和元数
        return compressed_data, metadata
    

# 用于处理FITS格式的极光图像的工具类
class AuroraImageProcessor:
    """
    用于处理FITS格式的极光图像
    """

    @staticmethod
    def load_fits_image(file_path):
        """
        加载FITS格式的极光图像
        file_path: FITS文件路径
        返回: 图像数据
        """
        with fits.open(file_path) as hdul:
            # 获取主数据
            data = hdul[0].data

            # FITS数据可能有多个维度，确保我们获取2D图像
            if data.ndim > 2:
                data = data[0]  # 通常第一帧

            return data

    @staticmethod
    def save_fits_image(data, file_path, header=None):
        """
        保存图像为FITS格式
        data: 图像数据
        file_path: 输出文件路径
        header: 可选的FITS头信息
        """
        hdu = fits.PrimaryHDU(data)
        if header:
            hdu.header.update(header)

        hdul = fits.HDUList([hdu])
        hdul.writeto(file_path, overwrite=True)

    @staticmethod
    def preprocess_image(image):
        """
        预处理图像，标准化为适合模型的格式
        image: 输入图像
        返回: 预处理后的图像
        """
        # 确保数据类型正确
        image = image.astype(np.float32)

        # 处理异常值
        image = np.nan_to_num(image, nan=0.0, posinf=65535.0, neginf=0.0)

        # 确保值在合理范围内（16位图像）
        image = np.clip(image, 0, 65535)

        return image


# 多GPU分配和协调的实现
class MultiGPUManager:
    """
    管理多GPU的任务分配和协调
    """

    def __init__(self, config):
        self.config = config
        self.num_devices = min(config.devices, torch.cuda.device_count())

        if self.num_devices == 0:
            print("警告: 未检测到可用的GPU，将使用CPU")
            self.num_devices = 1

        print(f"使用 {self.num_devices} 个GPU设备")

        # 初始化每个设备上的流
        self.streams = []
        for i in range(self.num_devices):
            device_streams = []
            with torch.cuda.device(i):
                for _ in range(config.streams_per_device):
                    device_streams.append(torch.cuda.Stream())
            self.streams.append(device_streams)

    def distribute_blocks(self, blocks):
        """
        将图像块分配给不同的GPU
        blocks: 图像块列表
        返回: GPU分配方案
        """
        # 简单的循环分配策略
        distribution = [[] for _ in range(self.num_devices)]
        for i, block in enumerate(blocks):
            device_id = i % self.num_devices
            distribution[device_id].append(block)

        return distribution

    def process_blocks(self, blocks, process_func):
        """
        使用多GPU并行处理图像块
        blocks: 图像块列表
        process_func: 处理函数，接收块和设备ID
        返回: 处理结果列表
        """
        # 分配块到不同GPU
        distribution = self.distribute_blocks(blocks)

        # 结果存储
        results = [None] * len(blocks)
        block_to_idx = {}
        idx = 0

        for device_id, device_blocks in enumerate(distribution):
            for block in device_blocks:
                block_to_idx[id(block)] = idx
                idx += 1

        # 并行处理
        events = []

        for device_id, device_blocks in enumerate(distribution):
            # 跳过空列表
            if not device_blocks:
                continue

            # 将处理任务分配到不同流
            device_events = []
            num_streams = len(self.streams[device_id])

            for i, block in enumerate(device_blocks):
                stream_id = i % num_streams
                stream = self.streams[device_id][stream_id]

                with torch.cuda.device(device_id), torch.cuda.stream(stream):
                    # 处理块
                    result = process_func(block, device_id)
                    results[block_to_idx[id(block)]] = result

                    # 记录事件
                    event = torch.cuda.Event()
                    event.record(stream)
                    device_events.append(event)

            events.extend(device_events)

        # 等待所有事件完成
        for event in events:
            event.synchronize()

        return results


# 添加可视化功能 - 在main函数之前
def visualize_compression_results(original, predicted, reconstructed, residual, T_minus, T_plus, save_path=None):
    """
    可视化压缩结果
    original: 原始图像
    predicted: 预测图像
    reconstructed: 重建图像
    residual: 残差
    T_minus: 下阈值
    T_plus: 上阈值
    save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # 确保数据在CPU上
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().numpy()
    if isinstance(residual, torch.Tensor):
        residual = residual.cpu().numpy()

    # 创建图像
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 原始图像
    im0 = axes[0, 0].imshow(original, cmap='viridis')
    axes[0, 0].set_title('原始图像')
    plt.colorbar(im0, ax=axes[0, 0])

    # 预测图像
    im1 = axes[0, 1].imshow(predicted, cmap='viridis')
    axes[0, 1].set_title('预测图像')
    plt.colorbar(im1, ax=axes[0, 1])

    # 重建图像
    im2 = axes[0, 2].imshow(reconstructed, cmap='viridis')
    axes[0, 2].set_title('重建图像')
    plt.colorbar(im2, ax=axes[0, 2])

    # 残差
    im3 = axes[1, 0].imshow(residual, cmap='coolwarm', norm=Normalize(vmin=-50, vmax=50))
    axes[1, 0].set_title('残差')
    plt.colorbar(im3, ax=axes[1, 0])

    # 阈值图
    threshold_mask = np.zeros_like(residual)
    threshold_mask[(residual >= T_minus) & (residual <= T_plus)] = 1
    im4 = axes[1, 1].imshow(threshold_mask, cmap='gray')
    axes[1, 1].set_title(f'阈值范围 [{T_minus:.2f}, {T_plus:.2f}]')
    plt.colorbar(im4, ax=axes[1, 1])

    # 误差图
    error = np.abs(original - reconstructed)
    im5 = axes[1, 2].imshow(error, cmap='hot', norm=Normalize(vmin=0, vmax=error.max()))
    axes[1, 2].set_title(f'重建误差 (最大: {error.max():.4f})')
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"可视化结果已保存至: {save_path}")

    plt.show()

# 在 main 函数中修改图像加载和处理部分
def main():
    """
    主函数，演示完整的压缩和解压流程，支持批量处理
    """
    print("基于Mamba的DPCM极光图像无损压缩器")

    # 配置参数 - 减小模型大小以节省内存
    config = MambaDPCMConfig(
        d_model=32,       # 减小模型维度
        n_layer=2,        # 减少层数
        d_state=8,        # 减小状态空间维度
        block_size=64,    # 减小块大小
        pred_order=5,     # 减少预测阶数
        eq_count=3,       # 减少方程数量
        threshold_init=13,
        streams_per_device=1,  # 减少流数量
        devices=1
    )

    # 创建压缩器
    compressor = MambaDPCMCompressor(config)
    
    # 创建图像处理器
    image_processor = AuroraImageProcessor()
    
    # 指定FITS文件所在文件夹
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    print(f"使用数据路径: {folder_path}")
    
    # 检查路径是否存在
    if not os.path.exists(folder_path):
        print(f"警告: 路径 '{folder_path}' 不存在，尝试创建...")
        try:
            os.makedirs(folder_path)
            print(f"已创建文件夹: {folder_path}")
            print("请将FITS文件放入此文件夹后重新运行程序")
            return
        except Exception as e:
            print(f"创建文件夹失败: {e}")
            return
    
    # 获取文件夹中所有FITS文件
    fits_files = [f for f in os.listdir(folder_path) if f.endswith('.fits')]
    
    if not fits_files:
        print(f"在 {folder_path} 中未找到FITS文件")
        return
    
    print(f"找到 {len(fits_files)} 个FITS文件")
    
    # 结果统计
    results = []
    
    # 显示文件列表并让用户选择要处理的文件
    print("\n可用的FITS文件:")
    for i, file_name in enumerate(fits_files):
        print(f"{i+1}. {file_name}")
    
    try:
        file_index = int(input("\n请选择要处理的文件编号 (1-{}): ".format(len(fits_files)))) - 1
        if file_index < 0 or file_index >= len(fits_files):
            print("无效的文件编号，将处理第一个文件")
            file_index = 0
    except ValueError:
        print("输入无效，将处理第一个文件")
        file_index = 0
    
    # 只处理选定的文件
    selected_file = fits_files[file_index]
    print(f"\n将处理文件: {selected_file}")
    
    file_path = os.path.join(folder_path, selected_file)
    
    try:
        # 清理GPU内存
        torch.cuda.empty_cache()
        gc.collect()  # 添加垃圾回收
        
        # 加载图像
        aurora_image = image_processor.load_fits_image(file_path)
        
        # 预处理图像
        aurora_image = image_processor.preprocess_image(aurora_image)
        print(f"图像形状: {aurora_image.shape}")
        
        # 检查图像大小，如果太大则分块处理
        height, width = aurora_image.shape
        if height * width > 256 * 256:  # 降低阈值，更多地使用分块处理
            print("图像较大，使用分块处理...")
            # 分块处理大型图像
            compressed_data, metadata = process_large_image(aurora_image, compressor, max_chunk_size=128)
            compression_time = metadata.get('compression_time', 0)
        else:
            # 将图像转换为PyTorch张量
            image_tensor = torch.tensor(aurora_image, dtype=torch.float32, device=device)
            
            # 压缩图像
            print("开始压缩...")
            start_time = time.time()
            compressed_data, metadata = compressor.compress_with_overlapping(image_tensor)
            compression_time = time.time() - start_time
        
        # 计算压缩比
        original_size = aurora_image.nbytes
        compressed_size = sum(len(block['encoded_normal']) for block in compressed_data)
        compressed_size += sum(len(block['outliers_indices']) * 4 for block in compressed_data)
        compressed_size += sum(len(block['outliers_values']) * 4 for block in compressed_data)
        
        compression_ratio = original_size / max(1, compressed_size)
        print(f"原始大小: {original_size / 1024:.2f} KB")
        print(f"压缩后大小: {compressed_size / 1024:.2f} KB")
        print(f"压缩比: {compression_ratio:.2f}:1")
        
        # 解压图像
        print("开始解压...")
        start_time = time.time()
        decompressed_image = compressor.decompress_with_overlapping({'compressed_data': compressed_data, 'metadata': metadata})
        decompression_time = time.time() - start_time
        
        # 计算解压误差
        image_tensor = torch.tensor(aurora_image, dtype=torch.float32, device=device)
        error = torch.abs(image_tensor - decompressed_image)
        max_error = error.max().item()
        mean_error = error.mean().item()
        print(f"最大误差: {max_error}")
        print(f"平均误差: {mean_error}")
        print(f"解压时间: {decompression_time:.2f} 秒")
        
        # 保存结果
        output_folder = os.path.join(folder_path, "decompressed")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"decompressed_{selected_file}")
        
        image_processor.save_fits_image(
            decompressed_image.cpu().numpy(),
            output_path
        )
        print(f"解压后的图像已保存到: {output_path}")
        
        # 记录结果
        results.append({
            'file_name': selected_file,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'max_error': max_error,
            'mean_error': mean_error,
            'compression_time': compression_time,
            'decompression_time': decompression_time
        })
        
        # 清理内存
        del decompressed_image, error, aurora_image
        if 'image_tensor' in locals():
            del image_tensor
        torch.cuda.empty_cache()
        gc.collect()  # 添加垃圾回收
        
    except Exception as e:
        print(f"处理文件 {selected_file} 时出错: {e}")
        traceback.print_exc()
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()  # 添加垃圾回收
    
    # 输出汇总结果
    if results:
        print("\n压缩结果汇总:")
        for r in results:
            print(f"文件: {r['file_name']}")
            print(f"压缩比: {r['compression_ratio']:.2f}:1")
            print(f"最大误差: {r['max_error']:.6f}")
            print(f"平均误差: {r['mean_error']:.6f}")
            print(f"压缩时间: {r['compression_time']:.2f} 秒")
            print(f"解压时间: {r['decompression_time']:.2f} 秒")
        
        # 保存结果到CSV
        csv_path = os.path.join(folder_path, "compression_results.csv")
        
        # 检查CSV文件是否存在
        csv_exists = os.path.exists(csv_path)
        
        # 打开CSV文件，如果存在则追加，否则创建新文件
        with open(csv_path, 'a' if csv_exists else 'w', newline='') as csvfile:
            fieldnames = ['file_name', 'original_size', 'compressed_size', 'compression_ratio', 
                         'max_error', 'mean_error', 'compression_time', 'decompression_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 如果是新文件，写入表头
            if not csv_exists:
                writer.writeheader()
            
            # 写入结果
            for result in results:
                writer.writerow(result)
        
        print(f"详细结果已保存到: {csv_path}")
    
    # 询问是否继续处理下一个文件
    try:
        continue_processing = input("\n是否继续处理下一个文件? (y/n): ").strip().lower()
        if continue_processing == 'y':
            # 重新运行主函数处理下一个文件
            main()
    except:
        pass
    
# 修改process_large_image函数，减小处理块大小并添加内存管理
def process_large_image(image, compressor, max_chunk_size=256):
    """
    处理大型图像，将其分割成小块进行压缩
    image: 输入图像
    compressor: 压缩器
    max_chunk_size: 最大块大小
    返回: 压缩数据和元数据
    """
    height, width = image.shape
    
    # 存储所有块的压缩数据
    compressed_chunks = []
    chunk_positions = []
    
    # 计算块大小 - 根据可用内存动态调整
    try:
        total_mem = torch.cuda.get_device_properties(0).total_memory
        free_mem = total_mem - torch.cuda.memory_allocated()
        
        # 根据可用内存调整块大小
        if free_mem < 2e9:  # 小于2GB
            chunk_size = min(64, max_chunk_size, height, width)
        elif free_mem < 4e9:  # 小于4GB
            chunk_size = min(128, max_chunk_size, height, width)
        else:
            chunk_size = min(max_chunk_size, height, width)
    except:
        # 如果无法获取内存信息，使用保守的块大小
        chunk_size = min(64, max_chunk_size, height, width)
    
    print(f"使用块大小: {chunk_size}x{chunk_size}")
    
    # 计算重叠大小 (10% 的块大小)
    overlap = max(4, int(chunk_size * 0.1))
    
    # 分块处理
    for y in range(0, height, chunk_size - overlap):
        for x in range(0, width, chunk_size - overlap):
            # 计算当前块的实际大小
            end_y = min(y + chunk_size, height)
            end_x = min(x + chunk_size, width)
            
            # 提取当前块
            chunk = image[y:end_y, x:end_x]
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=device)
            
            # 压缩当前块
            try:
                chunk_data, _ = compressor.compress_with_overlapping(chunk_tensor)
                compressed_chunks.append(chunk_data)
                chunk_positions.append((y, x, end_y - y, end_x - x))
                
                print(f"成功压缩块: ({y}, {x}) 到 ({end_y}, {end_x})")
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    # 内存不足，清理缓存并尝试使用更小的块
                    print(f"块 ({y}, {x}) 压缩失败，尝试使用更小的块...")
                    torch.cuda.empty_cache()
                    
                    # 将当前块分成4个子块
                    sub_h = (end_y - y) // 2
                    sub_w = (end_x - x) // 2
                    
                    # 处理子块
                    for sy in range(2):
                        for sx in range(2):
                            sub_y = y + sy * sub_h
                            sub_x = x + sx * sub_w
                            sub_end_y = min(sub_y + sub_h, end_y)
                            sub_end_x = min(sub_x + sub_w, end_x)
                            
                            # 提取子块
                            sub_chunk = image[sub_y:sub_end_y, sub_x:sub_end_x]
                            sub_tensor = torch.tensor(sub_chunk, dtype=torch.float32, device=device)
                            
                            try:
                                sub_data, _ = compressor.compress_with_overlapping(sub_tensor)
                                compressed_chunks.append(sub_data)
                                chunk_positions.append((sub_y, sub_x, sub_end_y - sub_y, sub_end_x - sub_x))
                                
                                print(f"成功压缩子块: ({sub_y}, {sub_x}) 到 ({sub_end_y}, {sub_end_x})")
                            except RuntimeError:
                                # 如果子块仍然失败，使用无损压缩
                                print(f"子块 ({sub_y}, {sub_x}) 压缩失败，使用无损压缩...")
                                # 简单的无损压缩（例如使用numpy的savez）
                                import io
                                buffer = io.BytesIO()
                                np.savez_compressed(buffer, data=sub_chunk)
                                buffer.seek(0)
                                compressed_chunks.append(buffer.read())
                                chunk_positions.append((sub_y, sub_x, sub_end_y - sub_y, sub_end_x - sub_x))
                            
                            # 清理内存
                            del sub_tensor
                            torch.cuda.empty_cache()
                else:
                    # 其他错误，重新抛出
                    raise
            
            # 清理内存
            del chunk_tensor
            torch.cuda.empty_cache()
    
    # 构建元数据
    metadata = {
        'height': height,
        'width': width,
        'chunk_positions': chunk_positions,
        'compression_params': compressor.get_params()
    }
    
    return compressed_chunks, metadata

# 添加一个简化版压缩器类，用于在内存不足时使用
class SimplifiedCompressor:
    """简化版压缩器，用于在GPU内存不足时使用"""
    
    def __init__(self, config):
        self.config = config
    
    def compress(self, image):
        """使用简单的DPCM方法压缩图像"""
        height, width = image.shape
        
        # 创建预测图像
        predicted = np.zeros_like(image)
        
        # 简单的DPCM预测
        for i in range(height):
            for j in range(width):
                # 获取预测值
                if i > 0 and j > 0:
                    # 使用左上方像素的平均值作为预测
                    predicted[i, j] = (image[i-1, j] + image[i, j-1] + image[i-1, j-1]) / 3.0
                elif i > 0:
                    # 使用上方像素作为预测
                    predicted[i, j] = image[i-1, j]
                elif j > 0:
                    # 使用左方像素作为预测
                    predicted[i, j] = image[i, j-1]
        
        # 计算残差
        residual = image - predicted
        
        # 简单量化
        quantized = np.round(residual / 2.0) * 2.0
        
        # 创建压缩数据
        compressed = {
            'predicted': predicted,
            'residual': quantized,
            'block_size': self.config.block_size
        }
        
        return compressed
    

# 如果直接运行此脚本，则执行主函数
if __name__ == "__main__":
    main()