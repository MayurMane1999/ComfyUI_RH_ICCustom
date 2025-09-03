# ComfyUI IC-Custom 节点

ComfyUI的自定义节点，集成IC-Custom模型，用于高质量图像定制和生成。

## ✨ 功能特性

- 🎨 **高质量图像生成**：基于FLUX.1-Fill-dev和IC-Custom模型
- 🖼️ **图像定制**：基于参考图像生成定制化图像
- 🎯 **灵活生成模式**：支持位置无关和精确位置生成
- ⚙️ **高级控制**：可配置引导比例、推理步数和种子控制
- 🚀 **性能优化**：模型量化和卸载，更好的内存效率

## 📦 安装说明

### 第一步：安装节点

```bash
# 进入ComfyUI自定义节点目录
cd ComfyUI/custom_nodes

# 克隆仓库
git clone https://github.com/HM-RunningHub/ComfyUI_RH_ICCustom

# 安装依赖
cd ComfyUI_RH_ICCustom
pip install -r requirements.txt
```

### 第二步：下载必需模型

在ComfyUI模型文件夹中创建以下目录结构：

#### 主要模型

**FLUX.1-Fill-dev 模型：**
- 下载地址：[FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/tree/main)
- 文件：`ae.safetensors`、`flux1-fill-dev.safetensors`
- 存放位置：`ComfyUI/models/black-forest-labs/FLUX.1-Fill-dev/`

**IC-Custom 模型：**
- 下载地址：[IC-Custom](https://huggingface.co/TencentARC/IC-Custom/tree/main)
- 文件：仓库中的所有文件
- 存放位置：`ComfyUI/models/IC-Custom/`

**FLUX Redux 模型：**
- 下载地址：[FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/tree/main)
- 文件：`flux1-redux-dev.safetensors`
- 存放位置：`ComfyUI/models/IC-Custom/`

#### CLIP 模型

**SigLIP 模型：**
- 下载地址：[siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384/tree/main)
- 文件：仓库中的所有文件
- 存放位置：`ComfyUI/models/clip/siglip-so400m-patch14-384/`

**XFlux 文本编码器：**
- 下载地址：[xflux_text_encoders](https://huggingface.co/XLabs-AI/xflux_text_encoders/tree/main)
- 文件：仓库中的所有文件
- 存放位置：`ComfyUI/models/clip/xflux_text_encoders/`

#### CLIP Vision 模型

**CLIP ViT Large：**
- 下载地址：[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)
- 文件：仓库中的所有文件
- 存放位置：`ComfyUI/models/clip_vision/clip-vit-large-patch14/`

## 🚀 使用方法

### 基础工作流

1. **添加模型加载器**：在工作流中添加"RunningHub ICCustom Loader"节点
2. **添加采样器**：添加"RunningHub ICCustom Sampler"节点并连接管道输出
3. **配置输入**：
   - 连接参考图像
   - 设置提示文本
   - 配置生成参数
   - 可选择添加目标图像和蒙版以进行精确控制

### 示例工作流

```
[参考图像] → [ICCustom 加载器] → [ICCustom 采样器] → [保存图像]
                                    ↓
                               [提示输入]
```

### 生成模式

- **位置无关**：不受目标约束生成（无需蒙版）
- **精确位置**：在特定目标位置生成（需要蒙版）

## ⚙️ 参数说明

- **提示词**：生成内容的文本描述
- **引导比例**：控制对提示词的遵循程度（默认：40.0）
- **真实引导**：额外的引导参数（默认：3.0）
- **推理步数**：推理步骤数量（默认：25）
- **种子**：用于可重现结果的随机种子

## 🔧 系统需求

- **显存**：推荐8GB+显存
- **内存**：推荐16GB+系统内存
- **存储空间**：所有模型约需25GB
- **依赖项**：PyTorch、Diffusers、Transformers

## 📄 许可证

本项目采用Apache 2.0许可证。

## 🔗 相关链接

- [IC-Custom](https://github.com/TencentARC/IC-Custom)
- [FLUX 模型](https://huggingface.co/black-forest-labs)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## 🙏 致谢

特别感谢 **AIwood爱屋研究室** ([B站](https://space.bilibili.com/503934057)) 帮助完成Windows环境的测试，并撰写部分安装说明。

## 🤝 贡献

欢迎贡献！请随时提交问题和拉取请求。
