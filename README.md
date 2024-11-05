# YOLO11-报纸版面分析

此项目使用 YOLO11 模型对图像文件夹中的报纸版面进行检测，并将结果保存为 XML 格式。模型将检测出图像中的不同元素（如标题、文本、图片等），并输出包含每个检测对象类别、置信度和边界框坐标的 XML 文件。

## 文件说明

- `11testYOLO.py`：主代码文件，用于加载模型、处理图像并生成检测结果。
- `requirements.txt`：项目的依赖项列表，确保所有必要的包都已安装。
- `output/`：保存 XML 格式的检测结果的文件夹。
- `yolo11.pt`：YOLO11 预训练模型文件。

## 依赖项

在运行代码之前，请确保已安装以下依赖项，或者运行以下命令安装：

```bash
pip install -r requirements.txt
```
## 使用说明

1. **准备项目结构**：确保项目结构正确，并将模型文件 `yolo11.pt` 放在代码中指定的位置。

2. **打开编辑器**：在 Visual Studio Code 或其他代码编辑器中打开项目。

3. **运行代码**：使用终端运行以下命令：
   

   ```bash
   python3 11testYOLO.py
   ```
4. 代码将自动处理 test 文件夹中的所有图像文件，并将检测结果保存到 output/ 文件夹中，生成每个图像对应的 XML 文件。
   
## 代码结构

- **类别名称列表**：`class_names` 列表定义了报纸版面各个元素的类别名称，顺序必须与模型训练时一致。
- **`save_to_xml` 函数**：用于将检测结果保存为 XML 格式，输出至 `output/` 文件夹。
- **主流程**：遍历 `test` 文件夹中的所有图像文件，对每张图像运行对象检测并保存 XML 文件。
