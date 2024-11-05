from ultralytics import YOLO
import os
import time
import xml.etree.ElementTree as ET

# 定义类别名称列表，确保顺序与模型训练时一致
class_names = ["Header", "Title", "Text", "Figure", "Foot"]

def save_to_xml(image_name, results, output_folder="..\output"):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 创建 XML 根元素
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_name

    for result in results:
        # 创建对象元素
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = result["class_name"]
        ET.SubElement(obj, "confidence").text = str(result["confidence"])

        # 创建边界框元素
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(result["bbox"][0])
        ET.SubElement(bbox, "ymin").text = str(result["bbox"][1])
        ET.SubElement(bbox, "xmax").text = str(result["bbox"][2])
        ET.SubElement(bbox, "ymax").text = str(result["bbox"][3])

    # 将 XML 保存到文件
    xml_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.xml")
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved XML result to: {xml_path}")

if __name__ == '__main__':
    # 开始计时
    start_time = time.time()

    # 加载训练好的模型
    model = YOLO(r'..\yolo11.pt')

    # 设置要检测的图像文件夹路径
    image_folder = "test"  # 替换为你要检测的图像文件夹路径

    # 遍历文件夹中的所有图像文件
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        # 确保只处理图像文件
        if image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            print(f"Processing image: {image_path}")

            # 在图像上执行对象检测
            results = model(image_path)

            # 将检测结果提取为字典格式
            result_data = []
            for result in results[0].boxes.data:  # 获取检测结果
                x_min, y_min, x_max, y_max = map(int, result[:4])
                confidence = float(result[4])
                class_id = int(result[5])

                # 使用预定义的类别名称
                class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

                result_data.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x_min, y_min, x_max, y_max]
                })

            # 保存为 XML 文件
            save_to_xml(image_name, result_data)

    # 结束计时
    end_time = time.time()

    # 计算并打印总时间
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
