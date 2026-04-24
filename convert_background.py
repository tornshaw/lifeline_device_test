# convert_background.py
import base64
import os

def convert_image_to_base64(image_path):
    """将图片转换为base64编码"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return base64_data

def update_login_window(base64_data):
    """更新login_window.py中的图片数据"""
    # 读取原始文件
    with open('login_window.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换图片数据
    start_marker = '# 这是一个非常简单的示例图片的base64编码'
    end_marker = '        return base64.b64decode(sample_image_base64)'
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker) + len(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        # 构造新的图片数据部分
        new_part = f'''        # 这是一个非常简单的示例图片的base64编码
        # 您需要将实际的背景图片转换为base64编码
        sample_image_base64 = "{base64_data}"
        
        # 如果您有实际的背景图片文件，可以用以下代码转换：
        # with open('data/guipic.png', 'rb') as f:
        #     image_data = f.read()
        #     sample_image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return base64.b64decode(sample_image_base64)'''
        
        # 替换内容
        new_content = content[:start_idx] + new_part + content[end_idx:]
        
        # 写回文件
        with open('login_window.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("login_window.py 已更新图片数据")
    else:
        print("未找到替换标记")

if __name__ == "__main__":
    # 转换您的背景图片
    image_path = input("请输入背景图片路径 (data/guipic.png): ").strip()
    if not image_path:
        image_path = 'data/guipic.png'
    
    if os.path.exists(image_path):
        base64_data = convert_image_to_base64(image_path)
        update_login_window(base64_data)
        print("背景图片已成功嵌入到代码中！")
    else:
        print(f"图片文件 {image_path} 不存在！")