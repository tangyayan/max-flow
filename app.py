from streamlit_drawable_canvas import st_canvas
from PIL import Image
import streamlit as st
import numpy as np

from main import predict_from_mask

st.set_page_config(
    page_title='交互式图像分割程序',
    page_icon=' ',
    layout='wide'
)

st.sidebar.title("Graph Cut 交互式前景分割程序")

extract_mode = st.sidebar.selectbox(
    "提取信息",
    ("整张图片", "边界"),
)
extract_mode_mapping = {
    '整张图片': 'full',
    '边界': 'border'
}
extract_mode = extract_mode_mapping[extract_mode]

drawing_region, display_region = st.columns(spec=2)

# 感谢 andfanilo 贡献的绘画组件 streamlit_drawable_canvas
# 论坛链接： https://discuss.streamlit.io/t/drawable-canvas/3671
tagging_mode = st.sidebar.selectbox(
    "标记模式:",
    ("mask 先验",),
)

drawing_mode_mapping = {
    'mask 先验': 'freedraw'
}

drawing_mode = drawing_mode_mapping[tagging_mode]

stroke_width = st.sidebar.slider("画笔宽度: ", 1, 100, 15)
if drawing_mode == "point":
    point_display_radius = st.sidebar.slider("点半径: ", 1, 25, 3)

# graph_cut_iterations = st.sidebar.slider('网络流迭代次数: ', 2, 30, 5)
if drawing_mode == 'rect':
    stroke_color = '#4dc114'
else:
    stroke_color = st.sidebar.selectbox(
        "标记区域",
        ('前景', '背景')
    )
    stroke_color_mapping = {
        '前景': '#4dc114',
        '背景': '#ff4b4b'
    }
    stroke_color = stroke_color_mapping[stroke_color]

bg_image = st.sidebar.file_uploader("上传需要标记的图像: ", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("实时更新", False)

if bg_image:
    img = Image.open(bg_image)
    img = img.convert('RGB')  # 确保图像是 RGB 模式
    img_array = np.array(img)
    canvas_height, canvas_width = img_array.shape[:2]
    with drawing_region:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color='#eee',
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius
            if drawing_mode == "point" else 0,
            display_toolbar=True,
            key="full_app",
        )
    if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
        if drawing_mode == 'freedraw':
            test_image = np.array([
                [[50, 50, 50],    [200, 200, 50],  [200, 200, 200]],
                [[50, 50, 50],    [50, 50, 70],  [200, 200, 200]],
                [[50, 50, 50],    [100, 100, 200],  [200, 200, 200]]
            ], dtype=np.uint8)
            canvas_data = np.zeros((3, 3, 4), dtype=np.uint8)
            canvas_data[0, 0] = [255, 0, 0, 255]     # 背景标注（左上）
            canvas_data[2, 2] = [0, 255, 0, 255] 
            masked_image, mask = predict_from_mask(img_array, canvas_result.image_data, extract_mode)         
            # masked_image, mask = predict_from_mask(test_image, canvas_data, 5)
            display_region.image(masked_image)