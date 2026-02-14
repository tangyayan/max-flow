import cv2
from streamlit_drawable_canvas import st_canvas
import numpy as np
import streamlit as st

import maxflow_cpp
# import maxflow_bk_cpp

def predict_from_mask(img_array, image_data, extract_mode='full'):
    background_mask = image_data[..., 0]
    foreground_mask = image_data[..., 1]
    _, background_mask = cv2.threshold(background_mask, 100, 255, cv2.THRESH_BINARY)
    _, foreground_mask = cv2.threshold(foreground_mask, 100, 255, cv2.THRESH_BINARY)

    # st.write(img_array)

    # st.write(img_array.shape, background_mask.shape, foreground_mask.shape)
    if len(img_array.shape) == 3 and img_array.shape[-1] > 3:
        img_array = img_array[..., :3]
    
    if len(img_array.shape) == 2:
        # 灰度图像，转换为 3 通道
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # RGBA 图像，转换为 RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
        # 单通道图像，转换为 3 通道
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    height, width, chennels = img_array.shape
    background_id = height * width
    foreground_id = height * width + 1
    dx = [1,0,0,-1]
    dy = [0,1,-1,0]
    # 0~height*width-1, height*width -> background, height*width+1 -> foreground

    foreground_pixels = img_array[foreground_mask == 255]
    background_pixels = img_array[background_mask == 255]
    
    # st.sidebar.write(f"前景像素数: {len(foreground_pixels)}")
    # st.sidebar.write(f"背景像素数: {len(background_pixels)}")
    
    bins = 8  # 直方图的bin数量
    # 计算前景直方图
    if len(foreground_pixels) > 0:
        hist_foreground, _ = np.histogramdd(
            foreground_pixels, 
            bins=[bins, bins, bins],
            range=[[0, 256], [0, 256], [0, 256]]
        )
        hist_foreground = hist_foreground + 1  # 平滑处理，避免概率为0
        hist_foreground = hist_foreground / np.sum(hist_foreground)  # 归一化
    else:
        hist_foreground = np.ones((bins, bins, bins)) / (bins ** 3)
    
    # 计算背景直方图
    if len(background_pixels) > 0:
        hist_background, _ = np.histogramdd(
            background_pixels,
            bins=[bins, bins, bins],
            range=[[0, 256], [0, 256], [0, 256]]
        )
        hist_background = hist_background + 1  # 平滑处理
        hist_background = hist_background / np.sum(hist_background)  # 归一化
    else:
        hist_background = np.ones((bins, bins, bins)) / (bins ** 3)
    
    total_diff = 0.0
    count = 0
    for i in range(height):
        for j in range(width):
            for t in range(4):
                ni = i + dy[t]
                nj = j + dx[t]
                if 0 <= ni < height and 0 <= nj < width:
                    diff = np.sum((img_array[i, j].astype(np.float64) - 
                                        img_array[ni, nj].astype(np.float64)) ** 2)
                    total_diff += diff
                    count += 1

    avg_diff = total_diff / count if count > 0 else 1.0
    beta = 1.0 / (avg_diff + 1e-10) if avg_diff > 0 else 0.0
    
    # st.sidebar.write(f"Beta: {beta:.6f}")
    st.sidebar.write(f"平均颜色差异: {avg_diff:.2f}")

    max_flow_solver = maxflow_cpp.MaxFlow(height * width + 2)
    # max_flow_solver = maxflow_bk_cpp.MaxFlow_bk(height * width + 2)

    K = -1
    for i in range(height):
        for j in range(width):
            pixel_id = i * width + j
            pixel_color = img_array[i, j]

            K_sum = 0
            for t in range(2):
                ni = i + dy[t]
                nj = j + dx[t]
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_id = ni * width + nj
                    diff = np.sum((img_array[i, j] - img_array[ni, nj]) ** 2)
                    weight = np.exp(-beta * diff)
                    # graph[pixel_id].append((neighbor_id, weight))
                    max_flow_solver.add_edge(pixel_id, neighbor_id, weight, weight)
            for t in range(4):
                ni = i + dy[t]
                nj = j + dx[t]
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_id = ni * width + nj
                    diff = np.sum((img_array[i, j] - img_array[ni, nj]) ** 2)
                    weight = np.exp(-beta * diff)
                    K_sum += weight
            K = max(K, K_sum)
    K = K+1

    # st.write(f"平滑项最大权重 K: {K:.2f}")

    for i in range(height):
        for j in range(width):
            pixel_id = i * width + j
            pixel_color = img_array[i, j]

            bin_r = min(int(pixel_color[0]) * bins // 256, bins - 1)
            bin_g = min(int(pixel_color[1]) * bins // 256, bins - 1)
            bin_b = min(int(pixel_color[2]) * bins // 256, bins - 1)
            
            # 计算数据项: -ln(P(Ip|obj)) 和 -ln(P(Ip|bkg))
            prob_foreground = hist_foreground[bin_r, bin_g, bin_b]
            prob_background = hist_background[bin_r, bin_g, bin_b]

            data_cost_foreground = -np.log(prob_foreground + 1e-10)
            data_cost_background = -np.log(prob_background + 1e-10)

            if background_mask[i, j] == 255:
                # graph[background_id].append((pixel_id, K))
                # graph[foreground_id].append((pixel_id, 0))
                max_flow_solver.add_edge(background_id, pixel_id, K, 0)
                max_flow_solver.add_edge(pixel_id, foreground_id, 0, 0)
            elif foreground_mask[i, j] == 255:
                # graph[foreground_id].append((pixel_id, K))
                # graph[background_id].append((pixel_id, 0))
                max_flow_solver.add_edge(pixel_id,foreground_id, K, 0)
                max_flow_solver.add_edge(background_id, pixel_id, 0, 0)
            else:
                # graph[background_id].append((pixel_id, data_cost_foreground))
                # graph[foreground_id].append((pixel_id, data_cost_background))
                max_flow_solver.add_edge(background_id, pixel_id, data_cost_foreground, 0)
                max_flow_solver.add_edge(pixel_id, foreground_id, data_cost_background, 0)
    
    st.write("开始计算最大流...")

    value = max_flow_solver.max_flow(background_id, foreground_id)
    st.sidebar.write(f"最大流值: {value}")

    result_mask = max_flow_solver.get_cut(height, width, background_id, extract_mode=='full')
    # st.write(result_mask)
    masked_image = img_array.copy()
    masked_image[result_mask == 0] = [0, 0, 0]
    
    return masked_image, result_mask
    
