## 最大流最小割的图像分割

### 快速开始

```bash
python setup.py build_ext --inplace #编译cpp程序

cd image
python down.py original_img -o output_img -m max_dimension #图片太大用于压缩

streamlit run app.py
```

### 参考资料

Interactive graph cuts for optimal boundary & region segmentation of objects in N-D images

[(99+ 封私信 / 81 条消息) 基于 Graph Cut 的可交互式图像分割——原理与代码实现 - 知乎](https://zhuanlan.zhihu.com/p/671470269)
